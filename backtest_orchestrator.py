"""
backtest_orchestrator.py — Unified backtest executor with Salesforce integration

Orchestrates execution of multiple strategies and handles:
- Strategy selection and execution
- Salesforce data insertion
- Result aggregation and logging
"""

import os
import sys
import logging
from datetime import datetime
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from simple_salesforce import Salesforce

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SF_BACKTEST_OBJECT = "historical_backtests__c"
IST_TZ = ZoneInfo("Asia/Kolkata")
UTC_TZ = ZoneInfo("UTC")


class BacktestOrchestrator:
    """Manages strategy execution and Salesforce sync."""

    def __init__(self):
        self.sf = None
        self.results = []
        self.strategy_name = None

    def connect_salesforce(self):
        """Initialize Salesforce connection."""
        if self.sf:
            return self.sf
        try:
            self.sf = Salesforce(
                username=os.getenv("SF_USERNAME"),
                password=os.getenv("SF_PASSWORD"),
                security_token=os.getenv("SF_SECURITY_TOKEN", ""),
                domain=os.getenv("SF_DOMAIN", "login"),
                version=os.getenv("SF_API_VERSION", "59.0"),
            )
            logger.info("Connected to Salesforce")
            return self.sf
        except Exception as exc:
            logger.error("Salesforce login failed: %s", exc)
            return None

    def run_strategy(self, strategy_name, **kwargs):
        """
        Execute a selected strategy.
        
        Args:
            strategy_name: 'gamma_blast' or 'v4_gamma'
            **kwargs: Additional parameters for backtest configuration
        """
        self.strategy_name = strategy_name
        logger.info(f"Starting {strategy_name} backtest...")

        try:
            if strategy_name == "gamma_blast":
                return self._run_gamma_blast()
            elif strategy_name == "v4_gamma":
                return self._run_v4_gamma(**kwargs)
            else:
                logger.error(f"Unknown strategy: {strategy_name}")
                return {"error": f"Unknown strategy: {strategy_name}"}
        except Exception as e:
            logger.error(f"Strategy execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _run_gamma_blast(self):
        """Execute Gamma Blast (gamma spike sell) strategy."""
        try:
            from backtest_gamma import GammaSpikeBacktester
            bt = GammaSpikeBacktester()
            bt.run()
            self.results = bt.results
            logger.info(f"Gamma Blast backtest completed: {len(self.results)} trades")
            return {
                "status": "success",
                "strategy": "gamma_blast",
                "total_trades": len(self.results),
                "message": "Gamma Blast backtest completed and synced to Salesforce"
            }
        except Exception as e:
            logger.error(f"Gamma Blast execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    def _run_v4_gamma(self, **kwargs):
        """Execute V4 Gamma (trailing SL) strategy."""
        try:
            from backtest_v4 import V4Backtester
            bt = V4Backtester()
            
            # Apply any custom parameters if provided
            if 'start_date' in kwargs:
                bt.start_date = kwargs['start_date']
            if 'end_date' in kwargs:
                bt.end_date = kwargs['end_date']
            if 'capital' in kwargs:
                bt.initial_capital = kwargs['capital']
            
            bt.run_v4_backtest()
            # V4 saves to DB directly, but we'll also sync to Salesforce if results exist
            self.results = bt.results if hasattr(bt, 'results') else []
            logger.info(f"V4 Gamma backtest completed: {len(self.results)} trades")
            return {
                "status": "success",
                "strategy": "v4_gamma",
                "total_trades": len(self.results),
                "message": "V4 Gamma backtest completed"
            }
        except Exception as e:
            logger.error(f"V4 Gamma execution failed: {e}", exc_info=True)
            return {"error": str(e)}

    def sync_to_salesforce(self, results=None, strategy_name=None):
        """
        Sync backtest results to Salesforce.
        
        Args:
            results: List of trade result dicts (if None, uses self.results)
            strategy_name: Strategy name for Salesforce (if None, uses self.strategy_name)
        """
        if results is None:
            results = self.results
        if strategy_name is None:
            strategy_name = self.strategy_name

        if not results:
            logger.warning("No results to sync")
            return {"status": "warning", "message": "No results to sync"}

        sf = self.connect_salesforce()
        if not sf:
            return {"status": "error", "message": "Salesforce connection failed"}

        run_ts = datetime.utcnow().strftime("%Y-%m-%d")

        def _to_sf_datetime(date_str, time_str):
            """Convert IST time to UTC datetime for Salesforce."""
            local_dt = datetime.strptime(
                f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=IST_TZ)
            utc_dt = local_dt.astimezone(UTC_TZ)
            return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Build trade records
        trade_records = []
        for r in results:
            record = {
                "Run_Date__c": run_ts,
                "Strategy_Name__c": strategy_name,
                "Trade_Date__c": r.get("Date", ""),
                "Entry_Time__c": _to_sf_datetime(r.get("Date", ""), r.get("Entry_Time", "09:30:00")),
                "Exit_Time__c": _to_sf_datetime(r.get("Date", ""), r.get("Exit_Time", "15:30:00")),
                "Option_Type__c": r.get("Option_Type", ""),
                "Action__c": r.get("Action", ""),
                "Qty__c": r.get("Qty", 0),
                "Buy_Price__c": float(r.get("Buy_Price", 0)) if r.get("Buy_Price") else None,
                "Peak_Price__c": float(r.get("Peak_Price", 0)) if r.get("Peak_Price") else None,
                "Sell_Price__c": float(r.get("Sell_Price", 0)) if r.get("Sell_Price") else None,
                "Reason__c": r.get("Reason", ""),
                "Win__c": r.get("Win", False),
                "Total_PNL__c": float(r.get("PNL", 0)) if r.get("PNL") else None,
                "Total_Return_Percentage__c": float(r.get("ROI%", 0)) if r.get("ROI%") else None,
                "Capital_ROI_Pct__c": float(r.get("Capital_ROI%", 0)) if r.get("Capital_ROI%") else None,
                "Run_Mode__c": "backtest",
                "Strike__c": str(r.get("Strike", "")),
                "PnL_INR__c": float(r.get("PNL", 0)) if r.get("PNL") else None,
                "Parameters__c": str(r.get("Parameters", "")),
            }
            trade_records.append(record)

        try:
            bulk_backtests = getattr(sf.bulk, SF_BACKTEST_OBJECT)
            res = bulk_backtests.insert(trade_records)
            ok = sum(1 for r in res if r.get("success"))
            fail = sum(1 for r in res if not r.get("success"))
            
            logger.info(
                "-> SF SYNC: %d trade rows saved to %s (%d failed).",
                ok, SF_BACKTEST_OBJECT, fail,
            )
            
            if fail:
                bad = [r for r in res if not r.get("success")]
                logger.warning("First error: %s", bad[0].get("errors", ""))
                return {
                    "status": "partial",
                    "message": f"{ok}/{len(trade_records)} records inserted",
                    "errors": bad[0].get("errors")
                }
            
            return {
                "status": "success",
                "message": f"All {ok} records inserted successfully",
                "inserted": ok
            }
        except Exception as exc:
            logger.error("Salesforce save failed: %s", exc)
            return {"status": "error", "message": str(exc)}

    def run_and_sync(self, strategy_name, **kwargs):
        """
        Execute strategy and sync results to Salesforce in one call.
        
        Args:
            strategy_name: 'gamma_blast' or 'v4_gamma'
            **kwargs: Additional backtest parameters
            
        Returns:
            dict with execution and sync results
        """
        exec_result = self.run_strategy(strategy_name, **kwargs)
        
        if "error" in exec_result:
            return exec_result
        
        sync_result = self.sync_to_salesforce(self.results, strategy_name)
        
        return {
            "execution": exec_result,
            "sync": sync_result,
            "overall_status": "success" if sync_result.get("status") == "success" else "partial"
        }


if __name__ == "__main__":
    # Test orchestrator
    import json
    
    orchestrator = BacktestOrchestrator()
    
    # Test strategy selection
    strategy = sys.argv[1] if len(sys.argv) > 1 else "gamma_blast"
    
    result = orchestrator.run_and_sync(strategy)
    print(json.dumps(result, indent=2))
