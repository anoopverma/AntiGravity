import os
import time
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dhanhq import dhanhq
from scipy.stats import norm
from sqlalchemy import create_engine

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

class OptionFetcher:
    def __init__(self, access_token):
        self.url = "https://api.dhan.co/v2/charts/rollingoption"
        self.headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": access_token
        }
        self.cache = {}
        
    def fetch(self, date_str, opt_type, rel_strike=0):
        key = (date_str, opt_type, rel_strike)
        if key in self.cache: return self.cache[key]
        
        strike_str = "ATM"
        if rel_strike > 0: strike_str = f"ATM+{rel_strike}"
        if rel_strike < 0: strike_str = f"ATM{rel_strike}"
        
        payload = {
            "exchangeSegment": "NSE_FNO",
            "interval": "5",
            "securityId": "13",
            "instrument": "OPTIDX",
            "expiryFlag": "WEEK",
            "expiryCode": 1,
            "strike": strike_str,
            "drvOptionType": "CALL" if opt_type == 'C' else "PUT",
            "requiredData": ["open", "high", "low", "close", "volume", "strike"],
            "fromDate": date_str,
            "toDate": date_str
        }
        for attempt in range(3):
            try:
                res = requests.post(self.url, json=payload, headers=self.headers)
                if res.status_code == 429:
                    time.sleep(1) # Wait if rate limited
                    continue
                res.raise_for_status()
                data = res.json()
                series = data.get("data", {}).get("ce" if opt_type == 'C' else "pe", {})
                
                result = {}
                if series and series.get("timestamp"):
                    for i, ts in enumerate(series["timestamp"]):
                        dt = pd.to_datetime(ts, unit='s') + pd.Timedelta(hours=5, minutes=30)
                        result[dt] = {
                            "close": series["close"][i],
                            "strike": series["strike"][i]
                        }
                self.cache[key] = result
                return result
            except Exception as e:
                if attempt == 2:
                    logger.error(f"Option fetch failed: {e}")
                time.sleep(0.5)
        return {}


# Constants for Black-Scholes Approximation
RISK_FREE_RATE = 0.1  
IMPLIED_VOL_ASSUMPTION = 0.15  

class V4Backtester:
    def __init__(self):
        load_dotenv()
        # Support both standard and Render env var names
        self.client_id = os.getenv('DHAN_CLIENT_ID') or os.getenv('DHAN_API_KEY')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN') or os.getenv('DHAN_CLIENT_SECRET')
        
        if not self.client_id or not self.access_token:
            raise ValueError("Dhan API credentials (ID/Key or Token/Secret) not found in environment")
            
        from dhanhq.dhan_context import DhanContext
        context = DhanContext(str(self.client_id), str(self.access_token))
        self.dhan = dhanhq(context)
        
        # Strategy Parameters
        self.initial_capital = 500000
        self.current_capital = self.initial_capital
        self.lot_size = 25
        self.results = []
        self.cached_data = {} 
        self.option_fetcher = OptionFetcher(str(self.access_token))
        
    def get_real_option_price(self, date_str, index_dt, target_strike, opt_type):
        """Fetch real options data from Dhan rolling API."""
        dt_key = index_dt.replace(second=0, microsecond=0)
        base_data = self.option_fetcher.fetch(date_str, opt_type, 0)
        if not base_data or dt_key not in base_data:
            return None
        
        current_atm_strike = base_data[dt_key]['strike']
        rel_strike = int(round((target_strike - current_atm_strike) / 50))
        
        rel_data = self.option_fetcher.fetch(date_str, opt_type, rel_strike)
        if rel_data and dt_key in rel_data:
            return rel_data[dt_key]['close']
        return None

    def get_last_n_nifty_expiries(self, n=48):
        """
        Return the last N Nifty 50 weekly expiry dates.
        Nifty 50 switched from Thursday expiry → Tuesday expiry from Apr 4, 2024 onwards.
        Before Apr 4 2024 → Thursdays (weekday=3)
        From Apr 4 2024  → Tuesdays  (weekday=1)
        """
        SWITCH_DATE = datetime(2024, 4, 4)   # official switchover date
        today = datetime.now()
        expiries = []

        # Walk backwards week by week, picking correct expiry day
        d = today
        while len(expiries) < n:
            if d >= SWITCH_DATE:
                # Find most recent Tuesday on or before d
                days_since_tue = (d.weekday() - 1) % 7
                expiry = d - timedelta(days=days_since_tue)
            else:
                # Find most recent Thursday on or before d
                days_since_thu = (d.weekday() - 3) % 7
                expiry = d - timedelta(days=days_since_thu)

            expiry_str = expiry.strftime("%Y-%m-%d")
            if expiry_str not in expiries and expiry.date() < today.date():
                expiries.append(expiry_str)
            d -= timedelta(days=7)

        return sorted(expiries)

    def fetch_yf_5min_fallback(self, date_str):
        try:
            import yfinance as yf
            ticker = yf.Ticker("^NSEI")
            start_date = date_str
            end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = ticker.history(start=start_date, end=end_date, interval="5m")
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                logger.info(f"Fallback: Fetched {len(df)} bars from YFinance for {date_str}")
                return df
        except Exception as e:
            logger.error(f"YFinance fallback failed for {date_str}: {e}")
        return pd.DataFrame()

    def fetch_dhan_5min_data(self, date_str, retries=3):
        for attempt in range(retries):
            try:
                req = self.dhan.intraday_minute_data(
                    security_id='13', exchange_segment=self.dhan.INDEX,
                    instrument_type='INDEX', from_date=date_str, to_date=date_str
                )
                
                if req.get('status') == 'success' and req.get('data'):
                    df = pd.DataFrame(req['data'])
                    if df.empty: return pd.DataFrame()
                        
                    time_col = 'timestamp' if 'timestamp' in df.columns else 'start_Time'
                    
                    if time_col:
                        if time_col == 'timestamp':
                            df['datetime'] = pd.to_datetime(df[time_col], unit='s') + pd.Timedelta(hours=5, minutes=30)
                        else:
                            df['datetime'] = pd.to_datetime(df[time_col]) + pd.Timedelta(hours=5, minutes=30)
                        
                        df.set_index('datetime', inplace=True)
                        df_5m = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
                        logger.info(f"Fetched {len(df_5m)} 5-min bars for {date_str}")
                        return df_5m
                elif req.get('remarks') and 'DH-904' in str(req.get('remarks')):
                    time.sleep(5)
                else:
                    break
            except Exception as e:
                break
        return self.fetch_yf_5min_fallback(date_str)

    def run_v4_backtest(self, vix_threshold=12.5, target_lock_in=0.20, trailing_step=0.15):
        tuesdays = self.get_last_n_nifty_expiries(48)
        logger.info(f"Starting V4 Trailing SL Strategy Backtest for Last 48 Nifty Expiry Days")
        self.results = []
        self.current_capital = self.initial_capital
        
        # Build a readable parameters string for DB logging
        entry_expansion = 1.25   # combined straddle must expand 25% from 1:30 PM baseline
        min_premium     = 5.0    # min option premium to enter (avoids junk OTM entries)
        entry_cutoff    = "15:01" # no new entries after this time
        initial_sl_pct  = 0.35   # 35% max loss on entry price before initial SL
        params_str = (
            f"lock_in={target_lock_in*100:.0f}%|"
            f"trail={trailing_step*100:.0f}%|"
            f"init_sl={initial_sl_pct*100:.0f}%|"
            f"expansion={entry_expansion*100:.0f}%|"
            f"min_prem={min_premium}|"
            f"cutoff={entry_cutoff}|"
            f"vix_thresh={vix_threshold}"
        )
        
        # Simple VIX mock > 12.5 assumption for backtesting past history
        current_vix = 13.0 
        
        for date_str in tuesdays:
            df = self.cached_data.get(date_str)
            if df is None:
                df = self.fetch_dhan_5min_data(date_str)
                self.cached_data[date_str] = df
            if df.empty: continue
            
            position = None
            benchmark_straddle = None
            benchmark_spot     = None
            entry_time, entry_spot, entry_strike, entry_premium = None, 0, 0, 0
            entry_qty = 0
            eod_time = df.index[-1].replace(hour=15, minute=30)

            # ATM strikes recorded at 1:30 PM
            atm_ce_130, atm_pe_130 = None, None
            
            for index, row in df.iterrows():
                spot_price = row['close']
                dte = (eod_time - index).total_seconds() / (365 * 24 * 3600)
                
                # 1. Capture Benchmark (1:30 PM)
                if index.hour == 13 and index.minute == 30:
                    benchmark_spot = spot_price
                    atm_strike = round(spot_price / 50) * 50
                    atm_ce_130 = atm_strike
                    atm_pe_130 = atm_strike
                    ce_price = self.get_real_option_price(date_str, index, atm_ce_130, 'C')
                    pe_price = self.get_real_option_price(date_str, index, atm_pe_130, 'P')
                    
                    if ce_price and pe_price:
                        benchmark_straddle = ce_price + pe_price
                    #logger.info(f"[{date_str}] 1:30 PM Benchmark Straddle: {benchmark_straddle}")

                # 2. Monitor for Entry (between 13:30 and 15:01, single trade per day)
                if benchmark_straddle and position is None and index.time() > datetime.strptime("13:30", "%H:%M").time() and index.time() <= datetime.strptime("15:01", "%H:%M").time():
                    curr_ce = self.get_real_option_price(date_str, index, atm_ce_130, 'C')
                    curr_pe = self.get_real_option_price(date_str, index, atm_pe_130, 'P')
                    
                    if curr_ce and curr_pe:
                        curr_straddle = curr_ce + curr_pe
                        # Entry filter: combined straddle must expand >=25% AND strong directional move
                        spot_move_pct = abs(spot_price - benchmark_spot) / benchmark_spot
                        if curr_straddle >= (benchmark_straddle * entry_expansion) and current_vix >= vix_threshold and spot_move_pct >= 0.003:
                            # Emulate Volume/Direction check using Point Differential
                            opt_type = 'C' if spot_price > benchmark_spot else 'P'
                            target_strike = round(spot_price / 50) * 50 # New ATM
                            entry_premium = self.get_real_option_price(date_str, index, target_strike, opt_type)
                            
                            # Minimum premium filter — avoid entering near-worthless OTM options
                            if entry_premium and entry_premium >= min_premium:
                                position = {
                                    'type': opt_type, 
                                    'strike': target_strike,
                                    'entry': entry_premium, 
                                    'peak': entry_premium,
                                    'qty': 1300
                                }
                                entry_time, entry_spot = index, spot_price
                                #logger.info(f"[{date_str}] ENTRY {opt_type} @ {entry_premium:.2f} | params: {params_str}")

                # 3. Handle Active Position (Trailing SL / Time Exit)
                elif position:
                    current_price = self.get_real_option_price(date_str, index, position['strike'], position['type'])
                    if current_price is None:
                        continue
                        
                    if current_price > position['peak']:
                        position['peak'] = current_price
                        
                    if position['peak'] >= position['entry'] * (1 + target_lock_in):
                        current_sl = position['peak'] * (1 - trailing_step)
                        reason = "Trailing SL"
                    else:
                        current_sl = position['entry'] * 0.65  # 35% max initial loss
                        reason = "Initial SL"
                        
                    time_exit = index.hour == 15 and index.minute >= 25
                    
                    if current_price <= current_sl or time_exit:
                        reason = "Time Exit" if time_exit else reason
                        pnl_points = current_price - position['entry']
                        pnl_inr = pnl_points * position['qty']
                        self.current_capital += pnl_inr
                        
                        capital_before = self.current_capital - pnl_inr
                        self.results.append({
                            'Date': date_str,
                            'Entry_Time': entry_time.strftime("%H:%M:%S"),
                            'Exit_Time': index.strftime("%H:%M:%S"),
                            'Option_Type': position['type'],
                            'Strike': f"{int(position['strike'])}-{date_str}-{position['type']}E",
                            'Action': 'BUY',
                            'Qty': position['qty'],
                            'Buy_Price': round(position['entry'], 2),
                            'Peak_Price': round(position['peak'], 2),
                            'Sell_Price': round(current_price, 2),
                            'PNL': round(pnl_inr, 2),
                            'ROI%': round((pnl_points / position['entry']) * 100, 2),
                            'Capital_ROI%': round((pnl_inr / capital_before) * 100, 2) if capital_before > 0 else 0,
                            'Reason': reason,
                            'Win': pnl_inr > 0,
                            'Parameters': params_str
                        })
                        #logger.info(f"[{date_str}] EXIT via {reason} @ {current_price:.2f} | PnL: ₹{pnl_inr:.2f}")
                        position = None
                        break  # Single trade per day

            time.sleep(0.1)
        
        self.print_summary()
        self.save_to_postgres(table_name="historical_backtests", strategy_name="v4_trailing_sl")

    def print_summary(self):
        if not self.results:
            print("\nNo Trades Executed in this Period.")
            return
            
        df = pd.DataFrame(self.results)
        df['Win'] = df['PNL'] > 0
        total_pnl = df['PNL'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        
        print("\n=== V4 STRATEGY 48-WEEK BACKTEST RESULTS ===")
        print(f"Total Return: {roi:.2f}% (₹{total_pnl:,.2f})")
        print(f"Total Trades: {len(df)}")
        print(f"Win Rate: {(df['Win'].sum() / len(df)) * 100:.2f}%\n")
        print(df.to_string(index=False))

    def save_to_postgres(self, table_name="historical_backtests", strategy_name="v4_trailing_sl"):
        uri = os.getenv("POSTGRES_URI", "postgresql://postgres:Aidni%40%23123@localhost:5432/postgres")
        if not self.results: return
        try:
            engine = create_engine(uri)
            df = pd.DataFrame(self.results)
            df.insert(0, 'Run_Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            df.insert(1, 'Strategy_Name', strategy_name)
            df.insert(2, 'Run_Mode', 'Backtest')
            
            try:
                existing = pd.read_sql(
                    f"SELECT * FROM {table_name} WHERE \"Strategy_Name\" != '{strategy_name}'",
                    con=engine)
                combined = pd.concat([existing, df], ignore_index=True)
            except Exception:
                # Table does not exist or first run
                combined = df
                
            combined.to_sql(table_name, con=engine, if_exists='replace', index=False)
            print(f"-> DB SYNC: {strategy_name} results saved to {table_name}. Total rows: {len(combined)}")
        except Exception as e:
            logger.error(f"DB Error: {e}")

if __name__ == "__main__":
    backtester = V4Backtester()
    backtester.run_v4_backtest()
