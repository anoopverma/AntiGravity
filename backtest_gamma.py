"""
backtest_gamma.py — Gamma Spike SELL Strategy Backtest
Morning session: benchmark at 9:20 AM, entry window 9:30–11:30 AM.
SELLs the spiked leg (overpriced premium), buys back at target profit or SL.
Saves results to DB under strategy_name = 'v4_gamma_sell'.
"""

import os
import sys
import time
import pickle
import logging
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from simple_salesforce import Salesforce

# Reuse OptionFetcher from backtest_v4
sys.path.insert(0, os.path.dirname(__file__))
from backtest_v4 import OptionFetcher

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

SF_BACKTEST_OBJECT = "historical_backtests__c"
IST_TZ = ZoneInfo("Asia/Kolkata")
UTC_TZ = ZoneInfo("UTC")

# ── Strategy Parameters ─────────────────────────────────────────────────────
LEG_EXPANSION    = 1.20   # individual CE or PE must spike 20% from benchmark to SELL
MIN_SELL_PRICE   = 20.0   # min premium to sell (too cheap = high gamma risk)
BENCHMARK_TIME  = "09:20" # capture baseline at this bar
ENTRY_START      = "09:30" # start watching for spikes
ENTRY_CUTOFF     = "11:30" # no new sells after this
TARGET_PCT       = 0.35   # buy back when premium drops 35% from sell price (profit)
SL_PCT           = 0.40   # buy back if premium rises 40% from sell price (loss)
QTY              = 1300
INITIAL_CAPITAL  = 500_000
# ────────────────────────────────────────────────────────────────────────────


def get_last_n_nifty_expiries(n=48):
    SWITCH_DATE = datetime(2024, 4, 4)
    today = datetime.now()
    expiries = []
    d = today
    while len(expiries) < n:
        if d >= SWITCH_DATE:
            days_since_tue = (d.weekday() - 1) % 7
            expiry = d - timedelta(days=days_since_tue)
        else:
            days_since_thu = (d.weekday() - 3) % 7
            expiry = d - timedelta(days=days_since_thu)
        expiry_str = expiry.strftime("%Y-%m-%d")
        if expiry_str not in expiries and expiry.date() < today.date():
            expiries.append(expiry_str)
        d -= timedelta(days=7)
    return sorted(expiries)


class GammaSpikeBacktester:
    CACHE_FILE = "/tmp/gamma_option_cache.pkl"

    def __init__(self):
        ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
        self.fetcher = OptionFetcher(ACCESS_TOKEN)
        self.initial_capital = INITIAL_CAPITAL
        self.current_capital = INITIAL_CAPITAL
        self.results = []

        # Load disk cache so re-runs skip API calls
        if os.path.exists(self.CACHE_FILE):
            try:
                with open(self.CACHE_FILE, 'rb') as f:
                    self.fetcher.cache = pickle.load(f)
                logger.info(f"Loaded disk cache: {len(self.fetcher.cache)} entries from {self.CACHE_FILE}")
            except Exception:
                logger.warning("Disk cache unreadable — starting fresh.")

        self.params_str = (
            f"SELL|leg_spike={LEG_EXPANSION*100:.0f}%|"
            f"target={TARGET_PCT*100:.0f}%|"
            f"sl={SL_PCT*100:.0f}%|"
            f"min_sell={MIN_SELL_PRICE}|"
            f"bench={BENCHMARK_TIME}|"
            f"entry={ENTRY_START}-{ENTRY_CUTOFF}"
        )
        self.strategy_name = 'gamma_blast'

    def run(self):
        dates = get_last_n_nifty_expiries(48)
        logger.info(f"Gamma Spike Backtest — {len(dates)} Nifty expiry days")
        logger.info(f"Params: {self.params_str}")
        self.current_capital = self.initial_capital

        for date_str in dates:
            logger.info(f"--- {date_str} ---")

            ce_data = self.fetcher.fetch(date_str, 'C')
            if not ce_data:
                time.sleep(0.3)
                ce_data = self.fetcher.fetch(date_str, 'C')
            pe_data = self.fetcher.fetch(date_str, 'P')
            if not pe_data:
                time.sleep(0.3)
                pe_data = self.fetcher.fetch(date_str, 'P')

            # Save to disk cache after each date
            try:
                with open(self.CACHE_FILE, 'wb') as f:
                    pickle.dump(self.fetcher.cache, f)
            except Exception:
                pass

            if not ce_data or not pe_data:
                logger.warning(f"  No data for {date_str}")
                continue

            sorted_times = sorted(ce_data.keys())

            # ── Capture benchmark at 9:20 AM ──────────────────────────────
            bm_h = int(BENCHMARK_TIME.split(":")[0])
            bm_m = int(BENCHMARK_TIME.split(":")[1])
            benchmark_ce = benchmark_pe = benchmark_spot = None
            for dt in sorted_times:
                if dt.hour == bm_h and dt.minute == bm_m:
                    benchmark_ce   = ce_data[dt]["close"]
                    benchmark_pe   = pe_data.get(dt, {}).get("close")
                    benchmark_spot = ce_data[dt]["strike"]
                    break

            if not benchmark_ce or not benchmark_pe:
                logger.warning(f"  No {BENCHMARK_TIME} bar for {date_str}")
                continue

            logger.info(f"  Benchmark {BENCHMARK_TIME} | CE:{benchmark_ce:.2f} PE:{benchmark_pe:.2f} Spot≈{benchmark_spot}")

            start_h  = int(ENTRY_START.split(":")[0])
            start_m  = int(ENTRY_START.split(":")[1])
            cutoff_h = int(ENTRY_CUTOFF.split(":")[0])
            cutoff_m = int(ENTRY_CUTOFF.split(":")[1])
            position   = None
            entry_time = None

            for dt in sorted_times:
                after_start   = dt.hour > start_h or (dt.hour == start_h and dt.minute >= start_m)
                before_cutoff = dt.hour < cutoff_h or (dt.hour == cutoff_h and dt.minute <= cutoff_m)

                # ── Entry check: SELL the spiked leg ───────────────────────────
                if after_start and before_cutoff and position is None:
                    ce_p = ce_data[dt]["close"]
                    pe_p = pe_data.get(dt, {}).get("close", 0)
                    if not ce_p or not pe_p:
                        continue

                    ce_exp = ce_p / benchmark_ce
                    pe_exp = pe_p / benchmark_pe

                    # Sell whichever leg spiked more (overpriced — mean revert)
                    if max(ce_exp, pe_exp) >= LEG_EXPANSION:
                        opt_type    = 'C' if ce_exp >= pe_exp else 'P'
                        sell_price  = ce_p if opt_type == 'C' else pe_p

                        if sell_price < MIN_SELL_PRICE:
                            logger.info(f"  Skip SELL: premium {sell_price:.2f} < min {MIN_SELL_PRICE}")
                            continue

                        position = {
                            'type':   opt_type,
                            'entry':  sell_price,   # sell price
                            'target': sell_price * (1 - TARGET_PCT),  # buy back at
                            'sl':     sell_price * (1 + SL_PCT),       # cut loss at
                        }
                        entry_time = dt
                        logger.info(
                            f"  SELL {opt_type} @ {sell_price:.2f} "
                            f"| CE+{(ce_exp-1)*100:.1f}% PE+{(pe_exp-1)*100:.1f}% "
                            f"| Target≤{position['target']:.2f} | SL≥{position['sl']:.2f} "
                            f"| {dt.strftime('%H:%M')}"
                        )

                # ── Position management: buy back to close ─────────────────────
                if position and after_start:
                    ce_p = ce_data[dt]["close"]
                    pe_p = pe_data.get(dt, {}).get("close", 0)
                    curr_price = ce_p if position['type'] == 'C' else pe_p
                    if not curr_price:
                        continue

                    exit_triggered = False
                    reason = ""

                    # Target hit: premium decayed enough — BUY BACK (profit)
                    if curr_price <= position['target']:
                        exit_triggered = True
                        reason = "Target Hit"

                    # SL hit: premium rose — BUY BACK (loss)
                    elif curr_price >= position['sl']:
                        exit_triggered = True
                        reason = "SL Hit"

                    # Time exit at 11:30 AM
                    is_eod = (dt.hour == 11 and dt.minute >= 30) or dt.hour > 11
                    if (is_eod or dt == sorted_times[-1]) and not exit_triggered:
                        exit_triggered = True
                        reason = "Time Exit (11:30)"

                    if exit_triggered:
                        buy_back_price = curr_price
                        entry          = position['entry']
                        # SELL strategy: profit = sell_price - buy_back_price
                        pnl = (entry - buy_back_price) * QTY
                        cap_before = self.current_capital
                        self.current_capital += pnl

                        self.results.append({
                            'Date':         date_str,
                            'Entry_Time':   entry_time.strftime("%H:%M:%S"),
                            'Exit_Time':    dt.strftime("%H:%M:%S"),
                            'Option_Type':  position['type'],
                            'Strike':       f"ATM-{date_str}-{position['type']}E",
                            'Action':       'SELL',
                            'Qty':          QTY,
                            'Buy_Price':    round(buy_back_price, 2),   # buy back price
                            'Peak_Price':   round(entry, 2),             # sell price = peak received
                            'Sell_Price':   round(entry, 2),             # sold at
                            'PNL':          round(pnl, 2),
                            'ROI%':         round((entry - buy_back_price) / entry * 100, 2),
                            'Capital_ROI%': round(pnl / cap_before * 100, 2) if cap_before > 0 else 0,
                            'Reason':       reason,
                            'Win':          pnl > 0,
                            'Parameters':   self.params_str
                        })
                        logger.info(f"  BUY BACK @ {buy_back_price:.2f} | PnL: {pnl:,.0f} | {reason}")
                        position = None
                        break   # one trade per day

        self.print_summary()
        self.save_to_salesforce()

    def print_summary(self):
        if not self.results:
            print("\n No trades found — no gamma spikes met criteria.")
            return
        df = pd.DataFrame(self.results)
        total_pnl    = df['PNL'].sum()
        win_rate     = df['Win'].sum() / len(df) * 100
        total_return = (self.current_capital - self.initial_capital) / self.initial_capital * 100

        print(f"\n{'='*65}")
        print(f"  GAMMA SPIKE BACKTEST — 48 Nifty Expiry Days")
        print(f"  Params: {self.params_str}")
        print(f"{'='*65}")
        print(f"  Total Return : {total_return:.2f}%  (₹{total_pnl:,.0f})")
        print(f"  Total Trades : {len(df)}")
        print(f"  Win Rate     : {win_rate:.1f}%")
        print()
        # Reporting labels: short option entry is SELL, exit is BUY BACK.
        display_df = df.rename(columns={
            'Sell_Price': 'Entry_Price',
            'Buy_Price': 'Exit_Price',
        })
        cols = ['Date','Entry_Time','Exit_Time','Option_Type','Entry_Price','Peak_Price','Exit_Price','PNL','Reason','Win']
        print(display_df[cols].to_string(index=False))
        print()

    def save_to_salesforce(self):
        if not self.results:
            return
        try:
            sf = Salesforce(
                username=os.getenv("SF_USERNAME"),
                password=os.getenv("SF_PASSWORD"),
                security_token=os.getenv("SF_SECURITY_TOKEN", ""),
                domain=os.getenv("SF_DOMAIN", "login"),
                version=os.getenv("SF_API_VERSION", "59.0"),
            )
        except Exception as exc:
            logger.error("Salesforce login failed: %s", exc)
            return

        run_ts = datetime.utcnow().strftime("%Y-%m-%d")

        def _to_sf_datetime(date_str, time_str):
            local_dt = datetime.strptime(
                f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
            ).replace(tzinfo=IST_TZ)
            utc_dt = local_dt.astimezone(UTC_TZ)
            return utc_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        trade_records = [
            {
                "Run_Date__c":                run_ts,
                "Strategy_Name__c":           self.strategy_name,
                "Total_PNL__c":               r["PNL"],
                "Total_Return_Percentage__c": r["ROI%"],
                "Trade_Date__c":              r["Date"],
                "Entry_Time__c":              _to_sf_datetime(r["Date"], r["Entry_Time"]),
                "Exit_Time__c":               _to_sf_datetime(r["Date"], r["Exit_Time"]),
                "Option_Type__c":             r["Option_Type"],
                "Action__c":                  r["Action"],
                "Qty__c":                     QTY,
                "Buy_Price__c":               r["Buy_Price"],
                "Peak_Price__c":              r["Peak_Price"],
                "Sell_Price__c":              r["Sell_Price"],
                "Reason__c":                  r["Reason"],
                "Win__c":                     r["Win"],
                "Capital_ROI_Pct__c":         r["Capital_ROI%"],
                "Run_Mode__c":                "backtest",
                "Strike__c":                  r["Strike"],
                "PnL_INR__c":                 r["PNL"],
                "Parameters__c":              r["Parameters"],
            }
            for r in self.results
        ]

        try:
            bulk_backtests = getattr(sf.bulk, SF_BACKTEST_OBJECT)

            # Insert per-trade rows
            res = bulk_backtests.insert(trade_records)
            ok   = sum(1 for r in res if r.get("success"))
            fail = sum(1 for r in res if not r.get("success"))
            logger.info(
                "-> SF SYNC: %d trade rows saved to %s (%d failed).",
                ok, SF_BACKTEST_OBJECT, fail,
            )
            if fail:
                bad = [r for r in res if not r.get("success")]
                logger.warning("First error: %s", bad[0].get("errors"))
        except Exception as exc:
            logger.error("Salesforce save failed: %s", exc)


if __name__ == "__main__":
    bt = GammaSpikeBacktester()
    bt.run()
