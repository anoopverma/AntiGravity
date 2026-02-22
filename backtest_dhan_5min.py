import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dhanhq import dhanhq, DhanContext
from scipy.stats import norm
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for Black-Scholes Approximation
RISK_FREE_RATE = 0.07  
IMPLIED_VOL_ASSUMPTION = 0.15  

class NiftyTuesdayDhanBacktester:
    def __init__(self):
        load_dotenv()
        self.client_id = os.getenv('DHAN_CLIENT_ID')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN')
        
        if not self.client_id or not self.access_token:
            raise ValueError("Dhan API credentials not found in .env")
            
        self.dhan = dhanhq(DhanContext(self.client_id, self.access_token))
        
        # Strategy Parameters
        self.initial_capital = 500000
        self.current_capital = self.initial_capital
        self.margin_per_lot = 120000
        self.lot_size = 25
        self.results = []
        self.cached_data = {} # Cache fetched data to avoid redundant API calls
        
    def estimate_option_price(self, spot_price, strike_price, time_to_expiry_years, iv, option_type):
        """Estimate Black Scholes price for options (simplified for PnL tracking)."""
        if time_to_expiry_years <= 0:
            return max(0.01, spot_price - strike_price) if option_type == 'C' else max(0.01, strike_price - spot_price)
            
        d1 = (np.log(spot_price / strike_price) + (RISK_FREE_RATE + 0.5 * iv**2) * time_to_expiry_years) / (iv * np.sqrt(time_to_expiry_years))
        d2 = d1 - iv * np.sqrt(time_to_expiry_years)
        
        if option_type == 'C':
            price = spot_price * norm.cdf(d1) - strike_price * np.exp(-RISK_FREE_RATE * time_to_expiry_years) * norm.cdf(d2)
        else: # 'P'
            price = strike_price * np.exp(-RISK_FREE_RATE * time_to_expiry_years) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        return max(0.01, price)

    def get_last_n_tuesdays(self, n=12):
        """Get the date strings for the last N Tuesdays."""
        today = datetime.now()
        # Find the most recent Tuesday
        offset = (today.weekday() - 1) % 7
        last_tuesday = today - timedelta(days=offset)
        if today.weekday() < 1: # If it's earlier than Tuesday this week
            last_tuesday = last_tuesday - timedelta(days=7)
        
        tuesdays = []
        for i in range(n):
            tuesdays.append((last_tuesday - timedelta(days=7*i)).strftime("%Y-%m-%d"))
        
        return sorted(tuesdays)

    def fetch_yf_5min_fallback(self, date_str):
        """Fetch 5-min data from YFinance for a specific date."""
        try:
            import yfinance as yf
            ticker = yf.Ticker("^NSEI")
            # YF only allows 5m data for the last 60 days. 
            # If date_str is older, this might fail or return empty.
            start_date = date_str
            end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = ticker.history(start=start_date, end=end_date, interval="5m")
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                # Ensure index is datetime and handled correctly
                logger.info(f"Fallback: Fetched {len(df)} 5-min bars from YFinance for {date_str}")
                return df
        except Exception as e:
            logger.error(f"YFinance fallback failed for {date_str}: {e}")
        return pd.DataFrame()

    def fetch_dhan_5min_data(self, date_str, retries=3):
        """Fetch 1-min intraday data and resample to 5-min."""
        for attempt in range(retries):
            try:
                req = self.dhan.intraday_minute_data(
                    security_id='13', # Nifty 50
                    exchange_segment=self.dhan.INDEX,
                    instrument_type='INDEX',
                    from_date=date_str,
                    to_date=date_str
                )
                
                if req.get('status') == 'success' and req.get('data'):
                    df = pd.DataFrame(req['data'])
                    if df.empty:
                        return pd.DataFrame()
                        
                    # Handle different versions of Dhan API response keys
                    time_col = None
                    if 'timestamp' in df.columns:
                        time_col = 'timestamp'
                    elif 'start_Time' in df.columns:
                        time_col = 'start_Time'
                    
                    if time_col:
                        if time_col == 'timestamp':
                            df['datetime'] = pd.to_datetime(df[time_col], unit='s') + pd.Timedelta(hours=5, minutes=30)
                        else:
                            df['datetime'] = pd.to_datetime(df[time_col]) + pd.Timedelta(hours=5, minutes=30)
                        
                        # Set index
                        df.set_index('datetime', inplace=True)
                        
                        df_5m = df.resample('5min').agg({
                            'open': 'first',
                            'high': 'max',
                            'low': 'min',
                            'close': 'last'
                        }).dropna()
                        
                        logger.info(f"Fetched {len(df_5m)} 5-min bars for {date_str}. Range: {df_5m.index[0]} to {df_5m.index[-1]}")
                        return df_5m
                elif req.get('remarks') and 'DH-904' in str(req.get('remarks')):
                    logger.warning(f"Rate limited on {date_str}. Waiting 5s (Attempt {attempt+1}/{retries})")
                    time.sleep(5)
                else:
                    logger.warning(f"No Dhan data for {date_str} - {req.get('remarks')}. Trying fallback...")
                    break
            except Exception as e:
                logger.error(f"Dhan API Error on {date_str}: {e}. Trying fallback...")
                break
        
        # If Dhan failed, try YF
        return self.fetch_yf_5min_fallback(date_str)

    def run_sell_backtest(self, stop_loss_pct=None, target_pct=0.75, silent=False):
        """Run backtest for the last 12 Tuesdays with Selling (Theta Decay) logic."""
        tuesdays = self.get_last_n_tuesdays(12)
        if not silent:
            logger.info(f"Starting 12-week Tuesday SELL Backtest (SL: {stop_loss_pct*100 if stop_loss_pct else 'None'}%, TGT: {target_pct*100}%)")
        
        self.results = []
        self.current_capital = self.initial_capital
        
        for date_str in tuesdays:
            df = self.cached_data.get(date_str)
            if df is None:
                df = self.fetch_dhan_5min_data(date_str)
                self.cached_data[date_str] = df
            if df.empty: continue
            
            in_position = False
            entry_time, entry_spot, entry_ce_strike, entry_pe_strike, entry_premium, entry_lots = None, 0, 0, 0, 0, 0
            peak_pnl, trough_pnl = 0, 0
            eod_time = df.index[-1].replace(hour=15, minute=30)
            
            for index, row in df.iterrows():
                spot_price = row['close']
                dte = (eod_time - index).total_seconds() / (365 * 24 * 3600)
                
                if in_position:
                    curr_ce = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                    curr_pe = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    curr_premium = curr_ce + curr_pe
                    pnl_points = entry_premium - curr_premium
                    peak_pnl, trough_pnl = max(peak_pnl, pnl_points), min(trough_pnl, pnl_points)
                    
                    if stop_loss_pct and curr_premium >= entry_premium * (1 + stop_loss_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'StopLoss', entry_premium, curr_premium, silent)
                        in_position = False; break
                    if target_pct and curr_premium <= entry_premium * (1 - target_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'Target', entry_premium, curr_premium, silent)
                        in_position = False; break
                    if index.hour == 15 and index.minute >= 15:
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'Time', entry_premium, curr_premium, silent)
                        in_position = False; break
                
                if not in_position and index.hour == 13 and index.minute >= 30:
                    in_position = True
                    entry_time, entry_spot = index, spot_price
                    entry_lots = max(1, int(self.current_capital // self.margin_per_lot))
                    atm_strike = round(spot_price / 50) * 50
                    entry_ce_strike, entry_pe_strike = atm_strike + 100, atm_strike - 100
                    entry_premium = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                    self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    peak_pnl, trough_pnl = 0, 0
                    if not silent: logger.info(f"Sell Entered: {date_str} {index.time()} Spot: {entry_spot}")
            time.sleep(0.1)
        return self.get_summary_sell(stop_loss_pct, target_pct)

    def run_buy_backtest(self, jump_trigger=0.40, stop_loss_pct=None, target_pct=None, silent=False):
        """Run backtest for the last 12 Tuesdays with Buy Momentum logic."""
        tuesdays = self.get_last_n_tuesdays(12)
        if not silent:
            logger.info(f"Starting 12-week Tuesday BUY Backtest (Trigger: {jump_trigger*100}%, SL: {stop_loss_pct*100 if stop_loss_pct else 'None'}%, TGT: {target_pct*100 if target_pct else 'EOD'}%)")
        
        self.results = []
        self.current_capital = self.initial_capital
        
        for date_str in tuesdays:
            df = self.cached_data.get(date_str)
            if df is None:
                df = self.fetch_dhan_5min_data(date_str)
                self.cached_data[date_str] = df
            if df.empty: continue
            
            in_position = False
            baseline_premium = None
            entry_time, entry_spot, entry_ce_strike, entry_pe_strike, entry_premium, entry_qty = None, 0, 0, 0, 0, 0
            peak_pnl, trough_pnl = 0, 0
            eod_time = df.index[-1].replace(hour=15, minute=30)
            
            for index, row in df.iterrows():
                spot_price = row['close']
                dte = (eod_time - index).total_seconds() / (365 * 24 * 3600)
                
                if in_position:
                    curr_ce = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                    curr_pe = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    curr_premium = curr_ce + curr_pe
                    pnl_points = curr_premium - entry_premium
                    peak_pnl, trough_pnl = max(peak_pnl, pnl_points), min(trough_pnl, pnl_points)
                    
                    if stop_loss_pct and curr_premium <= entry_premium * (1 - stop_loss_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_qty // self.lot_size, peak_pnl, trough_pnl, 'StopLoss', entry_premium, curr_premium, silent)
                        in_position = False; break
                    if target_pct and curr_premium >= entry_premium * (1 + target_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_qty // self.lot_size, peak_pnl, trough_pnl, 'Target', entry_premium, curr_premium, silent)
                        in_position = False; break
                    if index.hour == 15 and index.minute >= 15:
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_qty // self.lot_size, peak_pnl, trough_pnl, 'Time', entry_premium, curr_premium, silent)
                        in_position = False; break
                
                if index.hour == 13 and index.minute == 30 and baseline_premium is None:
                    atm_strike = round(spot_price / 50) * 50
                    entry_ce_strike, entry_pe_strike = atm_strike + 100, atm_strike - 100
                    baseline_premium = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                       self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    if not silent: logger.info(f"Baseline at 1:30 PM: {baseline_premium:.2f}")

                if not in_position and baseline_premium and index.time() > datetime.strptime("13:30", "%H:%M").time():
                    curr_premium = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                   self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    if curr_premium >= baseline_premium * (1 + jump_trigger):
                        in_position = True
                        entry_time, entry_spot, entry_premium = index, spot_price, curr_premium
                        entry_qty = max(self.lot_size, int(self.current_capital // (entry_premium * self.lot_size)) * self.lot_size)
                        peak_pnl, trough_pnl = 0, 0
                        if not silent: logger.info(f"Buy Triggered: {date_str} {index.time()} Prem: {entry_premium:.2f}")
            time.sleep(0.1)
        return self.get_summary_buy_optimized(jump_trigger, stop_loss_pct, target_pct)

    def run_directional_buy_backtest(self, stop_loss_pct=0.30, target_pct=1.00, silent=False):
        """Run directional buy backtest based on 1:30 PM movement."""
        tuesdays = self.get_last_n_tuesdays(12)
        if not silent:
            logger.info("Starting 12-week Tuesday DIRECTIONAL BUY Backtest")
        
        self.results = []
        self.current_capital = self.initial_capital
        
        for date_str in tuesdays:
            df = self.cached_data.get(date_str)
            if df is None:
                df = self.fetch_dhan_5min_data(date_str)
                self.cached_data[date_str] = df
            if df.empty: continue
            
            baseline_spot = None
            straddle_baseline = None
            in_position = False
            entry_time, entry_spot, entry_strike, entry_premium, entry_qty = None, 0, 0, 0, 0
            peak_pnl, trough_pnl = 0, 0
            eod_time = df.index[-1].replace(hour=15, minute=30)
            
            for index, row in df.iterrows():
                spot_price = row['close']
                dte = (eod_time - index).total_seconds() / (365 * 24 * 3600)
                
                # Setup Baseline at 1:30 PM
                if index.hour == 13 and index.minute == 30:
                    baseline_spot = spot_price
                    atm_strike = round(spot_price / 50) * 50
                    straddle_baseline = self.estimate_option_price(spot_price, atm_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                       self.estimate_option_price(spot_price, atm_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    if not silent: logger.info(f"1:30 PM Baseline - Spot: {baseline_spot}, Straddle: {straddle_baseline:.2f}")

                if in_position:
                    # Current price of the single directional option
                    curr_p = self.estimate_option_price(spot_price, entry_strike, dte, IMPLIED_VOL_ASSUMPTION, entry_type)
                    pnl_points = curr_p - entry_premium
                    peak_pnl, trough_pnl = max(peak_pnl, pnl_points), min(trough_pnl, pnl_points)
                    
                    if stop_loss_pct and curr_p <= entry_premium * (1 - stop_loss_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_strike, 0, pnl_points, entry_qty // self.lot_size, peak_pnl, trough_pnl, 'StopLoss', entry_premium, curr_p, silent)
                        in_position = False; break
                    if target_pct and curr_p >= entry_premium * (1 + target_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_strike, 0, pnl_points, entry_qty // self.lot_size, peak_pnl, trough_pnl, 'Target', entry_premium, curr_p, silent)
                        in_position = False; break
                    if index.hour == 15 and index.minute >= 15:
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_strike, 0, pnl_points, entry_qty // self.lot_size, peak_pnl, trough_pnl, 'Time', entry_premium, curr_p, silent)
                        in_position = False; break

                elif baseline_spot and index.time() > datetime.strptime("13:30", "%H:%M").time():
                    # Directional Logic
                    direction = 'UP' if spot_price > baseline_spot else 'DOWN'
                    
                    # Current Straddle at 1:30 PM ATM strikes
                    # (Breakout of the volatility recorded at 1:30 PM)
                    atm_strike_130 = round(baseline_spot / 50) * 50
                    curr_straddle = self.estimate_option_price(spot_price, atm_strike_130, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                    self.estimate_option_price(spot_price, atm_strike_130, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    
                    # Entry Condition: Current Straddle > 1:30 PM Straddle (Volatility Expansion)
                    if curr_straddle > straddle_baseline:
                        opt_type = 'C' if direction == 'UP' else 'P'
                        
                        # Selection: Find first strike > 9 rupees
                        target_strike = None
                        target_price = 0
                        step = 50
                        base = round(spot_price / 50) * 50
                        
                        if direction == 'UP':
                            for s in range(base, base + 500, step):
                                p = self.estimate_option_price(spot_price, s, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                                if p > 9:
                                    target_strike, target_price = s, p; break
                        else:
                            for s in range(base, base - 500, -step):
                                p = self.estimate_option_price(spot_price, s, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                                if p > 9:
                                    target_strike, target_price = s, p; break
                        
                        if target_strike:
                            in_position = True
                            entry_time, entry_spot, entry_strike, entry_premium, entry_type = index, spot_price, target_strike, target_price, opt_type
                            entry_qty = max(self.lot_size, int(self.current_capital // (entry_premium * self.lot_size)) * self.lot_size)
                            if not silent: logger.info(f"Straddle Breakout! {direction} Buy: Strike {entry_strike} @ {entry_premium:.2f} (Straddle: {curr_straddle:.2f} > {straddle_baseline:.2f})")
            
            time.sleep(0.1)
        return self.get_summary_directional(stop_loss_pct, target_pct)

    def run_directional_optimization_sweep(self):
        """Run a sweep over SL and Target combinations for Directional Buy Strategy."""
        sl_values = [0.10, 0.20, 0.30, 0.50, None]
        tgt_values = [0.50, 1.00, 2.00, 5.00, None]
        
        sweep_results = []
        logger.info("Starting Directional Strategy Optimization Sweep...")
        
        for sl in sl_values:
            for tgt in tgt_values:
                summary = self.run_directional_buy_backtest(stop_loss_pct=sl, target_pct=tgt, silent=True)
                if summary:
                    sweep_results.append(summary)
        
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df = sweep_df.sort_values(by='ROI%', ascending=False)
        
        print("\n=== Directional Optimization Sweep Results (Sorted by ROI) ===")
        print(sweep_df.to_string(index=False))
        
        summary_md = "\n\n### 🎯 Directional Strategy Optimization Sweep Results\n\n| SL | Target | Trades | PnL (INR) | ROI % | Accuracy |\n|---|---|---|---|---|---|---|\n"
        for _, row in sweep_df.iterrows():
            summary_md += f"| {row['SL']} | {row['TGT']} | {row['Trades']} | **₹{row['PnL']:,}** | {row['ROI%']}% | {row['Accuracy%']}% |\n"
            
        with open("/Users/anoop/.gemini/antigravity/brain/311c7cff-5d0e-40ca-b43a-de26854c129a/walkthrough.md", "a") as f:
            f.write(summary_md)

    def get_summary_directional(self, sl, tgt):
        if not self.results: return None
        df = pd.DataFrame(self.results)
        pnl = df['PnL_INR'].sum()
        roi = (pnl / self.initial_capital) * 100
        return {
            'SL': f"{int(sl*100)}%" if sl else "None",
            'TGT': f"{int(tgt*100)}%" if tgt else "EOD",
            'Trades': len(df),
            'ROI%': round(roi, 2),
            'PnL': round(pnl, 2),
            'Accuracy%': round((df['Win'].sum()/len(df))*100, 2)
        }

    def get_summary_sell(self, sl, tgt):
        if not self.results: return None
        df = pd.DataFrame(self.results)
        pnl = df['PnL_INR'].sum()
        return {'SL': f"{int(sl*100)}%" if sl else "None", 'TGT': f"{int(tgt*100)}%", 'Trades': len(df), 'ROI%': round((pnl/self.initial_capital)*100, 2), 'PnL': round(pnl, 2), 'Accuracy%': round((df['Win'].sum()/len(df))*100, 2)}

    def get_summary_buy_optimized(self, trigger, sl, tgt):
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        total_pnl = df['PnL_INR'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        accuracy = (df['Win'].sum() / len(df)) * 100
        return {
            'Trigger': f"{int(trigger*100)}%",
            'SL': f"{int(sl*100)}%" if sl else "None",
            'TGT': f"{int(tgt*100)}%" if tgt else "EOD",
            'Trades': len(df),
            'ROI%': round(roi, 2),
            'PnL': round(total_pnl, 2),
            'Accuracy%': round(accuracy, 2)
        }

    def run_buy_optimization_sweep(self):
        """Run a sweep over Jump Trigger, SL, and Target combinations for Buy Strategy."""
        trigger_values = [0.20, 0.40, 0.60]
        sl_values = [0.10, 0.20, 0.30, None]
        tgt_values = [0.50, 1.00, 2.00, None]
        
        sweep_results = []
        logger.info("Starting Buy Strategy Optimization Sweep...")
        
        for tr in trigger_values:
            for sl in sl_values:
                for tgt in tgt_values:
                    summary = self.run_buy_backtest(jump_trigger=tr, stop_loss_pct=sl, target_pct=tgt, silent=True)
                    if summary:
                        sweep_results.append(summary)
        
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df = sweep_df.sort_values(by='ROI%', ascending=False)
        
        print("\n=== Buy Optimization Sweep Results (Sorted by ROI) ===")
        print(sweep_df.to_string(index=False))
        
        summary_md = "\n\n### 🎯 Buy Strategy Optimization Sweep Results\n\n| Trigger | SL | Target | Trades | PnL (INR) | ROI % | Accuracy |\n|---|---|---|---|---|---|---|\n"
        for _, row in sweep_df.iterrows():
            summary_md += f"| {row['Trigger']} | {row['SL']} | {row['TGT']} | {row['Trades']} | **₹{row['PnL']:,}** | {row['ROI%']}% | {row['Accuracy%']}% |\n"
            
        with open("/Users/anoop/.gemini/antigravity/brain/311c7cff-5d0e-40ca-b43a-de26854c129a/walkthrough.md", "a") as f:
            f.write(summary_md)

    def record_trade(self, date_str, entry_time, exit_time, entry_spot, exit_spot, ce_strike, pe_strike, pnl_points, lots, peak, trough, reason, entry_prem, exit_prem, silent=False):
        pnl_inr = pnl_points * self.lot_size * lots
        self.current_capital += pnl_inr
        self.results.append({
            'Date': date_str,
            'Entry': entry_time.strftime("%H:%M"),
            'Exit': exit_time.strftime("%H:%M"),
            'Spot_Entry': round(entry_spot, 2),
            'Spot_Exit': round(exit_spot, 2),
            'Strikes': f"{ce_strike}/{pe_strike}",
            'Entry_Price': round(entry_prem, 2),
            'Exit_Price': round(exit_prem, 2),
            'PnL_Points': round(pnl_points, 2),
            'PnL_INR': round(pnl_inr, 2),
            'Capital': round(self.current_capital, 2),
            'Lots': lots,
            'Reason': reason,
            'Win': pnl_points > 0
        })
        logger.info(f"Trade Exited on {date_str} at {exit_time.time()} | Reason: {reason} | PnL: {pnl_inr:.2f}")

    def get_summary(self, sl, tgt):
        if not self.results:
            return None
        df = pd.DataFrame(self.results)
        total_pnl = df['PnL_INR'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        accuracy = (df['Win'].sum() / len(df)) * 100
        return {
            'SL': f"{int(sl*100)}%" if sl is not None else "None",
            'TGT': f"{int(tgt*100)}%",
            'Trades': len(df),
            'ROI%': round(roi, 2),
            'PnL': round(total_pnl, 2),
            'Accuracy%': round(accuracy, 2)
        }

    def print_statistics_sell(self, sl=None, tgt=0.75):
        if not self.results: return
        df = pd.DataFrame(self.results)
        df['Profit%'] = (df['PnL_Points'] / df['Entry_Price']) * 100
        roi = (df['PnL_INR'].sum() / self.initial_capital) * 100
        print(f"\n--- SELLING STRATEGY RESULTS ---")
        print(f"Total ROI: {roi:.2f}% | Accuracy: {(df['Win'].sum()/len(df))*100:.2f}%")
        print(df[['Date', 'Entry_Price', 'Exit_Price', 'PnL_INR', 'Profit%', 'Reason']].to_string(index=False))

    def print_statistics_directional(self):
        if not self.results:
            print("\n--- DIRECTIONAL BUY STRATEGY: No Trades Triggered ---")
            return
        df = pd.DataFrame(self.results)
        df['Profit%'] = (df['PnL_Points'] / df['Entry_Price']) * 100
        roi = (df['PnL_INR'].sum() / self.initial_capital) * 100
        print(f"\n--- DIRECTIONAL BUY STRATEGY RESULTS ---")
        print(f"Total ROI: {roi:.2f}% | Accuracy: {(df['Win'].sum()/len(df))*100:.2f}%")
        print(df[['Date', 'Entry_Price', 'Exit_Price', 'PnL_INR', 'Profit%', 'Reason']].to_string(index=False))

    def print_statistics_buy(self, trigger=0.40):
        if not self.results: return
        df = pd.DataFrame(self.results)
        roi = (df['PnL_INR'].sum() / self.initial_capital) * 100
        print(f"\n--- BUYING STRATEGY RESULTS ---")
        print(f"Total ROI: {roi:.2f}% | Accuracy: {(df['Win'].sum()/len(df))*100:.2f}%")
        print(df[['Date', 'Entry', 'Exit', 'PnL_INR', 'Reason']].to_string(index=False))

if __name__ == "__main__":
    backtester = NiftyTuesdayDhanBacktester()
    
    print("\n=== Nifty Tuesday Expiry Strategy Selection ===")
    print("1. Optimal SELLING Strategy (75% Target)")
    print("2. DIRECTIONAL BUY Strategy Matrix")
    print("3. Momentum BUYING Strategy (40% Jump)")
    print("4. Optimization SWEEP: Directional Buy")
    
    choice = "1" # Set back to the optimal 100% accuracy selling strategy
    
    if choice == "1":
        backtester.run_sell_backtest(stop_loss_pct=None, target_pct=0.75)
        backtester.print_statistics_sell(sl=None, tgt=0.75)
    elif choice == "2":
        backtester.run_directional_buy_backtest(stop_loss_pct=0.30, target_pct=1.00)
        backtester.print_statistics_directional()
    elif choice == "3":
        backtester.run_buy_backtest(jump_trigger=0.40)
        backtester.print_statistics_buy(trigger=0.40)
    elif choice == "4":
        backtester.run_directional_optimization_sweep()
