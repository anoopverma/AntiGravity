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

    def run_backtest(self, stop_loss_pct=None, target_pct=0.75, silent=False):
        """Run backtest for the last 12 Tuesdays with specific SL and Target."""
        tuesdays = self.get_last_n_tuesdays(12)
        if not silent:
            logger.info(f"Starting 12-week Tuesday Backtest (SL: {stop_loss_pct*100 if stop_loss_pct else 'None'}%, TGT: {target_pct*100}%)")
        
        self.results = []
        self.current_capital = self.initial_capital
        
        for date_str in tuesdays:
            if date_str in self.cached_data:
                df = self.cached_data[date_str]
            else:
                df = self.fetch_dhan_5min_data(date_str)
                self.cached_data[date_str] = df
                
            if df.empty:
                continue
            
            in_position = False
            entry_time = None
            entry_spot = 0
            entry_ce_strike = 0
            entry_pe_strike = 0
            entry_premium = 0
            entry_lots = 0
            peak_pnl = 0
            trough_pnl = 0
            
            # End of day 3:30 PM
            eod_time = df.index[-1].replace(hour=15, minute=30)
            
            for index, row in df.iterrows():
                spot_price = row['close']
                hour = index.hour
                minute = index.minute
                
                dte = (eod_time - index).total_seconds() / (365 * 24 * 3600)
                
                if in_position:
                    # Calculate current premium
                    curr_ce = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                    curr_pe = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    curr_premium = curr_ce + curr_pe
                    
                    pnl_points = entry_premium - curr_premium # Short Strangle
                    peak_pnl = max(peak_pnl, pnl_points)
                    trough_pnl = min(trough_pnl, pnl_points)
                    
                    # Check SL
                    if stop_loss_pct is not None and curr_premium >= entry_premium * (1 + stop_loss_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'StopLoss', silent)
                        in_position = False
                        break
                    
                    # Check Target
                    if curr_premium <= entry_premium * (1 - target_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'Target', silent)
                        in_position = False
                        break
                        
                    # Exit at EOD
                    if hour == 15 and minute >= 15:
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'Time', silent)
                        in_position = False
                        break
                
                # Entry after 1:30 PM
                if not in_position and hour == 13 and minute >= 30:
                    in_position = True
                    entry_time = index
                    entry_spot = spot_price
                    entry_lots = max(1, int(self.current_capital // self.margin_per_lot))
                    
                    atm_strike = round(spot_price / 50) * 50
                    entry_ce_strike = atm_strike + 100
                    entry_pe_strike = atm_strike - 100
                    
                    ce_p = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                    pe_p = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    entry_premium = ce_p + pe_p
                    peak_pnl = 0
                    trough_pnl = 0
                    if not silent:
                        logger.info(f"Trade Entered on {date_str} at {index.time()} | Spot: {entry_spot} | Lot: {entry_lots}")
            
            time.sleep(0.5)
        
        return self.get_summary(stop_loss_pct, target_pct)

    def record_trade(self, date_str, entry_time, exit_time, entry_spot, exit_spot, ce_strike, pe_strike, pnl_points, lots, peak, trough, reason, silent=False):
        pnl_inr = pnl_points * self.lot_size * lots
        self.current_capital += pnl_inr
        self.results.append({
            'Date': date_str,
            'Entry': entry_time.strftime("%H:%M"),
            'Exit': exit_time.strftime("%H:%M"),
            'Spot_Entry': round(entry_spot, 2),
            'Spot_Exit': round(exit_spot, 2),
            'Strikes': f"{ce_strike}/{pe_strike}",
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

    def run_optimization_sweep(self):
        """Run a sweep over SL and Target combinations."""
        sl_values = [0.30, 0.50, 1.00, None]
        tgt_values = [0.50, 0.75, 1.00]
        
        sweep_results = []
        logger.info("Starting Optimization Sweep...")
        
        for sl in sl_values:
            for tgt in tgt_values:
                summary = self.run_backtest(stop_loss_pct=sl, target_pct=tgt, silent=True)
                if summary:
                    sweep_results.append(summary)
        
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df = sweep_df.sort_values(by='ROI%', ascending=False)
        
        print("\n=== Optimization Sweep Results (Sorted by ROI) ===")
        print(sweep_df.to_string(index=False))
        
        summary_md = "\n\n### 🎯 Strategy Optimization Sweep Results\n\n| SL | Target | Trades | PnL (INR) | ROI % | Accuracy |\n|---|---|---|---|---|---|\n"
        for _, row in sweep_df.iterrows():
            summary_md += f"| {row['SL']} | {row['TGT']} | {row['Trades']} | **₹{row['PnL']:,}** | {row['ROI%']}% | {row['Accuracy%']}% |\n"
            
        with open("/Users/anoop/.gemini/antigravity/brain/311c7cff-5d0e-40ca-b43a-de26854c129a/walkthrough.md", "a") as f:
            f.write(summary_md)

    def print_statistics(self, sl=None, tgt=0.75):
        if not self.results:
            logger.info("No trades executed.")
            return
            
        df = pd.DataFrame(self.results)
        total_pnl = df['PnL_INR'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        
        md = f"""
### 📊 Nifty Tuesday Optimal Backtest (12 Weeks)
**Settings:** SL: {sl*100 if sl else 'None'}% | Target: {tgt*100}%

| Metric | Value |
|--------|-------|
| **Initial Capital** | ₹{self.initial_capital:,.2f} |
| **Final Capital** | ₹{self.current_capital:,.2f} |
| **Total ROI** | {roi:.2f}% |
| **Total Trades** | {len(df)} |
| **Accuracy** | {(df['Win'].sum() / len(df)) * 100:.2f}% |

<br/>

### 📝 Trade Log
| Date | Entry | Exit | Spot Entry | Spot Exit | Strikes | PnL (INR) | Reason |
|------|-------|------|------------|-----------|---------|-----------|--------|
"""
        for _, row in df.iterrows():
            md += f"| {row['Date']} | {row['Entry']} | {row['Exit']} | {row['Spot_Entry']} | {row['Spot_Exit']} | {row['Strikes']} | **₹{row['PnL_INR']:,.2f}** | {row['Reason']} |\n"
        
        print(md)

if __name__ == "__main__":
    backtester = NiftyTuesdayDhanBacktester()
    # Run the final optimal backtest
    backtester.run_backtest(stop_loss_pct=None, target_pct=0.75)
    backtester.print_statistics(sl=None, tgt=0.75)
