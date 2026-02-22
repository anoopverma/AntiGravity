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
        self.stop_loss_pct = 0.30
        self.target_pct = 1.00 # 100% Target
        self.results = []
        
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

    def get_last_two_tuesdays(self):
        """Get the date strings for the last two Tuesdays."""
        today = datetime.now()
        # Find the most recent Tuesday
        offset = (today.weekday() - 1) % 7
        last_tuesday = today - timedelta(days=offset)
        if today.weekday() < 1: # If it's earlier than Tuesday this week
            last_tuesday = last_tuesday - timedelta(days=7)
        
        tuesdays = [
            (last_tuesday - timedelta(days=7)).strftime("%Y-%m-%d"),
            last_tuesday.strftime("%Y-%m-%d")
        ]
        return tuesdays

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
                    logger.warning(f"No Dhan data for {date_str} - {req.get('remarks')}")
                    return pd.DataFrame()
            except Exception as e:
                logger.error(f"Dhan API Error on {date_str}: {e}")
                time.sleep(5)
        return pd.DataFrame()

    def run_backtest(self):
        """Run backtest for the last 2 Tuesdays."""
        tuesdays = self.get_last_two_tuesdays()
        logger.info(f"Starting 2-week Tuesday Backtest for: {tuesdays}")
        
        for date_str in tuesdays:
            df = self.fetch_dhan_5min_data(date_str)
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
                    
                    # Check SL (30% of combined premium)
                    if curr_premium >= entry_premium * (1 + self.stop_loss_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'StopLoss')
                        in_position = False
                        break
                    
                    # Check Target (100% of premium - basically premium goes to 0 or very low)
                    # For a short strangle, 100% target means we collect the full premium.
                    if curr_premium <= entry_premium * (1 - self.target_pct):
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'Target')
                        in_position = False
                        break
                        
                    # Exit at EOD
                    if hour == 15 and minute >= 15:
                        self.record_trade(date_str, entry_time, index, entry_spot, spot_price, entry_ce_strike, entry_pe_strike, pnl_points, entry_lots, peak_pnl, trough_pnl, 'Time')
                        in_position = False
                        break
                
                # Entry after 1:30 PM
                if not in_position and hour == 13 and minute >= 30:
                    # Log once when we cross 1:30 PM
                    if not hasattr(self, '_logged_entry_check'):
                        self._logged_entry_check = set()
                    if date_str not in self._logged_entry_check:
                        logger.info(f"Entry check triggered at {index} for {date_str}")
                        self._logged_entry_check.add(date_str)
                        
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
                    logger.info(f"Trade Entered on {date_str} at {index.time()} | Spot: {entry_spot} | Lot: {entry_lots}")
                    
            time.sleep(1)
        
        self.print_statistics()

    def record_trade(self, date_str, entry_time, exit_time, entry_spot, exit_spot, ce_strike, pe_strike, pnl_points, lots, peak, trough, reason):
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

    def print_statistics(self):
        if not self.results:
            logger.info("No trades executed.")
            return
            
        df = pd.DataFrame(self.results)
        total_pnl = df['PnL_INR'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        
        md = f"""
### 📊 Nifty Tuesday Expiry Backtest (Last 2 Weeks, Dhan API)

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
        
        with open("/Users/anoop/.gemini/antigravity/brain/311c7cff-5d0e-40ca-b43a-de26854c129a/walkthrough.md", "a") as f:
            f.write("\n\n---\n\n" + md)
            
        print(md)

if __name__ == "__main__":
    backtester = NiftyTuesdayDhanBacktester()
    backtester.run_backtest()
