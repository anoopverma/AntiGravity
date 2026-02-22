import time
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.stats import norm
from collections import deque
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for Black-Scholes Approximation
RISK_FREE_RATE = 0.07  # Assume 7% risk-free rate in India
IMPLIED_VOL_ASSUMPTION = 0.15  # Assume a constant 15% IV for approximation

class StrangleBacktester1Yr:
    def __init__(self):
        self.symbol = "^NSEI" # Nifty 50 symbol on Yahoo Finance
        self.gamma_window = deque(maxlen=20)
        self.spike_threshold = 1.1  # Lowered for hourly data (10% spike vs 30% for 1m data)
        self.results = []
        
    def get_past_tuesdays_1yr(self):
        """Get all Tuesdays in the last 52 weeks."""
        today = datetime.now()
        offset = (today.weekday() - 1) % 7
        last_tuesday = today - timedelta(days=offset)
        
        tuesdays = []
        # yfinance 1m data limitation:
        # Note: Yahoo Finance restricts 1-minute interval data to the last 7 days (or 30 days max depending on tier).
        # We will attempt to get a year of 1m data, but if it fails, we fall back to the max available 1m data.
        for i in range(52):
            t_date = last_tuesday - timedelta(weeks=i)
            next_day = t_date + timedelta(days=1)
            tuesdays.append((t_date.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d")))
        return reversed(tuesdays) # Oldest to newest

    def approximate_gamma(self, spot_price, strike_price, time_to_expiry_years, iv):
        """Estimate Gamma using Black-Scholes formula."""
        if time_to_expiry_years <= 0 or spot_price <= 0 or iv <= 0:
            return 0.0
            
        d1 = (np.log(spot_price / strike_price) + (RISK_FREE_RATE + 0.5 * iv**2) * time_to_expiry_years) / (iv * np.sqrt(time_to_expiry_years))
        gamma = norm.pdf(d1) / (spot_price * iv * np.sqrt(time_to_expiry_years))
        return gamma
        
    def estimate_option_price(self, spot_price, strike_price, time_to_expiry_years, iv, option_type):
        """Estimate Black Scholes price for options (simplified for PnL tracking)."""
        if time_to_expiry_years <= 0:
            return max(0, spot_price - strike_price) if option_type == 'C' else max(0, strike_price - spot_price)
            
        d1 = (np.log(spot_price / strike_price) + (RISK_FREE_RATE + 0.5 * iv**2) * time_to_expiry_years) / (iv * np.sqrt(time_to_expiry_years))
        d2 = d1 - iv * np.sqrt(time_to_expiry_years)
        
        if option_type == 'C':
            price = spot_price * norm.cdf(d1) - strike_price * np.exp(-RISK_FREE_RATE * time_to_expiry_years) * norm.cdf(d2)
        else: # 'P'
            price = strike_price * np.exp(-RISK_FREE_RATE * time_to_expiry_years) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        return max(0.01, price)

    def run_backtest_year(self):
        """Run backtest on historical Nifty 50 minute data for the max available duration."""
        logger.info(f"--- Starting 1-Year (Hourly Available) Backtest ---")
        
        tuesdays = list(self.get_past_tuesdays_1yr())
        ticker = yf.Ticker(self.symbol)
        
        start_date = tuesdays[0][0]
        end_date = tuesdays[-1][1]
        
        logger.info(f"Requesting 1-hour data from {start_date} to {end_date}...")
        try:
            # yfinance allows 730 days for 1h interval, so we can fetch all at once
            df_all = ticker.history(start=start_date, end=end_date, interval="1h")
            if df_all.empty:
                logger.error("Failed to retrieve any 1h data.")
                return
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return
            
        logger.info(f"Actually retrieved {len(df_all)} hours of data from {df_all.index[0].date()} to {df_all.index[-1].date()}")
        
        # Group by day
        df_all['Date'] = df_all.index.date
        unique_days = df_all['Date'].unique()
        
        for current_date in unique_days:
            # We ONLY want Tuesdays
            if current_date.weekday() != 1: 
                continue
                
            df = df_all[df_all['Date'] == current_date].copy()
            if df.empty:
                continue
                
            self.gamma_window.clear()
            in_position = False
            total_daily_pnl = 0
            trades_today = 0
            
            # End of trading day (15:30 IST)
            end_of_day_time = df.index[-1]
            
            entry_ce_strike = 0
            entry_pe_strike = 0
            entry_premium_paid = 0
            peak_pnl = 0
            trough_pnl = 0
            
            for index, row in df.iterrows():
                spot_price = row['Close']
                hour = index.hour
                minute = index.minute
                
                # Ensure timezone awareness to match yfinance index
                end_of_day_time = df.index[-1].replace(hour=15, minute=30)
                if end_of_day_time < index:
                    end_of_day_time = index
                
                time_left = end_of_day_time - index
                minutes_left = max(1, time_left.total_seconds() / 60)
                dte = minutes_left / (365 * 24 * 60)
                
                # Approximate ATM Gamma
                atm_strike = round(spot_price / 50) * 50
                current_gamma = self.approximate_gamma(spot_price, atm_strike, dte, IMPLIED_VOL_ASSUMPTION)
                
                # Track PnL if in position
                if in_position:
                    # Current Strangle Price
                    current_ce_price = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                    current_pe_price = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    current_strangle_value = current_ce_price + current_pe_price
                    
                    unrealized_pnl = entry_premium_paid - current_strangle_value # SHORT Strangle
                    peak_pnl = max(peak_pnl, unrealized_pnl)
                    trough_pnl = min(trough_pnl, unrealized_pnl)
                    
                    # Exit near close (e.g. at the 14:15 or 15:15 hourly bar)
                    if hour >= 14:
                        pnl = unrealized_pnl
                        total_daily_pnl += pnl
                        self.results.append({
                            'Date': current_date.strftime("%Y-%m-%d"),
                            'Entry_Time': entry_time.strftime("%H:%M"),
                            'Exit_Time': index.strftime("%H:%M"),
                            'Spot_Entry': round(entry_spot, 2),
                            'Spot_Exit': round(spot_price, 2),
                            'CE_Strike': entry_ce_strike,
                            'PE_Strike': entry_pe_strike,
                            'Max_Drawdown': round(trough_pnl, 2),
                            'Peak_Profit': round(peak_pnl, 2),
                            'PnL_Points': round(pnl, 2),
                            'Win': pnl > 0
                        })
                        in_position = False
                        continue
                
                # Strategy logic - Execute Strangle unconditionally at or shortly after 1:30 PM
                if not in_position and hour >= 13:
                    # In hourly data, we might not have a 13:30 bar specifically, we might have 13:15 or 14:15.
                    # As soon as we pass market time 13:30, we enter.
                    time_passed = hour > 13 or (hour == 13 and minute >= 30)
                    if time_passed:
                        # Enter Long Strangle (Spot+100 CE, Spot-100 PE)
                        in_position = True
                        trades_today += 1
                        entry_spot = spot_price
                        entry_time = index
                        
                        entry_ce_strike = atm_strike + 100
                        entry_pe_strike = atm_strike - 100
                        
                        ce_price = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                        pe_price = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                        entry_premium_paid = ce_price + pe_price
                        
                        peak_pnl = 0
                        trough_pnl = 0

                    if current_gamma > 0:
                        self.gamma_window.append(current_gamma)

        self.print_statistics()

    def print_statistics(self):
        """Calculate and print tabular statistics."""
        if not self.results:
            logger.info("No trades were executed during the backtest period.")
            return
            
        df = pd.DataFrame(self.results)
        
        total_trades = len(df)
        wins = df['Win'].sum()
        losses = total_trades - wins
        accuracy = (wins / total_trades) * 100
        
        total_pnl_points = df['PnL_Points'].sum()
        # Nifty lot size is 25
        lot_size = 25
        total_profit_inr = total_pnl_points * lot_size
        
        max_drawdown = df['Max_Drawdown'].min() * lot_size
        max_profit = df['Peak_Profit'].max() * lot_size
        avg_pnl = df['PnL_Points'].mean() * lot_size
        
        md = f"""
### 📊 Backtest Performance Statistics (Tuesdays, >1:30 PM, Strangle)

| Metric | Value |
|--------|-------|
| **Total Trades** | {total_trades} |
| **Wins / Losses** | {wins} / {losses} |
| **Accuracy** | {accuracy:.2f}% |
| **Overall Net Profit (INR @ 1 Lot)** | ₹{total_profit_inr:,.2f} |
| **Total Net Points** | {total_pnl_points:.2f} pts |
| **Average Trade PnL (INR)** | ₹{avg_pnl:,.2f} |
| **Max Drawdown inside a trade (INR)** | ₹{max_drawdown:,.2f} |
| **Highest Peak Profit inside a trade (INR)** | ₹{max_profit:,.2f} |

<br/>

### 📝 Trade Log
| Date | Entry time | Exit time | Spot Entry | Spot Exit | Strikes (CE/PE) | Max Drawdown (pts) | Net PnL (pts) |
|------|------------|-----------|-------------|------------|-----------------|---------------------|---------------|
"""
        for _, row in df.iterrows():
            md += f"| {row['Date']} | {row['Entry_Time']} | {row['Exit_Time']} | {row['Spot_Entry']} | {row['Spot_Exit']} | {row['CE_Strike']} / {row['PE_Strike']} | {row['Max_Drawdown']} | **{row['PnL_Points']}** |\n"
            
        # Write to markdown file so we can show it to user
        with open("/Users/anoop/.gemini/antigravity/brain/311c7cff-5d0e-40ca-b43a-de26854c129a/walkthrough.md", "a") as f:
            f.write("\n\n---\n\n" + md)
            
        print(md)

def main():
    backtester = StrangleBacktester1Yr()
    backtester.run_backtest_year()

if __name__ == "__main__":
    main()
