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
RISK_FREE_RATE = 0.07  
IMPLIED_VOL_ASSUMPTION = 0.15  

class YF5MinGammaBacktester:
    def __init__(self):
        self.symbol = "^NSEI" # Nifty 50 symbol on Yahoo Finance
        self.gamma_window = deque(maxlen=20)
        self.spike_threshold = 1.30  # Original Gamma Spike Threshold (30% above trailing avg)
        self.results = []
        
        self.initial_capital = 500000
        self.current_capital = self.initial_capital
        self.margin_per_lot = 120000
        self.lot_size = 25
        self.stop_loss_pct = 0.30

    def get_past_thursdays(self, df):
        """Extract unique Thursdays from Yahoo Finance dataframe index."""
        df['Date'] = df.index.date
        unique_days = df['Date'].unique()
        thursdays = []
        for d in unique_days:
            if d.weekday() == 3: # 3 is Thursday
                thursdays.append(d)
        return thursdays

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
            return max(0.01, spot_price - strike_price) if option_type == 'C' else max(0.01, strike_price - spot_price)
            
        d1 = (np.log(spot_price / strike_price) + (RISK_FREE_RATE + 0.5 * iv**2) * time_to_expiry_years) / (iv * np.sqrt(time_to_expiry_years))
        d2 = d1 - iv * np.sqrt(time_to_expiry_years)
        
        if option_type == 'C':
            price = spot_price * norm.cdf(d1) - strike_price * np.exp(-RISK_FREE_RATE * time_to_expiry_years) * norm.cdf(d2)
        else: # 'P'
            price = strike_price * np.exp(-RISK_FREE_RATE * time_to_expiry_years) * norm.cdf(-d2) - spot_price * norm.cdf(-d1)
        return max(0.01, price)

    def run_backtest_60d(self):
        """Run backtest on historical Nifty 50 5-minute data max available 60 days."""
        logger.info(f"--- Starting YF 5-Min Native Expiry Backtest (Last 60 Days max) ---")
        
        ticker = yf.Ticker(self.symbol)
        
        logger.info(f"Requesting 5-minute data limit...")
        try:
            df_all = ticker.history(period="60d", interval="5m")
            if df_all.empty:
                logger.error("Failed to retrieve any 5m data.")
                return
        except Exception as e:
            logger.error(f"Failed to fetch historical data: {e}")
            return
            
        logger.info(f"Actually retrieved {len(df_all)} periods from {df_all.index[0].date()} to {df_all.index[-1].date()}")
        
        # Get all Thursdays in that range
        thursdays = self.get_past_thursdays(df_all)
        
        for current_date in thursdays:
            df = df_all[df_all['Date'] == current_date].copy()
            if df.empty:
                continue
                
            self.gamma_window.clear()
            in_position = False
            total_daily_pnl = 0
            
            # End of trading day (15:30 IST)
            end_of_day_time = df.index[-1].replace(hour=15, minute=30)
            
            entry_ce_strike = 0
            entry_pe_strike = 0
            entry_premium_paid = 0
            entry_lots = 0
            peak_pnl = 0
            trough_pnl = 0
            
            for index, row in df.iterrows():
                spot_price = row['Close']
                hour = index.hour
                minute = index.minute
                
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
                    current_ce_price = self.estimate_option_price(spot_price, entry_ce_strike, dte, IMPLIED_VOL_ASSUMPTION, 'C')
                    current_pe_price = self.estimate_option_price(spot_price, entry_pe_strike, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    current_strangle_value = current_ce_price + current_pe_price
                    
                    unrealized_pnl_points = entry_premium_paid - current_strangle_value # SHORT Strangle
                    peak_pnl = max(peak_pnl, unrealized_pnl_points)
                    trough_pnl = min(trough_pnl, unrealized_pnl_points)
                    
                    # Check 30% Stoploss
                    if current_strangle_value >= entry_premium_paid * (1 + self.stop_loss_pct):
                        pnl_points = unrealized_pnl_points
                        pnl_inr = pnl_points * self.lot_size * entry_lots
                        self.current_capital += pnl_inr
                        total_daily_pnl += pnl_points
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
                            'PnL_Points': round(pnl_points, 2),
                            'PnL_INR': round(pnl_inr, 2),
                            'Capital': round(self.current_capital, 2),
                            'Lots': entry_lots,
                            'Win': False,
                            'Exit_Reason': 'StopLoss'
                        })
                        in_position = False
                        break # Limit 1 Loss per day logic since we are taking directional Gamma bets
                    
                    # Exit near close (e.g. 15:15 5-min bar)
                    if hour == 15 and minute >= 15:
                        pnl_points = unrealized_pnl_points
                        pnl_inr = pnl_points * self.lot_size * entry_lots
                        self.current_capital += pnl_inr
                        total_daily_pnl += pnl_points
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
                            'PnL_Points': round(pnl_points, 2),
                            'PnL_INR': round(pnl_inr, 2),
                            'Capital': round(self.current_capital, 2),
                            'Lots': entry_lots,
                            'Win': pnl_points > 0,
                            'Exit_Reason': 'Time'
                        })
                        in_position = False
                        continue
                
                # Strategy logic - Original Gamma Spike Detection (Long Straddle / Short Strangle after 1:00 PM)
                if current_gamma > 0 and not in_position and hour >= 13:
                    if len(self.gamma_window) >= 10: # Wait for trailing base window to fill
                        baseline_gamma = sum(self.gamma_window) / len(self.gamma_window)
                        if baseline_gamma > 0:
                            spike_ratio = current_gamma / baseline_gamma
                            
                            # If spike is > 30% over baseline, enter the trade!
                            if spike_ratio >= self.spike_threshold and hour < 15:
                                logger.info(f"[{current_date} {index.strftime('%H:%M')}] GAMMA SPIKE! Ratio {spike_ratio:.2f}")
                                in_position = True
                                entry_spot = spot_price
                                entry_time = index
                                entry_lots = max(1, int(self.current_capital // self.margin_per_lot))
                                
                                # Long Straddle (Gamma Explosion betting) requires changing margin/short logic
                                # For consistency with user's earlier rule change, we enter Short Strangle on Spikes:
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
        total_profit_inr = df['PnL_INR'].sum()
        
        max_drawdown = df['Max_Drawdown'].min() * self.lot_size 
        max_profit = df['Peak_Profit'].max() * self.lot_size
        avg_pnl_inr = df['PnL_INR'].mean()
        
        roi = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        md = f"""
### 📊 Yahoo Finance 5-Min Backtest (Nifty Thursday Expiries, >1:00 PM, Gamma Spike Strategy, 30% SL)

| Metric | Value |
|--------|-------|
| **Strategy** | Short Strangle (Spot +/- 100) on 30% Gamma Explosion |
| **Initial Capital** | ₹{self.initial_capital:,.2f} |
| **Final Capital** | ₹{self.current_capital:,.2f} |
| **Overall ROI** | {roi:.2f}% |
| **Total Trades Taken** | {total_trades} |
| **Wins / Losses** | {wins} / {losses} |
| **Accuracy** | {accuracy:.2f}% |
| **Overall Net Profit (INR)** | ₹{total_profit_inr:,.2f} |
| **Total Net Points** | {total_pnl_points:.2f} pts |
| **Average Trade PnL (INR)** | ₹{avg_pnl_inr:,.2f} |
| **Max Drawdown inside a trade (INR per Lot)** | ₹{max_drawdown:,.2f} |

<br/>

### 📝 Trade Log
| Date | Entry | Exit | Spot Entry | Spot Exit | Exit Reason | Strikes (CE/PE) | Lots | Net PnL (INR) | Capital |
|------|-------|------|-------------|------------|-------------|-----------------|------|---------------|---------|
"""
        for _, row in df.iterrows():
            md += f"| {row['Date']} | {row['Entry_Time']} | {row['Exit_Time']} | {row['Spot_Entry']} | {row['Spot_Exit']} | {row['Exit_Reason']} | {row['CE_Strike']} / {row['PE_Strike']} | {row['Lots']} | **₹{row['PnL_INR']:,.2f}** | ₹{row['Capital']:,.2f} |\n"
            
        # Write to markdown file so we can show it to user
        with open("/Users/anoop/.gemini/antigravity/brain/311c7cff-5d0e-40ca-b43a-de26854c129a/walkthrough.md", "a") as f:
            f.write("\n\n---\n\n" + md)
            
        print(md)

def main():
    logger.info("Starting Yahoo Finance 5-Min Gamma Backtester...")
    try:
        backtester = YF5MinGammaBacktester()
        backtester.run_backtest_60d()
    except Exception as e:
        logger.error(f"Fatal Error: {e}")

if __name__ == "__main__":
    main()
