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
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for Black-Scholes Approximation
RISK_FREE_RATE = 0.07  # Assume 7% risk-free rate in India
IMPLIED_VOL_ASSUMPTION = 0.15  # Assume a constant 15% IV for approximation
TIME_TO_EXPIRY_END = 1.0 / (365 * 24 * 60) # 1 minute before expiry

class BacktestGammaYfStrategy:
    def __init__(self):
        self.symbol = "^NSEI" # Nifty 50 symbol on Yahoo Finance
        self.gamma_window = deque(maxlen=20)
        self.spike_threshold = 1.3
        
    def get_past_tuesdays(self, num_weeks=4):
        """Get the dates of the last `num_weeks` Tuesdays."""
        today = datetime.now()
        offset = (today.weekday() - 1) % 7
        last_tuesday = today - timedelta(days=offset)
        
        tuesdays = []
        for i in range(num_weeks):
            t_date = last_tuesday - timedelta(weeks=i)
            # yfinance needs the next day as the end period to get full day data
            next_day = t_date + timedelta(days=1)
            tuesdays.append((t_date.strftime("%Y-%m-%d"), next_day.strftime("%Y-%m-%d")))
        return tuesdays

    def approximate_atm_gamma(self, spot_price, strike_price, time_to_expiry_years, iv):
        """Estimate ATM Gamma using Black-Scholes formula."""
        if time_to_expiry_years <= 0 or spot_price <= 0 or iv <= 0:
            return 0.0
            
        d1 = (np.log(spot_price / strike_price) + (RISK_FREE_RATE + 0.5 * iv**2) * time_to_expiry_years) / (iv * np.sqrt(time_to_expiry_years))
        gamma = norm.pdf(d1) / (spot_price * iv * np.sqrt(time_to_expiry_years))
        return gamma

    def run_backtest(self, start_date, end_date):
        """Run backtest on historical Nifty 50 minute data for a specific day."""
        logger.info(f"--- Running Backtest for Expiry Date: {start_date} ---")
        
        try:
            # Fetch 1-minute interval data for Nifty 50
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(start=start_date, end=end_date, interval="1m")
            
            if df.empty:
                logger.warning(f"No data available for {start_date}. Note: yfinance limits 1m data to the last 30 days.")
                return
                
            logger.info(f"Fetched {len(df)} 1-minute data points.")
            
            in_position = False
            entry_price = 0
            entry_time = None
            total_pnl = 0
            
            self.gamma_window.clear()
            
            # End of trading day (15:30 IST)
            end_of_day_time = df.index[-1]
            
            for index, row in df.iterrows():
                spot_price = row['Close']
                
                # Approximate Time to Expiry (in years)
                time_left = end_of_day_time - index
                minutes_left = max(1, time_left.total_seconds() / 60)
                dte = minutes_left / (365 * 24 * 60)
                
                # Assume ATM strike is the closest 50-point interval
                atm_strike = round(spot_price / 50) * 50
                
                # Calculate Approximation for Gamma
                current_gamma = self.approximate_atm_gamma(spot_price, atm_strike, dte, IMPLIED_VOL_ASSUMPTION)
                
                # Exiting at the end of the day if in position
                if in_position and minutes_left <= 15:
                    exit_price = spot_price
                    # Simulate Long Straddle payoff (simplified)
                    # Absolute movement of index from entry strike represents option payoff
                    price_movement = abs(exit_price - entry_price) 
                    premium_paid = entry_price * 0.01 # Simplified premium assumption
                    pnl = price_movement - premium_paid
                    total_pnl += pnl
                    logger.info(f"[{index.strftime('%H:%M')}] EXIT Long Straddle. Exit Price: {exit_price:.2f} | Movement: {price_movement:.2f} | Approx PnL: {pnl:.2f}")
                    in_position = False
                    continue
                
                # Strategy logic
                if current_gamma > 0 and not in_position:
                    if len(self.gamma_window) >= 10:
                        baseline_gamma = sum(self.gamma_window) / len(self.gamma_window)
                        if baseline_gamma > 0:
                            spike_ratio = current_gamma / baseline_gamma
                            
                            # Log every 60 mins just to show progress
                            if index.minute == 0:
                                logger.info(f"[{index.strftime('%H:%M')}] Spot: {spot_price:.2f}, Gamma: {current_gamma:.6f}, Ratio: {spike_ratio:.2f}")
                            
                            if spike_ratio >= self.spike_threshold and minutes_left > 30:
                                logger.warning(f"[{index.strftime('%H:%M')}] GAMMA SPIKE DETECTED! Ratio: {spike_ratio:.2f} >= {self.spike_threshold}")
                                logger.info(f"[{index.strftime('%H:%M')}] Executing Long Straddle at Spot: {spot_price:.2f} (ATM Strike: {atm_strike})")
                                
                                # Enter Trade
                                in_position = True
                                entry_price = spot_price
                                entry_time = index
                    
                    self.gamma_window.append(current_gamma)

            logger.info(f"--- Day Total PnL points (estimated): {total_pnl:.2f} ---\n")
                
        except Exception as e:
            logger.error(f"Error during backtest: {e}")

def main():
    logger.info("Starting Gamma Spike yfinance Backtester...")
    backtester = BacktestGammaYfStrategy()
    
    # We can fetch 1m data for max 7 days in yfinance reliably. Let's do the last Tuesday.
    tuesdays = backtester.get_past_tuesdays(1) # Run for 1 most recent Tuesday
    
    for start_date, end_date in tuesdays:
        backtester.run_backtest(start_date, end_date)
        time.sleep(2) # Prevent blocking by API

if __name__ == "__main__":
    main()
