import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dhanhq import dhanhq
from scipy.stats import norm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for Black-Scholes Approximation
RISK_FREE_RATE = 0.07  
IMPLIED_VOL_ASSUMPTION = 0.15  

class V4Backtester:
    def __init__(self):
        load_dotenv()
        # Support both standard and Render env var names
        self.client_id = os.getenv('DHAN_CLIENT_ID') or os.getenv('DHAN_API_KEY')
        self.access_token = os.getenv('DHAN_ACCESS_TOKEN') or os.getenv('DHAN_CLIENT_SECRET')
        
        if not self.client_id or not self.access_token:
            raise ValueError("Dhan API credentials (ID/Key or Token/Secret) not found in environment")
            
        self.dhan = dhanhq(str(self.client_id), str(self.access_token))
        
        # Strategy Parameters
        self.initial_capital = 500000
        self.current_capital = self.initial_capital
        self.lot_size = 25
        self.results = []
        self.cached_data = {} 
        
    def estimate_option_price(self, spot_price, strike_price, time_to_expiry_years, iv, option_type):
        """Estimate Black Scholes price for options."""
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
        offset = (today.weekday() - 1) % 7
        last_tuesday = today - timedelta(days=offset)
        if today.weekday() < 1: 
            last_tuesday = last_tuesday - timedelta(days=7)
        
        tuesdays = []
        for i in range(n):
            tuesdays.append((last_tuesday - timedelta(days=7*i)).strftime("%Y-%m-%d"))
        return sorted(tuesdays)

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

    def run_v4_backtest(self, vix_threshold=12.5, target_lock_in=0.20, trailing_step=0.10):
        tuesdays = self.get_last_n_tuesdays(12)
        logger.info(f"Starting V4 Trailing SL Strategy Backtest for Last 12 Weeks")
        self.results = []
        self.current_capital = self.initial_capital
        
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
            benchmark_spot = None
            entry_time, entry_spot, entry_strike, entry_premium = None, 0, 0, 0
            entry_qty = 0
            eod_time = df.index[-1].replace(hour=15, minute=30)
            
            # Note ATM Strikes recorded at 1:30 PM
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
                    benchmark_straddle = self.estimate_option_price(spot_price, atm_ce_130, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                         self.estimate_option_price(spot_price, atm_pe_130, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                    #logger.info(f"[{date_str}] 1:30 PM Benchmark Straddle: {benchmark_straddle:.2f}")

                # 2. Monitor for Entry
                if benchmark_straddle and position is None and index.time() > datetime.strptime("13:30", "%H:%M").time():
                    curr_straddle = self.estimate_option_price(spot_price, atm_ce_130, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                    self.estimate_option_price(spot_price, atm_pe_130, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                                    
                    if curr_straddle >= (benchmark_straddle * 1.20) and current_vix >= vix_threshold:
                        # Emulate Volume/Direction check using Point Differential
                        opt_type = 'C' if spot_price > benchmark_spot else 'P'
                        target_strike = round(spot_price / 50) * 50 # New ATM
                        entry_premium = self.estimate_option_price(spot_price, target_strike, dte, IMPLIED_VOL_ASSUMPTION, opt_type)
                        
                        position = {
                            'type': opt_type, 
                            'strike': target_strike,
                            'entry': entry_premium, 
                            'peak': entry_premium,
                            'qty': max(self.lot_size, int(self.current_capital // (entry_premium * self.lot_size)) * self.lot_size)
                        }
                        
                        entry_time, entry_spot = index, spot_price
                        #logger.info(f"[{date_str}] ENTRY {opt_type} @ {entry_premium:.2f}")

                # 3. Handle Active Position (Trailing SL / Time Exit)
                elif position:
                    current_price = self.estimate_option_price(spot_price, position['strike'], dte, IMPLIED_VOL_ASSUMPTION, position['type'])
                    
                    if current_price > position['peak']:
                        position['peak'] = current_price
                        
                    if position['peak'] >= position['entry'] * (1 + target_lock_in):
                        current_sl = position['peak'] * (1 - trailing_step)
                        reason = "Trailing SL"
                    else:
                        current_sl = position['entry'] * 0.70
                        reason = "Initial SL"
                        
                    time_exit = index.hour == 15 and index.minute >= 25
                    
                    if current_price <= current_sl or time_exit:
                        reason = "Time Exit" if time_exit else reason
                        pnl_points = current_price - position['entry']
                        pnl_inr = pnl_points * position['qty']
                        self.current_capital += pnl_inr
                        
                        self.results.append({
                            'Date': date_str,
                            'Entry_Time': entry_time.strftime("%H:%M"),
                            'Exit_Time': index.strftime("%H:%M"),
                            'Type': position['type'],
                            'Entry_Price': round(position['entry'], 2),
                            'Peak_Price': round(position['peak'], 2),
                            'Exit_Price': round(current_price, 2),
                            'PnL_INR': round(pnl_inr, 2),
                            'ROI%': round((pnl_points / position['entry']) * 100, 2),
                            'Reason': reason
                        })
                        #logger.info(f"[{date_str}] EXIT via {reason} @ {current_price:.2f} | PnL: ₹{pnl_inr:.2f}")
                        position = None
                        break # Done for the day

            time.sleep(0.1)
        
        self.print_summary()

    def print_summary(self):
        if not self.results:
            print("\nNo Trades Executed in this Period.")
            return
            
        df = pd.DataFrame(self.results)
        df['Win'] = df['PnL_INR'] > 0
        total_pnl = df['PnL_INR'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        
        print("\n=== V4 STRATEGY 12-WEEK BACKTEST RESULTS ===")
        print(f"Total Return: {roi:.2f}% (₹{total_pnl:,.2f})")
        print(f"Total Trades: {len(df)}")
        print(f"Win Rate: {(df['Win'].sum() / len(df)) * 100:.2f}%\n")
        print(df.to_string(index=False))

if __name__ == "__main__":
    backtester = V4Backtester()
    backtester.run_v4_backtest()
