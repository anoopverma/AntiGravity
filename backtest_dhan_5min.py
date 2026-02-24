import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
from dhanhq import dhanhq
from scipy.stats import norm
import sqlalchemy
from sqlalchemy import create_engine
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
            
        self.dhan = dhanhq(str(self.client_id), str(self.access_token))
        
        # Strategy Parameters
        self.initial_capital = 500000
        self.current_capital = self.initial_capital
        self.margin_per_lot = 120000
        self.lot_size = 75 # Based on latest Nifty lots
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
        offset = (today.weekday() - 1) % 7
        last_tuesday = today - timedelta(days=offset)
        if today.weekday() < 1: 
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
            start_date = date_str
            end_date = (datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
            df = ticker.history(start=start_date, end=end_date, interval="5m")
            if not df.empty:
                df.columns = [c.lower() for c in df.columns]
                logger.info(f"Fallback: Fetched {len(df)} 5-min bars from YFinance for {date_str}")
                return df
        except Exception as e:
            logger.error(f"YFinance fallback failed for {date_str}: {e}")
        return pd.DataFrame()

    def fetch_dhan_5min_data(self, date_str, retries=3):
        """Fetch intraday data from Dhan."""
        for attempt in range(retries):
            try:
                req = self.dhan.intraday_minute_data(
                    security_id='13', 
                    exchange_segment=self.dhan.INDEX,
                    instrument_type='INDEX',
                    from_date=date_str,
                    to_date=date_str
                )
                
                if req.get('status') == 'success' and req.get('data'):
                    df = pd.DataFrame(req['data'])
                    if df.empty: return pd.DataFrame()
                    
                    time_col = 'timestamp' if 'timestamp' in df.columns else 'start_Time'
                    if time_col in df.columns:
                        if time_col == 'timestamp':
                            df['datetime'] = pd.to_datetime(df[time_col], unit='s') + pd.Timedelta(hours=5, minutes=30)
                        else:
                            df['datetime'] = pd.to_datetime(df[time_col]) + pd.Timedelta(hours=5, minutes=30)
                        
                        df.set_index('datetime', inplace=True)
                        df_5m = df.resample('5min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna()
                        return df_5m
                elif req.get('remarks') and 'DH-904' in str(req.get('remarks')):
                    time.sleep(5)
                else: break
            except Exception: break
        return self.fetch_yf_5min_fallback(date_str)

    def run_v4_backtest(self, vix_threshold=12.5, target_lock_in=0.30, trailing_step=0.15, initial_sl=0.50, expansion_threshold=1.15, trend_filter_pct=0.15):
        tuesdays = self.get_last_n_tuesdays(12)
        logger.info(f"Starting V4 Trailing SL Strategy Backtest (Enhanced) - Last {len(tuesdays)} Weeks")
        self.results = []
        self.current_capital = self.initial_capital
        current_vix = 13.0
        
        for date_str in tuesdays:
            df = self.cached_data.get(date_str)
            if df is None:
                df = self.fetch_dhan_5min_data(date_str)
                self.cached_data[date_str] = df
            
            if df is None or df.empty: continue
            
            position = None
            benchmark_straddle = 0.0
            benchmark_spot = 0.0
            entry_time = None
            
            eod_time = df.index[-1].replace(hour=15, minute=30)
            
            for index, row in df.iterrows():
                spot_price = float(row['close'])
                dte = (eod_time - index).total_seconds() / (365 * 24 * 3600)
                
                if index.hour == 13 and index.minute == 45:
                    benchmark_spot = spot_price
                    atm = round(spot_price / 50) * 50
                    benchmark_straddle = self.estimate_option_price(spot_price, atm, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                         self.estimate_option_price(spot_price, atm, dte, IMPLIED_VOL_ASSUMPTION, 'P')

                is_entry_window = (index.hour == 14) 
                
                if benchmark_straddle > 0 and position is None and is_entry_window:
                    atm = round(benchmark_spot / 50) * 50
                    curr_straddle = self.estimate_option_price(spot_price, atm, dte, IMPLIED_VOL_ASSUMPTION, 'C') + \
                                    self.estimate_option_price(spot_price, atm, dte, IMPLIED_VOL_ASSUMPTION, 'P')
                                    
                    trend_pct = ((spot_price - benchmark_spot) / benchmark_spot) * 100
                    trend_ok = abs(trend_pct) >= trend_filter_pct
                    momentum_ok = abs(row['high'] - row['low']) >= (spot_price * 0.001)

                    if (curr_straddle >= (benchmark_straddle * expansion_threshold) or trend_ok) and current_vix >= vix_threshold and momentum_ok:
                        opt_type = 'C' if spot_price > benchmark_spot else 'P'
                        strike = round(spot_price / 50) * 50
                        entry_prem = self.estimate_option_price(spot_price, strike, dte, IMPLIED_VOL_ASSUMPTION, opt_type)
                        
                        target_inv = self.initial_capital * 0.10
                        qty = max(self.lot_size, int(target_inv // (max(1.0, entry_prem) * self.lot_size)) * self.lot_size)
                        
                        position = {'type': opt_type, 'strike': strike, 'entry': entry_prem, 'peak': entry_prem, 'qty': qty}
                        entry_time = index

                elif position:
                    curr_p = self.estimate_option_price(spot_price, position['strike'], dte, IMPLIED_VOL_ASSUMPTION, position['type'])
                    if curr_p > position['peak']: position['peak'] = curr_p
                        
                    profit_pct = (position['peak'] - position['entry']) / position['entry']
                    
                    if profit_pct >= 1.00: 
                        sl = position['peak'] * 0.90
                        reason = "Super Trail (10%)"
                    elif profit_pct >= 0.40: 
                        sl = position['peak'] * 0.85
                        reason = "Strong Trail (15%)"
                    elif profit_pct >= 0.20: 
                        sl = position['entry']
                        reason = "Break-Even SL"
                    else:
                        sl = position['entry'] * (1 - initial_sl)
                        reason = "Initial SL"
                        
                    time_exit = index.hour == 15 and index.minute >= 25
                    if curr_p <= sl or time_exit:
                        reason = "Time Exit" if time_exit else reason
                        pnl = (curr_p - position['entry']) * position['qty']
                        self.current_capital += pnl
                        self.results.append({
                            'Date': date_str,
                            'Entry_Time': entry_time.strftime("%H:%M:%S") if entry_time else "00:00:00",
                            'Exit_Time': index.strftime("%H:%M:%S"),
                            'Option_Type': position['type'],
                            'Buy_Price': round(position['entry'], 2),
                            'Sell_Price': round(curr_p, 2),
                            'PnL_INR': round(pnl, 2),
                            'ROI%': round(((curr_p - position['entry']) / position['entry']) * 100, 2),
                            'Reason': reason,
                            'Win': pnl > 0
                        })
                        position = None
                        break
        
        self.print_summary_v4()
        self.save_to_postgres(table_name="historical_backtests", strategy_name="V4_Trailing_SL")

    def print_summary_v4(self):
        if not self.results:
            print("\nNo Trades Executed.")
            return
        df = pd.DataFrame(self.results)
        total_pnl = df['PnL_INR'].sum()
        roi = (total_pnl / self.initial_capital) * 100
        print(f"\n=== V4 STRATEGY BACKTEST RESULTS ===\nTotal Return: {roi:.2f}% (₹{total_pnl:,.2f})\nTrades: {len(df)}\nWin Rate: {(df['Win'].sum()/len(df))*100:.2f}%\n")
        print(df.to_string(index=False))

    def save_to_postgres(self, table_name="historical_backtests", strategy_name="Unknown"):
        uri = os.getenv("POSTGRES_URI", "postgresql://postgres:Aidni%40%23123@localhost:5432/postgres")
        if not self.results: return
        try:
            engine = create_engine(uri)
            df = pd.DataFrame(self.results)
            df.insert(0, 'Run_Date', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            df.insert(1, 'Strategy_Name', strategy_name)
            df.to_sql(table_name, con=engine, if_exists='replace', index=False)
            print(f"-> DB SYNC: {strategy_name} results REPLACED in {table_name}.")
        except Exception as e:
            logger.error(f"DB Error: {e}")

if __name__ == "__main__":
    backtester = NiftyTuesdayDhanBacktester()
    backtester.run_v4_backtest()
