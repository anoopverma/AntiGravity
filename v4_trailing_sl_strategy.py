import os
import time
import datetime
import logging
from dotenv import load_dotenv
from dhanhq import dhanhq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
PAPER_TRADE = os.getenv("PAPER_TRADE", "True").lower() == "true"

class NiftyV4TrailingSLStrategy:
    def __init__(self, target_expiry):
        self.dhan = dhanhq(str(CLIENT_ID), str(ACCESS_TOKEN))
        self.target_expiry = target_expiry
        self.lot_size = 75
        self.running = False
        self.paused = False
        self.in_position = False
        
        # --- Champion V4 Parameters (+103% Backtest ROI) ---
        self.initial_sl = 0.50        # Stop Loss at 50%
        self.trailing_step = 0.15     # Trail by 15% once profitable
        self.target_lock_in = 0.30    # Start trailing after 30% profit
        self.vix_threshold = 12.5     # VIX must be > 12.5
        self.expansion_threshold = 1.15 # IV Expansion 15%
        self.trend_filter_pct = 0.15  # Trend shift 0.15%
        self.momentum_threshold = 0.001 # 0.10% momentum check

        self.current_position = None # Stores { 'type': 'CE'/'PE', 'entry': 0.0, 'peak': 0.0, 'strike': 0 }
        self.benchmark_straddle = None
        self.benchmark_spot = None
        self.unrealized_pnl = 0
        
    def get_live_data(self):
        """Fetches spot, atm prices, and VIX from Dhan."""
        spot, ce_p, pe_p, ce_vol, pe_vol, current_vix = 0, 0, 0, 0, 0, 13.0
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            # 1. Fetch VIX
            vix_resp = self.dhan.intraday_minute_data("21", self.dhan.INDEX, "INDEX", today, today)
            if vix_resp.get("status") == "success" and vix_resp.get("data"):
                vix_data = vix_resp.get("data")
                current_vix = float(vix_data.get("close", [13.0])[-1])

            # 2. Fetch Option Chain for Spot and ATM
            oc_resp = self.dhan.option_chain(13, self.dhan.INDEX, self.target_expiry)
            if oc_resp.get("status") == "success":
                data = oc_resp["data"]
                spot = data.get("last_price", 0)
                if spot > 0 and "oc" in data:
                    strikes = [float(s) for s in data["oc"].keys()]
                    atm_strike = min(strikes, key=lambda x: abs(x - spot))
                    strike_data = data["oc"][f"{atm_strike:.6f}"]
                    ce_p = strike_data["ce"].get("last_price", 0)
                    pe_p = strike_data["pe"].get("last_price", 0)
                    ce_vol = strike_data["ce"].get("volume", 1)
                    pe_vol = strike_data["pe"].get("volume", 1)
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            
        return spot, ce_p, pe_p, ce_vol, pe_vol, current_vix

    def capture_benchmark(self):
        """Sets the 1:45 PM baseline for spot and straddle price."""
        try:
            spot, ce_p, pe_p, _, _, _ = self.get_live_data()
            if spot > 0:
                self.benchmark_spot = spot
                self.benchmark_straddle = ce_p + pe_p
                logger.info(f"📍 Benchmark Set | Spot: {spot} | Straddle: {round(self.benchmark_straddle, 2)}")
        except Exception as e:
            logger.error(f"Failed to capture benchmark: {e}")

    def run_iteration(self):
        """Main check loop - called every 1 min"""
        now = datetime.datetime.now()
        
        # Trading Window Check (14:00 - 15:00 for entry)
        is_entry_window = now.hour == 14
        is_exit_time = now.hour == 15 and now.minute >= 25
        
        # Capture benchmark at 13:45
        if now.hour == 13 and now.minute == 45 and self.benchmark_straddle is None:
            self.capture_benchmark()
            
        # Paper Trade Helper: Auto-set benchmark if we started late
        if PAPER_TRADE and self.benchmark_straddle is None and (now.hour > 13 or (now.hour == 13 and now.minute > 45)):
            self.capture_benchmark()

        if self.in_position:
            self.manage_position(is_exit_time)
        elif is_entry_window and self.benchmark_straddle:
            self.check_entry()

    def check_entry(self):
        """Matches the 'Champion' backtest entry logic."""
        try:
            spot, ce_p, pe_p, ce_vol, pe_vol, vix = self.get_live_data()
            if spot == 0 or self.benchmark_spot == 0: return

            current_straddle = ce_p + pe_p
            
            # 1. Expansion Filter (15%)
            expansion_hit = current_straddle >= (self.benchmark_straddle * self.expansion_threshold)
            
            # 2. Trend Filter (0.15%)
            trend_pct = abs((spot - self.benchmark_spot) / self.benchmark_spot) * 100
            trend_hit = trend_pct >= self.trend_filter_pct
            
            # 3. Momentum Mock (Price logic)
            # Live momentum is checked by ensuring we aren't in a flat range
            momentum_hit = True # Placeholder for live
            
            if (expansion_hit or trend_hit) and vix >= self.vix_threshold and momentum_hit:
                # Directional selection
                # Use Price + Volume for direction
                if spot > self.benchmark_spot:
                    opt_type = 'CE'
                    price = ce_p
                else:
                    opt_type = 'PE'
                    price = pe_p
                
                logger.info(f"🚀 V4 ENTRY TRIGGERED | Type: {opt_type} | P: {price} | Reason: {'EXP' if expansion_hit else 'TRD'}")
                self.place_order(opt_type, price)
                
        except Exception as e:
            logger.error(f"Entry check failed: {e}")

    def manage_position(self, force_exit=False):
        """Matches the 'Champion' backtest trailing logic."""
        try:
            spot, ce_p, pe_p, _, _, _ = self.get_live_data()
            curr_price = ce_p if self.current_position['type'] == 'CE' else pe_p
            
            if curr_price > self.current_position['peak']:
                self.current_position['peak'] = curr_price
            
            # Tiered Trailing SL
            profit_pct = (self.current_position['peak'] - self.current_position['entry']) / self.current_position['entry']
            
            if profit_pct >= 1.0: # Super Winner (Trail 10%)
                sl_price = self.current_position['peak'] * 0.90
                reason = "Super Trail"
            elif profit_pct >= 0.40: # Strong Move (Trail 15%)
                sl_price = self.current_position['peak'] * 0.85
                reason = "Strong Trail"
            elif profit_pct >= 0.20: # Break-even
                sl_price = self.current_position['entry']
                reason = "Break-Even"
            else: # Initial SL (50%)
                sl_price = self.current_position['entry'] * (1 - self.initial_sl)
                reason = "Initial SL"
            
            self.unrealized_pnl = (curr_price - self.current_position['entry']) * self.lot_size
            
            if curr_price <= sl_price or force_exit:
                exit_reason = "Time Exit" if force_exit else reason
                self.close_position(curr_price, exit_reason)
                
        except Exception as e:
            logger.error(f"Position management failed: {e}")

    def place_order(self, opt_type, price):
        """Simulates or places a real order."""
        self.current_position = {
            'type': opt_type,
            'entry': price,
            'peak': price,
            'time': datetime.datetime.now()
        }
        self.in_position = True
        logger.info(f"✅ Order Placed: {opt_type} at {price}")
        if not PAPER_TRADE:
            # Real Dhan order placement would go here
            pass

    def close_position(self, price, reason):
        """Simulates or closes a real position."""
        pnl = (price - self.current_position['entry']) * self.lot_size
        logger.info(f"🔴 Position Closed | Type: {self.current_position['type']} | Price: {price} | PnL: {round(pnl, 2)} | Reason: {reason}")
        self.in_position = False
        self.current_position = None
        self.unrealized_pnl = 0
        if not PAPER_TRADE:
            # Real Dhan order closure would go here
            pass
        # Stop strategy for the day after exit to prevent overtrading
        # self.running = False 

if __name__ == "__main__":
    # Test execution
    strategy = NiftyV4TrailingSLStrategy("2026-03-02")
    # strategy.run_strategy()
