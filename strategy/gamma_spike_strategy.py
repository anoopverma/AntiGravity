import os
import time
import logging
from collections import deque
from datetime import datetime
from dotenv import load_dotenv
from dhanhq import dhanhq

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "YOUR_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "YOUR_ACCESS_TOKEN")
PAPER_TRADE = os.getenv("PAPER_TRADE", "True").lower() == "true"

class NiftyGammaSpikeStrategy:
    def __init__(self, client_id, access_token):
        """Initialize the Dhan API client and strategy parameters."""
        self.dhan = dhanhq(str(client_id), str(access_token))
        
        self.nifty_security_id = "13"
        self.exchange_segment = self.dhan.INDEX # Index segment
        self.fno_segment = self.dhan.NSE_FNO
        
        # Gamma Spike parameters
        self.gamma_window = deque(maxlen=20)
        self.spike_threshold = 1.3  # 30% spike in gamma above moving average
        self.in_position = False
        
    def get_option_chain(self, expiry_date):
        """Fetch the current option chain for Nifty."""
        try:
            response = self.dhan.option_chain(
                under_security_id=int(self.nifty_security_id),
                under_exchange_segment=self.exchange_segment,
                expiry=expiry_date
            )
            if response.get("status") == "success":
                return response.get("data")
            else:
                logger.error(f"Failed to fetch option chain: {response}")
                return None
        except Exception as e:
            logger.error(f"Error fetching option chain: {e}")
            return None

    def extract_atm_gamma(self, chain_data):
        """Extract the Gamma for the ATM strike."""
        if not chain_data or "oc" not in chain_data:
            return None
            
        last_price = chain_data.get("last_price", 0)
        if last_price == 0:
            return None
            
        # Find ATM strike (closest to last_price)
        strikes = [float(s) for s in chain_data["oc"].keys()]
        if not strikes:
            return None
            
        atm_strike = min(strikes, key=lambda x: abs(x - last_price))
        atm_strike_str = f"{atm_strike:.6f}"
        
        # Average Call and Put Gamma
        ce_gamma = chain_data["oc"][atm_strike_str]["ce"].get("greeks", {}).get("gamma", 0)
        pe_gamma = chain_data["oc"][atm_strike_str]["pe"].get("greeks", {}).get("gamma", 0)
        
        # Avoid zero divisions or anomalies
        if ce_gamma > 0 and pe_gamma > 0:
            return (ce_gamma + pe_gamma) / 2
        return max(ce_gamma, pe_gamma)

    def detect_spike(self, current_gamma):
        """Detect if current Gamma is a significant spike over the baseline."""
        if current_gamma <= 0:
            return False
            
        if len(self.gamma_window) < 10:
            # Need more data points to establish a baseline
            self.gamma_window.append(current_gamma)
            logger.info(f"Building baseline... Current Gamma: {current_gamma:.6f}, count: {len(self.gamma_window)}")
            return False
            
        baseline_gamma = sum(self.gamma_window) / len(self.gamma_window)
        
        # Avoid division by zero
        if baseline_gamma == 0:
            self.gamma_window.append(current_gamma)
            return False
            
        spike_ratio = current_gamma / baseline_gamma
        
        logger.info(f"Current Gamma: {current_gamma:.6f}, Baseline: {baseline_gamma:.6f}, Ratio: {spike_ratio:.2f}")
        
        self.gamma_window.append(current_gamma)
        
        if spike_ratio >= self.spike_threshold:
            logger.warning(f"GAMMA SPIKE DETECTED! Ratio {spike_ratio:.2f} >= {self.spike_threshold}")
            return True
            
        return False

    def place_long_straddle(self, ce_security_id, pe_security_id, lot_size=65):
        """Place market BUY orders for ATM CE and PE (Long Straddle) for Gamma explosion."""
        if PAPER_TRADE:
            logger.info(f"[PAPER TRADE] Placed Long Straddle for CE: {ce_security_id}, PE: {pe_security_id}, Qty: {lot_size}")
            self.in_position = True
            return True
            
        try:
            # Buy Call
            ce_order = self.dhan.place_order(
                security_id=str(ce_security_id),
                exchange_segment=self.fno_segment,
                transaction_type=self.dhan.BUY,
                quantity=lot_size,
                order_type=self.dhan.MARKET,
                product_type=self.dhan.INTRA,
                price=0
            )
            logger.info(f"CE Buy Order Response: {ce_order}")
            
            # Buy Put
            pe_order = self.dhan.place_order(
                security_id=str(pe_security_id),
                exchange_segment=self.fno_segment,
                transaction_type=self.dhan.BUY,
                quantity=lot_size,
                order_type=self.dhan.MARKET,
                product_type=self.dhan.INTRA,
                price=0
            )
            logger.info(f"PE Buy Order Response: {pe_order}")
            
            self.in_position = True
            return True
        except Exception as e:
            logger.error(f"Error placing orders: {e}")
            return False

    def run_iteration(self, expiry_date):
        """Run a single iteration of the strategy: fetch data, check spike, trade."""
        if self.in_position:
            logger.info("Already in position. Monitoring for exit conditions...")
            # Exit logic would go here (e.g. Stop Loss, Target Profit, or End of Day)
            return

        chain_data = self.get_option_chain(expiry_date)
        current_gamma = self.extract_atm_gamma(chain_data)
        
        if current_gamma is not None:
            if self.detect_spike(current_gamma):
                logger.info("Executing Long Straddle Trade due to Gamma Spike!")
                
                # Fetch ATM security IDs to place trade
                last_price = chain_data.get("last_price", 0)
                strikes = [float(s) for s in chain_data["oc"].keys()]
                atm_strike = min(strikes, key=lambda x: abs(x - last_price))
                atm_strike_str = f"{atm_strike:.6f}"
                
                ce_sec_id = chain_data["oc"][atm_strike_str]["ce"].get("security_id")
                pe_sec_id = chain_data["oc"][atm_strike_str]["pe"].get("security_id")
                
                if ce_sec_id and pe_sec_id:
                    self.place_long_straddle(ce_sec_id, pe_sec_id)
                else:
                    logger.error("Could not find security IDs for ATM strike to place trade.")
        else:
            logger.warning("Could not extract Gamma from option chain data or current price is 0.")

def get_next_tuesday_expiry():
    """Returns the next Tuesday date formatted as YYYY-MM-DD for Nifty Expiry"""
    # Note: Currently Nifty expires on Thursdays. Assuming the user meant either FinNifty (Tuesday) 
    # or a custom scenario. We will just format a placeholder expiry date or find the next Tuesday.
    today = datetime.now()
    days_ahead = 1 - today.weekday() # Tuesday is 1
    if days_ahead <= 0: # Target day already happened this week
        days_ahead += 7
    next_tuesday = today.replace(day=today.day + days_ahead) if days_ahead > 0 else today
    # You may need to format this properly according to Dhan's exact expiry string requirements
    return next_tuesday.strftime("%Y-%m-%d")

def main():
    logger.info("Starting Nifty Expiry Day Gamma Spike Strategy...")
    strategy = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
    
    # We use a static expiry for demonstration, but you can use `get_next_tuesday_expiry()`
    current_expiry = "2024-10-31" 
    
    logger.info(f"Monitoring Gamma for Expiry: {current_expiry}")
    
    # Run the loop every 60 seconds (respecting rate limits)
    try:
        while True:
            strategy.run_iteration(current_expiry)
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user.")

if __name__ == "__main__":
    main()
