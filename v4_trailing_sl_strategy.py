import os
import time
import datetime
import logging
from dotenv import load_dotenv
from dhanhq import dhanhq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
PAPER_TRADE = os.getenv("PAPER_TRADE", "True").lower() == "true"

dhan = dhanhq(str(CLIENT_ID), str(ACCESS_TOKEN))

def get_live_data(dhan, target_expiry):
    """
    Mock integration for the strategy function to fetch live Dhan data.
    Now fully operational mapping to Nifty and Live India VIX via API.
    """
    spot = 0
    ce_p = 0
    pe_p = 0
    ce_vol = 0
    pe_vol = 0
    current_vix = 13.0 # Default fallback VIX
    
    today = datetime.datetime.now().strftime("%Y-%m-%d")

    try:
        # Fetch Real-Time India VIX (Security ID 21 on Dhan INDEX)
        vix_response = dhan.intraday_minute_data(
            security_id="21", 
            exchange_segment=dhan.INDEX, 
            instrument_type="INDEX", 
            from_date=today, 
            to_date=today
        )
        if vix_response.get("status") == "success" and vix_response.get("data"):
            vix_data = vix_response.get("data")
            if isinstance(vix_data, dict) and "close" in vix_data and len(vix_data["close"]) > 0:
                current_vix = float(vix_data["close"][-1])
            elif isinstance(vix_data, list) and len(vix_data) > 0 and "close" in vix_data[-1]:
                current_vix = float(vix_data[-1]["close"])
    except Exception as e:
        logger.error(f"Failed to fetch live India VIX: {e}. Using fallback {current_vix}")

    try:
        # Fetch Option Chain
        response = dhan.option_chain(
            under_security_id=13, # Nifty Index
            under_exchange_segment=dhan.INDEX,
            expiry=target_expiry
        )

        if response.get("status") == "success":
            data = response["data"]
            spot = data.get("last_price", 0)
            
            # Find ATM strike
            if spot > 0 and "oc" in data:
                strikes = [float(s) for s in data["oc"].keys()]
                if strikes:
                    atm_strike = min(strikes, key=lambda x: abs(x - spot))
                    atm_strike_str = f"{atm_strike:.6f}"
                    
                    strike_data = data["oc"][atm_strike_str]
                    
                    ce_p = strike_data["ce"].get("last_price", 0)
                    pe_p = strike_data["pe"].get("last_price", 0)
                    
                    # Using Open Interest or Volume depending on what's available
                    ce_vol = strike_data["ce"].get("volume", 1)  
                    pe_vol = strike_data["pe"].get("volume", 1)
    except Exception as e:
        logger.error(f"Failed to fetch Option Chain: {e}")

    return spot, ce_p, pe_p, ce_vol, pe_vol, current_vix

def trade_with_trailing_sl(target_expiry):
    # --- ENHANCED CONFIGURATION ---
    VIX_THRESHOLD = 12.5
    VOL_MULTIPLIER = 1.5
    TARGET_LOCK_IN = 0.30    # At 30% profit, start trailing
    TRAILING_STEP = 0.15     # Trail by 15% behind the peak price
    INITIAL_SL = 0.40        # 40% strict initial option stop loss
    
    benchmark_straddle = None
    position = None          # Stores { 'type': 'CE'/'PE', 'entry': 0.0, 'peak': 0.0 }

    print("===== V4 ACTIVE: VIX + Volume + 1:45 PM Window + Trailing SL =====")
    
    if PAPER_TRADE:
        print("[MODE] PAPER TRADING ENABLED")
    else:
        print("[MODE] LIVE TRADING WARNING - Ensure VIX feed is accurate")

    poll_interval = 60 # Check every 60 seconds

    while True:
        now = datetime.datetime.now()
        
        try:
            spot, ce_p, pe_p, ce_vol, pe_vol, current_vix = get_live_data(dhan, target_expiry)
            
            if spot == 0:
                print(f"[{now.strftime('%H:%M:%S')}] Awaiting Market Data...")
                time.sleep(poll_interval)
                continue

            # 1. SET BENCHMARK (1:45 PM)
            if now.hour == 13 and now.minute >= 45 and benchmark_straddle is None:
                benchmark_straddle = ce_p + pe_p
                benchmark_spot = spot
                print(f"[{now.strftime('%H:%M:%S')}] Benchmark Straddle Set at: {benchmark_straddle:.2f}")

            # For testing/paper trading, automatically set a benchmark if we skipped 1:45 PM
            if PAPER_TRADE and benchmark_straddle is None and (now.hour > 13 or (now.hour == 13 and now.minute > 45)):
                benchmark_straddle = ce_p + pe_p
                benchmark_spot = spot
                print(f"[{now.strftime('%H:%M:%S')}] [TEST] Late Benchmark Straddle Auto-Set at: {benchmark_straddle:.2f}")

            # 2. ENTRY LOGIC
            if benchmark_straddle and position is None:
                current_straddle = ce_p + pe_p
                
                print(f"[{now.strftime('%H:%M:%S')}] Monitoring Entry | ATM CE: {ce_p} PE: {pe_p} | Straddle: {current_straddle:.2f} (Target: {benchmark_straddle * 1.15:.2f})")
                
                # Setup Trend Filter Logic
                pct_change_from_benchmark = ((spot - benchmark_spot) / benchmark_spot) * 100
                trend_confirmed = abs(pct_change_from_benchmark) >= 0.15

                if (current_straddle >= (benchmark_straddle * 1.15) or trend_confirmed) and current_vix >= VIX_THRESHOLD:
                    
                    # Volume Check
                    if ce_p > pe_p and ce_vol > (pe_vol * VOL_MULTIPLIER):
                        position = {'type': 'CE', 'entry': ce_p, 'peak': ce_p}
                    elif pe_p > ce_p and pe_vol > (ce_vol * VOL_MULTIPLIER):
                        position = {'type': 'PE', 'entry': pe_p, 'peak': pe_p}
                    
                    if position:
                        print(f"[{now.strftime('%H:%M:%S')}] >>> STRATEGY ENTRY TRIGGERED <<<")
                        print(f"ENTRY: {position['type']} at {position['entry']}")
                        
                        if PAPER_TRADE:
                            print(f"[PAPER] Bought NIFTY ATM {position['type']} at {position['entry']}")
                        else:
                            print(f"[LIVE] Trigger live buy for {position['type']}...")

            # 3. TRAILING EXIT LOGIC
            elif position:
                current_price = ce_p if position['type'] == 'CE' else pe_p
                
                # Update Peak Price
                if current_price > position['peak']:
                    position['peak'] = current_price
                
                # Calculate Dynamic SL
                # If profit > 30%, SL = Peak - 15%. Otherwise, SL = Entry - 40%
                if position['peak'] >= position['entry'] * (1 + TARGET_LOCK_IN):
                    current_sl = position['peak'] * (1 - TRAILING_STEP)
                    sl_type = "Trailing SL"
                else:
                    current_sl = position['entry'] * (1 - INITIAL_SL) # Initial robust SL block
                    sl_type = "Initial SL"
                    
                pnl_current = ((current_price - position['entry']) / position['entry']) * 100
                print(f"[{now.strftime('%H:%M:%S')}] POSITION ACTIVE [{position['type']}] | Curr: {current_price} | Peak: {position['peak']} | SL ({sl_type}): {current_sl:.2f} | P/L: {pnl_current:.2f}%")

                # Check for Exit
                if current_price <= current_sl:
                    pnl = ((current_price - position['entry']) / position['entry']) * 100
                    print(f"[{now.strftime('%H:%M:%S')}] >>> EXIT TRIGGERED <<<")
                    print(f"EXIT: {sl_type} Hit at {current_price} | Final P/L: {pnl:.2f}%")
                    
                    if PAPER_TRADE:
                        print("[PAPER] Position closed.")
                    else:
                        print("[LIVE] Trigger live sell...")
                        
                    break # Strategy ends for the day

            # Exit logic based on time
            if now.hour >= 15 and now.minute >= 25:
                print(f"[{now.strftime('%H:%M:%S')}] Time End. Force closing positions.")
                if position:
                    current_price = ce_p if position['type'] == 'CE' else pe_p
                    pnl = ((current_price - position['entry']) / position['entry']) * 100
                    print(f"EXIT: End of Day Time limit hit at {current_price} | Final P/L: {pnl:.2f}%")
                    if PAPER_TRADE:
                        print("[PAPER] Position closed.")
                break
            
        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}")
            
        time.sleep(poll_interval)

if __name__ == "__main__":
    TARGET_EXPIRY = "2026-03-02"
    trade_with_trailing_sl(TARGET_EXPIRY)
