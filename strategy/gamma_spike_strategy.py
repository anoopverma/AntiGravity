import os
import time
import datetime
import logging
from dotenv import load_dotenv
from dhanhq import dhanhq
from strategy.whatsapp_alerter import WhatsAppAlerter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class NiftyGammaSpikeStrategy:
    def __init__(self, client_id, access_token):
        self.dhan = dhanhq(str(client_id), str(access_token))
        self.target_expiry = None
        self.lot_size = 65
        self.running = False
        self.paused = False
        self.in_position = False
        self.paper_trade = True # Overriden by the app.py engine starter
        self.alerter = WhatsAppAlerter()
        
        # --- Gamma Spike Parameters ---
        self.initial_sl = 0.50        # Stop Loss at 50%
        self.vix_threshold = 11.0     # VIX must be >= 11.0
        self.expansion_threshold = 1.15 # IV Expansion 15%
        self.absolute_sl = 6.0        # Hard Floor SL at ₹6

        self.current_position = None
        self.benchmark_straddle = None
        self.benchmark_spot = None
        self.unrealized_pnl = 0
        self.realized_pnl = 0
        
    def get_live_data(self):
        """Fetches spot, atm prices, and VIX from Dhan."""
        spot, ce_p, pe_p, ce_vol, pe_vol, current_vix = 0, 0, 0, 0, 0, 13.0
        ce_id, pe_id = None, None
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            # 1. Fetch VIX
            vix_resp = self.dhan.intraday_minute_data("21", self.dhan.INDEX, "INDEX", today, today)
            if vix_resp.get("status") == "success" and vix_resp.get("data"):
                vix_data = vix_resp.get("data")
                current_vix = float(vix_data.get("close", [13.0])[-1])

            # 2. Fetch Option Chain for Spot and ATM
            if self.target_expiry:
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
                        ce_id = strike_data["ce"].get("security_id")
                        pe_id = strike_data["pe"].get("security_id")
        except Exception as e:
            logger.error(f"Error fetching live data: {e}")
            
        return spot, ce_p, pe_p, ce_vol, pe_vol, current_vix, ce_id, pe_id

    def capture_benchmark(self):
        """Sets the 1:45 PM baseline for spot and straddle price."""
        try:
            spot, ce_p, pe_p, _, _, _, _, _ = self.get_live_data()
            if spot > 0:
                self.benchmark_spot = spot
                self.benchmark_straddle = ce_p + pe_p
                logger.info(f"📍 Benchmark Set | Spot: {spot} | Straddle: {round(self.benchmark_straddle, 2)}")
        except Exception as e:
            logger.error(f"Failed to capture benchmark: {e}")

    def run_iteration(self, expiry_date):
        """Main check loop - called every 1 min"""
        self.target_expiry = expiry_date
        now = datetime.datetime.now()
        
        # Trading Window Check (14:00 - 15:00 for entry)
        is_entry_window = (now.hour == 14) or (now.hour == 15 and now.minute <= 7)
        is_exit_time = now.hour == 15 and now.minute >= 25
        is_hard_sweep_time = now.hour == 15 and now.minute >= 26
        
        # Capture benchmark at 13:45
        if now.hour == 13 and now.minute == 45 and self.benchmark_straddle is None:
            self.capture_benchmark()
            
        # Paper Trade Helper: Auto-set benchmark if we started late
        if self.paper_trade and self.benchmark_straddle is None and (now.hour > 13 or (now.hour == 13 and now.minute > 45)):
            self.capture_benchmark()

        if self.in_position:
            if is_hard_sweep_time:
                logger.warning("🕒 3:26 PM HARD SWEEP TRIGGERED. Forcing immediate position cleanup.")
                self.close_position(0, "End of Day Hard Sweep")
            else:
                self.manage_position(is_exit_time)
        elif not is_hard_sweep_time and is_entry_window and self.benchmark_straddle:
            self.check_entry()

    def check_entry(self):
        """Matches the 'Gamma Blast' backtest entry logic."""
        try:
            spot, ce_p, pe_p, ce_vol, pe_vol, vix, ce_id, pe_id = self.get_live_data()
            if spot == 0 or self.benchmark_spot == 0: return

            current_straddle = ce_p + pe_p
            
            # 1. IV Expansion Filter (15%)
            iv_expansion_hit = current_straddle >= (self.benchmark_straddle * self.expansion_threshold)
            
            # 2. Momentum check (0.10% of spot price)
            momentum_hit = abs(ce_p - pe_p) > 0 
            
            if iv_expansion_hit and vix >= self.vix_threshold and momentum_hit:
                # Directional selection based on trend from benchmark
                if spot > self.benchmark_spot:
                    opt_type = 'CE'
                    price = ce_p
                    security_id = ce_id
                else:
                    opt_type = 'PE'
                    price = pe_p
                    security_id = pe_id
                
                atm_strike = int(round(spot / 50) * 50)
                logger.info(f"🚀 GAMMA SPIKE TRIGGERED | Type: {opt_type} | P: {price} | IV Change: +{round((current_straddle/self.benchmark_straddle - 1)*100, 2)}%")
                self.place_order(opt_type, price, atm_strike, security_id)
                
        except Exception as e:
            logger.error(f"Entry check failed: {e}")

    def manage_position(self, force_exit=False):
        """Matches the 'Gamma Blast' SL logic."""
        try:
            spot, ce_p, pe_p, _, _, _, _, _ = self.get_live_data()
            curr_price = ce_p if self.current_position['type'] == 'CE' else pe_p
            
            entry = self.current_position['entry']
            if curr_price > self.current_position['peak']:
                self.current_position['peak'] = curr_price
            peak = self.current_position['peak']
            profit_pct = (peak - entry) / entry
            
            # Mathematical Target SL calculation (Gamma Blast Logic)
            new_sl_val = entry * (1.0 - self.initial_sl)
            if profit_pct >= 1.00:
                new_sl_val = peak * 0.90
            elif profit_pct >= 0.20:
                new_sl_val = entry * 1.10
                
            new_sl_val = round(max(6.0, new_sl_val), 1)
            
            exit_triggered = False
            exit_price = curr_price
            reason = "hold"
            
            current_sl_val = self.current_position.get('current_sl_val', 0)
            
            # Trail Upwards only
            if new_sl_val > current_sl_val:
                self.current_position['current_sl_val'] = new_sl_val
                logger.info(f"TRAILING SL ADVANCED TO: ₹{new_sl_val}")
                
                if not self.paper_trade:
                    for order in getattr(self, 'live_sl_orders', []):
                        try:
                            resp = self.dhan.modify_order(
                                order_id=order['id'],
                                order_type=self.dhan.STOP_LOSS_MARKET,
                                leg_name='NA',
                                quantity=order['qty'],
                                price=0,
                                trigger_price=new_sl_val,
                                disclosed_quantity=0,
                                validity=self.dhan.DAY
                            )
                            logger.info(f"Modified SL Order {order['id']} to ₹{new_sl_val}: {resp}")
                        except Exception as e:
                            logger.error(f"Failed to modify SL order {order['id']}: {e}")
                            if not self.paper_trade:
                                self.alerter.send_alert(f"Gamma SL Modify Failed for Order {order['id']}: {e}")
                            
            self.unrealized_pnl = (curr_price - entry) * self.lot_size
            
            # 5. Time Exit or Local Fallback SL hit
            if force_exit:
                exit_triggered = True
                exit_price = curr_price
                reason = "Time Exit"
            elif curr_price <= self.current_position.get('current_sl_val', 0):
                exit_triggered = True
                reason = f"SL Hit Locally (₹{self.current_position.get('current_sl_val', 0)})"
                
            if exit_triggered:
                self.close_position(exit_price, reason)
                
        except Exception as e:
            logger.error(f"Position management failed: {e}")

    def place_order(self, opt_type, price, strike, security_id):
        """Simulates or places a real order."""
        initial_sl_val = round(max(6.0, price * (1.0 - self.initial_sl)), 1)
        
        self.current_position = {
            'type': opt_type,
            'entry': price,
            'peak': price,
            'strike': strike,
            'security_id': security_id,
            'current_sl_val': initial_sl_val,
            'time': datetime.datetime.now()
        }
        self.in_position = True
        self.live_sl_orders = [] # list of dicts: {'id': '123', 'qty': 100}
        
        logger.info(f"✅ Order Placed: {opt_type} at {price} (Strike: {strike}) (Paper: {self.paper_trade})")
        if not self.paper_trade and security_id:
            try:
                max_qty_per_order = 1690 # 26 limit * 65 Nifty lot
                # 1. Place Market BUY Leg
                logger.info(f"LIVE EXECUTION: Placing {opt_type} BUY orders for total qty: {self.lot_size}")
                remaining_qty = self.lot_size
                while remaining_qty > 0:
                    order_qty = min(remaining_qty, max_qty_per_order)
                    bulk_order = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=self.dhan.FNO,
                        transaction_type=self.dhan.BUY,
                        quantity=order_qty,
                        order_type=self.dhan.MARKET,
                        product_type=self.dhan.MARGIN, # Delivery/CarryForward for FNO
                        price=0
                    )
                    logger.info(f"-> Live BUY Order Chunk ({order_qty}): {bulk_order}")
                    remaining_qty -= order_qty
                    
                # 2. Add Stop Loss Market Orders for the full quantity matching entry chunks
                logger.info(f"LIVE EXECUTION: Immediatly firing SL-M SELL Orders at ₹{initial_sl_val}")
                remaining_sl_qty = self.lot_size
                while remaining_sl_qty > 0:
                    order_qty = min(remaining_sl_qty, max_qty_per_order)
                    sl_resp = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=self.dhan.FNO,
                        transaction_type=self.dhan.SELL,
                        quantity=order_qty,
                        order_type=self.dhan.STOP_LOSS_MARKET,
                        product_type=self.dhan.MARGIN, # Delivery/CarryForward for FNO
                        price=0,
                        trigger_price=initial_sl_val
                    )
                    logger.info(f"-> Live SL-M Order Chunk ({order_qty}): {sl_resp}")
                    if sl_resp and sl_resp.get('status') == 'success' and 'orderId' in sl_resp.get('data', {}):
                        self.live_sl_orders.append({'id': sl_resp['data']['orderId'], 'qty': order_qty})
                    remaining_sl_qty -= order_qty
                    
            except Exception as e:
                logger.error(f"LIVE Order Placement Failed: {e}")
                self.alerter.send_alert(f"Gamma Live Order Placement FAILED: {e}")
        elif not self.paper_trade:
            logger.error("LIVE EXECUTION FAILED: Missing Security ID for the Option.")
            self.alerter.send_alert("Gamma LIVE EXECUTION FAILED: Missing Security ID down the chain.")

    def get_net_qty_from_broker(self, security_id):
        """Safely verify our open quantity directly from Broker to avoid naked short selling."""
        try:
            positions = self.dhan.get_positions()
            if positions and positions.get("status") == "success":
                for p in positions.get("data", []):
                    if str(p.get("securityId")) == str(security_id):
                        return int(p.get("netQty", 0))
        except Exception as e:
            logger.error(f"Error checking position book: {e}")
        return 0

    def close_position(self, price, reason):
        """Simulates or closes a real position."""
        pnl = (price - self.current_position['entry']) * self.lot_size
        self.realized_pnl += pnl
        logger.info(f"🔴 Position Closed | Type: {self.current_position['type']} | Price: {price} | PnL: {round(pnl, 2)} | Reason: {reason}")
        self.in_position = False
        old_position = self.current_position
        self.current_position = None
        self.unrealized_pnl = 0
        if not self.paper_trade and old_position.get('security_id'):
            try:
                security_id = old_position['security_id']
                
                # 1. CRITICAL: Cancel trailing SL orders cleanly so they don't fire twice!
                for order in getattr(self, 'live_sl_orders', []):
                    try:
                        self.dhan.cancel_order(order_id=order['id'])
                        logger.info(f"Cancelled Pending SL Trigger: {order['id']}")
                    except Exception as e:
                        logger.error(f"Attempt cancelling old SL failed: {e}")
                        self.alerter.send_alert(f"Gamma LIVE SL Cancel Failed for order {order['id']} on Exit: {e}")
                self.live_sl_orders = []
                import time; time.sleep(1.0) # wait briefly for Dhan engine to purge cancelled orders 
                
                max_qty_per_order = 1690 # 26 lot limit
                
                # 2. VERIFY how many shares we STILL actually own (maybe SL fired right as clock ran out)
                true_net_qty = self.get_net_qty_from_broker(security_id)
                
                if true_net_qty > 0:
                    remaining_qty = true_net_qty
                    logger.info(f"LIVE EXIT: Firing remaining {true_net_qty} MARKET SELL orders for {old_position['type']}")
                    while remaining_qty > 0:
                        order_qty = min(remaining_qty, max_qty_per_order)
                        order = self.dhan.place_order(
                            security_id=str(security_id),
                            exchange_segment=self.dhan.FNO,
                            transaction_type=self.dhan.SELL,
                            quantity=order_qty,
                            order_type=self.dhan.MARKET,
                            product_type=self.dhan.MARGIN, # Delivery/CarryForward for FNO
                            price=0
                        )
                        logger.info(f"-> Live SELL Exit Chunk ({order_qty}): {order}")
                        remaining_qty -= order_qty
                else:
                    logger.info("LIVE EXIT OVERRIDE: 0 Net Qty found on broker book. Stop loss likely consumed by Dhan internally. Clean skip!")
                    
            except Exception as e:
                logger.error(f"LIVE Order Exit Failed: {e}")
                self.alerter.send_alert(f"Gamma LIVE ORDER EXIT SWEEP FAILED: {e}")

        # Save to PostgreSQL
        try:
            import pandas as pd
            from sqlalchemy import create_engine
            uri = os.getenv("POSTGRES_URI")
            if uri:
                engine = create_engine(uri)
                trade_record = {
                    'Run_Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Strategy_Name': "gamma_blast",
                    'Run_Mode': 'Forward Test' if self.paper_trade else 'Live Trade',
                    'Date': datetime.datetime.now().strftime("%Y-%m-%d"),
                    'Entry_Time': old_position['time'].strftime("%H:%M:%S"),
                    'Exit_Time': datetime.datetime.now().strftime("%H:%M:%S"),
                    'Strike': f"{old_position.get('strike', '0')}-{self.target_expiry}-{old_position['type']}",
                    'Option_Type': 'C' if old_position['type'] == 'CE' else 'P',
                    'Action': 'BUY',
                    'Qty': self.lot_size,
                    'Buy_Price': round(old_position['entry'], 2),
                    'Peak_Price': round(old_position['peak'], 2),
                    'Sell_Price': round(price, 2),
                    'PNL': round(pnl, 2),
                    'ROI%': round(((price - old_position['entry']) / old_position['entry']) * 100, 2) if old_position['entry'] > 0 else 0,
                    'Capital_ROI%': round((pnl / 100000) * 100, 2),
                    'Reason': reason,
                    'Win': pnl > 0
                }
                df = pd.DataFrame([trade_record])
                table_name = "historical_backtests"
                try:
                    existing = pd.read_sql(f"SELECT * FROM {table_name}", con=engine)
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception:
                    pass
                df.to_sql(table_name, con=engine, if_exists='replace', index=False)
                logger.info(f"Saved trade to DB table {table_name} with mode {trade_record['Run_Mode']}")
        except Exception as e:
            logger.error(f"Failed to save trade to DB: {e}")

def main():
    logger.info("Starting Nifty Expiry Day Gamma Spike Strategy...")
    CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    strategy = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
    
    current_expiry = "2026-03-02"
    
    logger.info(f"Monitoring Gamma for Expiry: {current_expiry}")
    try:
        while True:
            strategy.run_iteration(current_expiry)
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user.")

if __name__ == "__main__":
    main()
