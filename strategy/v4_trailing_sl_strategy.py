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

class NiftyV4TrailingSLStrategy:
    def __init__(self, target_expiry):
        self.dhan = dhanhq(str(CLIENT_ID), str(ACCESS_TOKEN))
        self.target_expiry = target_expiry
        self.lot_size = 65
        self.running = False
        self.paused = False
        self.in_position = False
        self.paper_trade = True 
        
        # Override fields
        self.index_id = "13"          # Default Nifty 50
        self.manual_base_time = None  # e.g. "13:45"
        self.force_run = False
        
        # --- Backtest-Matched V4 Parameters ---
        self.initial_sl          = 0.35        # Initial SL: 35% loss from entry
        self.trailing_step       = 0.15        # Trail drops 15% from peak once profitable
        self.target_lock_in      = 0.20        # Start trailing after 20% gain
        self.vix_threshold       = 12.5        # VIX must be >= 12.5
        self.expansion_threshold = 1.17        # Individual leg must expand 17% from 1:30 PM
        self.spot_move_pct       = 0.003       # Spot must have moved >= 0.3% from benchmark
        self.min_premium         = 5.0         # Min option premium to enter (avoids OTM junk)
        self.absolute_sl         = 6.0         # Hard Floor SL at ₹6
        self.entry_cutoff        = (15, 1)     # No new entries after 3:01 PM (hour, minute)

        self.current_position    = None
        self.benchmark_straddle  = None
        self.benchmark_spot      = None
        self.benchmark_ce        = None   # individual CE price at 1:30 PM
        self.benchmark_pe        = None   # individual PE price at 1:30 PM
        self.unrealized_pnl      = 0
        self.realized_pnl        = 0
        
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
            oc_resp = self.dhan.option_chain(int(self.index_id), self.dhan.INDEX, self.target_expiry)
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
                self.benchmark_spot     = spot
                self.benchmark_straddle = ce_p + pe_p
                self.benchmark_ce       = ce_p    # save individual legs
                self.benchmark_pe       = pe_p
                logger.info(f"📍 Benchmark Set (V4) | Spot: {spot} | CE: {ce_p} | PE: {pe_p} | Straddle: {round(self.benchmark_straddle, 2)}")
            else:
                logger.warning(f"Benchmark Set (V4) Failed: Could not fetch spot > 0 for expiry {self.target_expiry}. (Ensure valid expiry date and market hours)")
        except Exception as e:
            logger.error(f"Failed to capture benchmark: {e}")

    def run_iteration(self):
        """Main check loop - called every 1 min"""
        now = datetime.datetime.now()
        
        # Trading Window Check (14:00 - 15:00 for entry)
        is_entry_window = (now.hour == 14) or (now.hour == 15 and now.minute <= 7)
        is_exit_time = now.hour == 15 and now.minute >= 25
        is_hard_sweep_time = now.hour == 15 and now.minute >= 26
        
        # Expiry Day Check: only trade if today is expiry (or forced)
        today_str = now.strftime("%Y-%m-%d")
        if not self.force_run and today_str != self.target_expiry:
            # We still allow position management if we're in one (e.g. overnight carry, though V4 is intraday)
            if not self.in_position:
                return
        
        # Capture benchmark logic — at 1:30 PM exactly (matches backtest)
        base_h, base_m = 13, 30
        if self.manual_base_time:
            try:
                base_h, base_m = map(int, self.manual_base_time.split(':'))
            except: pass
            
        if now.hour == base_h and now.minute == base_m and self.benchmark_straddle is None:
            self.capture_benchmark()
            
        # Auto-set benchmark if we started late
        if self.benchmark_straddle is None and (now.hour > base_h or (now.hour == base_h and now.minute > base_m)):
            self.capture_benchmark()

        # Entry window: after 1:30 PM benchmark time but NOT after 3:01 PM
        ch, cm = self.entry_cutoff
        past_benchmark = (now.hour > base_h or (now.hour == base_h and now.minute > base_m))
        before_cutoff  = (now.hour < ch or (now.hour == ch and now.minute <= cm))
        is_entry_window = past_benchmark and before_cutoff

        if self.in_position:
            if is_hard_sweep_time:
                logger.warning("🕒 3:26 PM HARD SWEEP TRIGGERED. Forcing immediate position cleanup.")
                self.close_position(0, "End of Day Hard Sweep")
            else:
                self.manage_position(is_exit_time)
        elif not is_hard_sweep_time and is_entry_window and self.benchmark_straddle:
            self.check_entry()

    def check_entry(self):
        """Entry: individual CE or PE leg must expand >=17% from its 1:30 PM price."""
        try:
            spot, ce_p, pe_p, ce_vol, pe_vol, vix, ce_id, pe_id = self.get_live_data()
            if spot == 0 or self.benchmark_spot == 0:
                logger.warning("Entry Check skipped: Spot is 0.")
                return
            if not self.benchmark_ce or not self.benchmark_pe:
                logger.warning("Entry Check skipped: Benchmark legs not set.")
                return

            # Individual leg expansion check (17%)
            ce_exp = ce_p / self.benchmark_ce
            pe_exp = pe_p / self.benchmark_pe
            leg_spike = max(ce_exp, pe_exp) >= self.expansion_threshold
            spike_leg = 'CE' if ce_exp >= pe_exp else 'PE'
            spike_pct = round((max(ce_exp, pe_exp) - 1) * 100, 2)

            spot_move = abs(spot - self.benchmark_spot) / self.benchmark_spot
            spot_move_hit = spot_move >= self.spot_move_pct

            logger.info(
                f"Entry Check (V4) | Spot: {spot} | VIX: {vix} | "
                f"CE: {ce_p:.2f} ({ce_exp*100-100:+.1f}%) | PE: {pe_p:.2f} ({pe_exp*100-100:+.1f}%) | "
                f"SpotMove: {round(spot_move*100,3)}% | LegSpike: {leg_spike}"
            )

            if leg_spike and vix >= self.vix_threshold and spot_move_hit:
                # Direction: follow the spot from benchmark
                if spot > self.benchmark_spot:
                    opt_type = 'CE'
                    price = ce_p
                    security_id = ce_id
                else:
                    opt_type = 'PE'
                    price = pe_p
                    security_id = pe_id

                # Min premium filter
                if price < self.min_premium:
                    logger.info(f"Entry skipped: premium ₹{price} < min ₹{self.min_premium}")
                    return

                atm_strike = int(round(spot / 50) * 50)
                logger.info(f"🚀 V4 ENTRY | {opt_type} @ ₹{price} | Strike: {atm_strike} | {spike_leg} spike +{spike_pct}%")
                self.place_order(opt_type, price, atm_strike, security_id)

        except Exception as e:
            logger.error(f"Entry check failed: {e}")

    def manage_position(self, force_exit=False):
        """Trailing SL logic matching backtest exactly."""
        try:
            spot, ce_p, pe_p, _, _, _, _, _ = self.get_live_data()
            curr_price = ce_p if self.current_position['type'] == 'CE' else pe_p

            entry = self.current_position['entry']
            if curr_price > self.current_position['peak']:
                self.current_position['peak'] = curr_price
            peak = self.current_position['peak']

            # --- Backtest-matched SL logic ---
            if peak >= entry * (1 + self.target_lock_in):
                # Trailing SL: 15% drop from peak
                new_sl_val = peak * (1 - self.trailing_step)
                sl_type = "Trailing SL"
            else:
                # Initial SL: 35% loss from entry
                new_sl_val = entry * (1 - self.initial_sl)
                sl_type = "Initial SL"

            new_sl_val = round(max(self.absolute_sl, new_sl_val), 1)
            
            exit_triggered = False
            exit_price = curr_price
            reason = "hold"
            
            current_sl_val = self.current_position.get('current_sl_val', 0)

            # Trail upwards only
            if new_sl_val > current_sl_val:
                self.current_position['current_sl_val'] = new_sl_val
                logger.info(f"📈 {sl_type} advanced → ₹{new_sl_val} (Peak: ₹{peak:.2f}, Entry: ₹{entry:.2f})")

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
                            logger.info(f"Modified SL Order {order['id']} → ₹{new_sl_val}: {resp}")
                        except Exception as e:
                            logger.error(f"Failed to modify SL order {order['id']}: {e}")

            self.unrealized_pnl = (curr_price - entry) * self.lot_size

            exit_triggered = False
            exit_price = curr_price
            reason = sl_type

            if force_exit:
                exit_triggered = True
                exit_price = curr_price
                reason = "Time Exit"
            elif curr_price <= self.current_position.get('current_sl_val', 0):
                exit_triggered = True

            if exit_triggered:
                self.close_position(exit_price, reason)
                
        except Exception as e:
            logger.error(f"Position management failed: {e}")

    def place_order(self, opt_type, price, strike, security_id):
        """Simulates or places a real order."""
        initial_sl_val = round(max(self.absolute_sl, price * (1.0 - self.initial_sl)), 1)
        
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
        elif not self.paper_trade:
            logger.error("LIVE EXECUTION FAILED: Missing Security ID for the Option.")

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

        # Save to PostgreSQL
        try:
            import pandas as pd
            from sqlalchemy import create_engine
            uri = os.getenv("POSTGRES_URI")
            if uri:
                engine = create_engine(uri)
                params_str = (
                    f"lock_in={self.target_lock_in*100:.0f}%|"
                    f"trail={self.trailing_step*100:.0f}%|"
                    f"init_sl={self.initial_sl*100:.0f}%|"
                    f"leg_expansion={self.expansion_threshold*100:.0f}%|"
                    f"min_prem={self.min_premium}|"
                    f"cutoff={self.entry_cutoff[0]:02d}:{self.entry_cutoff[1]:02d}|"
                    f"vix_thresh={self.vix_threshold}"
                )
                trade_record = {
                    'Run_Date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Strategy_Name': "v4_gamma",
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
                    'Win': pnl > 0,
                    'Parameters': params_str
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

if __name__ == "__main__":
    # Test execution
    strategy = NiftyV4TrailingSLStrategy("2026-03-02")
    # strategy.run_strategy()
