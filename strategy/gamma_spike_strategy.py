import os
import time
import datetime
import logging
from dotenv import load_dotenv
from dhanhq import dhanhq

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

class NiftyGammaSpikeStrategy:
    """
    Gamma Spike SELL Strategy — Morning Session
    ─────────────────────────────────────────────
    1. Capture ATM CE & PE prices at 9:20 AM (benchmark).
    2. From 9:30 AM, watch for any leg spiking >= 20% above benchmark.
    3. SELL that spiked leg (it's overpriced — morning fear premium).
    4. Buy back when:
       • Premium drops 35% from sell price → TARGET HIT (profit)
       • Premium rises 40% from sell price → SL HIT (loss)
       • 11:30 AM reached → TIME EXIT (whatever price)
    Matches backtest_gamma.py logic exactly.
    """

    def __init__(self, client_id, access_token):
        from dhanhq.dhan_context import DhanContext
        context = DhanContext(str(client_id), str(access_token))
        self.dhan = dhanhq(context)

        self.target_expiry   = None
        self.lot_size        = 1300          # 20 lots × 65 qty
        self.running         = False
        self.paused          = False
        self.in_position     = False
        self.paper_trade     = True          # overridden by app.py engine starter

        # ── Strategy Parameters (must match backtest_gamma.py) ───────────
        self.leg_expansion   = 1.20          # CE or PE must spike 20% from 9:20 AM
        self.min_sell_price  = 20.0          # don't sell if premium < ₹20
        self.target_pct      = 0.35          # buy back when drops 35% → profit
        self.sl_pct          = 0.40          # buy back when rises 40% → loss
        self.benchmark_hour  = 9
        self.benchmark_min   = 20
        self.entry_start     = (9,  30)      # start scanning from 9:30 AM
        self.entry_cutoff    = (11, 30)      # no new sells after 11:30 AM
        self.hard_exit       = (11, 30)      # force close at 11:30 AM
        # ────────────────────────────────────────────────────────────────

        self.force_run       = False          # set True by app.py when override panel is used
        self.benchmark_ce    = None          # CE price at 9:20 AM
        self.benchmark_pe    = None          # PE price at 9:20 AM
        self.benchmark_spot  = None
        self.current_position = None
        self.unrealized_pnl  = 0
        self.realized_pnl    = 0

    # ─────────────────────────────────────────────────────────────────────
    # Data Fetching
    # ─────────────────────────────────────────────────────────────────────

    def get_live_data(self):
        """Fetch spot, ATM CE/PE prices from Dhan option chain."""
        spot, ce_p, pe_p, ce_vol, pe_vol, vix = 0, 0, 0, 0, 0, 13.0
        ce_id, pe_id = None, None
        today = datetime.datetime.now().strftime("%Y-%m-%d")

        try:
            # VIX
            vix_resp = self.dhan.intraday_minute_data("21", self.dhan.INDEX, "INDEX", today, today)
            if vix_resp.get("status") == "success" and vix_resp.get("data"):
                vix = float(vix_resp["data"].get("close", [13.0])[-1])

            # Option chain
            if self.target_expiry:
                oc_resp = self.dhan.option_chain(13, self.dhan.INDEX, self.target_expiry)
                if oc_resp.get("status") == "success":
                    data = oc_resp["data"]
                    spot = data.get("last_price", 0)
                    if spot > 0 and "oc" in data:
                        strikes = [float(s) for s in data["oc"].keys()]
                        atm_strike = min(strikes, key=lambda x: abs(x - spot))
                        sd = data["oc"][f"{atm_strike:.6f}"]
                        ce_p   = sd["ce"].get("last_price", 0)
                        pe_p   = sd["pe"].get("last_price", 0)
                        ce_vol = sd["ce"].get("volume", 1)
                        pe_vol = sd["pe"].get("volume", 1)
                        ce_id  = sd["ce"].get("security_id")
                        pe_id  = sd["pe"].get("security_id")
        except Exception as e:
            logger.error(f"get_live_data error: {e}")

        return spot, ce_p, pe_p, ce_vol, pe_vol, vix, ce_id, pe_id

    # ─────────────────────────────────────────────────────────────────────
    # Benchmark
    # ─────────────────────────────────────────────────────────────────────

    def capture_benchmark(self):
        """Capture individual CE & PE prices at 9:20 AM as baseline."""
        try:
            spot, ce_p, pe_p, _, _, _, _, _ = self.get_live_data()
            if spot > 0 and ce_p > 0 and pe_p > 0:
                self.benchmark_spot = spot
                self.benchmark_ce   = ce_p
                self.benchmark_pe   = pe_p
                logger.info(
                    f"📍 Benchmark Set (9:20 AM) | Spot: {spot} "
                    f"| CE: {ce_p:.2f} | PE: {pe_p:.2f}"
                )
            else:
                logger.warning(f"Benchmark capture failed: spot={spot} ce={ce_p} pe={pe_p}")
        except Exception as e:
            logger.error(f"capture_benchmark error: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Main Loop
    # ─────────────────────────────────────────────────────────────────────

    def run_iteration(self, expiry_date):
        """Called every ~60 seconds by app.py engine."""
        self.target_expiry = expiry_date
        now = datetime.datetime.now()

        # ── 1. Capture benchmark exactly at 9:20 AM ───────────────────
        if (now.hour == self.benchmark_hour and now.minute == self.benchmark_min
                and self.benchmark_ce is None):
            self.capture_benchmark()

        # Auto-capture if we started late
        is_past_benchmark = (now.hour > self.benchmark_hour or
                             (now.hour == self.benchmark_hour and now.minute >= self.benchmark_min))
        if is_past_benchmark and self.benchmark_ce is None:
            self.capture_benchmark()

        if not self.benchmark_ce:
            logger.info("⏳ Waiting for 9:20 AM benchmark...")
            return

        # ── 2. Time gates ──────────────────────────────────────────────
        sh, sm = self.entry_start
        ch, cm = self.entry_cutoff
        hh, hm = self.hard_exit

        in_entry_window = (
            (now.hour > sh or (now.hour == sh and now.minute >= sm)) and
            (now.hour < ch or (now.hour == ch and now.minute <= cm))
        )
        is_hard_exit = now.hour > hh or (now.hour == hh and now.minute >= hm)

        # ── 3. Manage open position ────────────────────────────────────
        if self.in_position:
            if is_hard_exit:
                logger.warning("🕒 11:30 AM — Hard exit triggered.")
                self._close_with_market("Time Exit (11:30)")
            else:
                self.manage_position(force_exit=False)
        elif in_entry_window:
            self.check_entry()
        else:
            if not is_hard_exit:
                logger.info(f"⏳ Waiting for entry window (9:30–11:30 AM)... [{now.strftime('%H:%M')}]")

    # ─────────────────────────────────────────────────────────────────────
    # Entry: SELL the spiked leg
    # ─────────────────────────────────────────────────────────────────────

    def check_entry(self):
        """Sell whichever leg has spiked >= 20% from 9:20 AM benchmark."""
        try:
            spot, ce_p, pe_p, _, _, vix, ce_id, pe_id = self.get_live_data()
            if spot == 0:
                logger.warning("Entry skipped: spot = 0")
                return

            ce_exp = ce_p / self.benchmark_ce if self.benchmark_ce else 0
            pe_exp = pe_p / self.benchmark_pe if self.benchmark_pe else 0

            logger.info(
                f"Entry Check | CE: {ce_p:.2f} (+{(ce_exp-1)*100:.1f}%) "
                f"PE: {pe_p:.2f} (+{(pe_exp-1)*100:.1f}%) | VIX: {vix:.1f}"
            )

            if max(ce_exp, pe_exp) < self.leg_expansion:
                return  # no spike yet

            # Sell the more spiked leg
            if ce_exp >= pe_exp:
                opt_type    = 'CE'
                sell_price  = ce_p
                security_id = ce_id
                spike_pct   = (ce_exp - 1) * 100
            else:
                opt_type    = 'PE'
                sell_price  = pe_p
                security_id = pe_id
                spike_pct   = (pe_exp - 1) * 100

            if sell_price < self.min_sell_price:
                logger.info(f"Entry skipped: sell price ₹{sell_price:.2f} < min ₹{self.min_sell_price}")
                return

            target_price = round(sell_price * (1 - self.target_pct), 2)
            sl_price     = round(sell_price * (1 + self.sl_pct),     2)

            logger.info(
                f"🔴 SELL {opt_type} @ ₹{sell_price:.2f} "
                f"| Spike: +{spike_pct:.1f}% | "
                f"Target ≤ ₹{target_price} | SL ≥ ₹{sl_price}"
            )
            self.place_sell_order(opt_type, sell_price, target_price, sl_price, security_id, spot)

        except Exception as e:
            logger.error(f"check_entry error: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Position Management: buy back to close
    # ─────────────────────────────────────────────────────────────────────

    def manage_position(self, force_exit=False):
        """Check if target or SL hit; buy back to close the short."""
        try:
            spot, ce_p, pe_p, _, _, _, _, _ = self.get_live_data()
            curr_price = ce_p if self.current_position['type'] == 'CE' else pe_p

            entry_sell  = self.current_position['entry']   # price we sold at
            target      = self.current_position['target']  # buy-back target
            sl          = self.current_position['sl']      # buy-back SL

            pnl_pct = (entry_sell - curr_price) / entry_sell * 100
            self.unrealized_pnl = (entry_sell - curr_price) * self.lot_size

            logger.info(
                f"Position Monitor | {self.current_position['type']} "
                f"Sold @ ₹{entry_sell:.2f} | Now: ₹{curr_price:.2f} "
                f"| P&L: {pnl_pct:.1f}% | Target ≤ ₹{target} | SL ≥ ₹{sl}"
            )

            if force_exit:
                self.close_position(curr_price, "Time Exit")
            elif curr_price <= target:
                logger.info(f"✅ TARGET HIT @ ₹{curr_price:.2f}")
                self.close_position(curr_price, "Target Hit")
            elif curr_price >= sl:
                logger.warning(f"❌ SL HIT @ ₹{curr_price:.2f}")
                self.close_position(curr_price, "SL Hit")

        except Exception as e:
            logger.error(f"manage_position error: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # Order Execution
    # ─────────────────────────────────────────────────────────────────────

    def place_sell_order(self, opt_type, sell_price, target_price, sl_price, security_id, spot):
        """Record or execute a SELL (short) order."""
        atm_strike = int(round(spot / 50) * 50)

        self.current_position = {
            'type':        opt_type,
            'entry':       sell_price,   # SELL price
            'target':      target_price,
            'sl':          sl_price,
            'strike':      atm_strike,
            'security_id': security_id,
            'time':        datetime.datetime.now()
        }
        self.in_position    = True
        self.live_sl_orders = []   # track SL buy orders for live mode

        logger.info(f"✅ SELL Order Placed | {opt_type} @ ₹{sell_price:.2f} | Paper: {self.paper_trade}")

        if not self.paper_trade and security_id:
            try:
                max_chunk = 1690
                remaining = self.lot_size
                # 1. SELL (short) the option
                while remaining > 0:
                    qty = min(remaining, max_chunk)
                    resp = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=self.dhan.FNO,
                        transaction_type=self.dhan.SELL,
                        quantity=qty,
                        order_type=self.dhan.MARKET,
                        product_type=self.dhan.MARGIN,
                        price=0
                    )
                    logger.info(f"-> Live SELL chunk ({qty}): {resp}")
                    remaining -= qty

                # 2. Place a BUY SL-M to protect (buys back if price rises to SL)
                remaining_sl = self.lot_size
                while remaining_sl > 0:
                    qty = min(remaining_sl, max_chunk)
                    sl_resp = self.dhan.place_order(
                        security_id=str(security_id),
                        exchange_segment=self.dhan.FNO,
                        transaction_type=self.dhan.BUY,
                        quantity=qty,
                        order_type=self.dhan.STOP_LOSS_MARKET,
                        product_type=self.dhan.MARGIN,
                        price=0,
                        trigger_price=sl_price
                    )
                    logger.info(f"-> Live BUY SL chunk ({qty}): {sl_resp}")
                    if sl_resp and sl_resp.get('status') == 'success':
                        oid = sl_resp.get('data', {}).get('orderId')
                        if oid:
                            self.live_sl_orders.append({'id': oid, 'qty': qty})
                    remaining_sl -= qty

            except Exception as e:
                logger.error(f"Live SELL order failed: {e}")
        elif not self.paper_trade:
            logger.error("Live SELL failed: no security_id")

    def _close_with_market(self, reason):
        """Force market close on time exit."""
        try:
            _, ce_p, pe_p, _, _, _, _, _ = self.get_live_data()
            curr = ce_p if self.current_position['type'] == 'CE' else pe_p
            self.close_position(curr, reason)
        except Exception as e:
            logger.error(f"Force close failed: {e}")
            self.close_position(0, reason)

    def get_net_qty_from_broker(self, security_id):
        """Check actual short qty on broker to avoid mismatches."""
        try:
            positions = self.dhan.get_positions()
            if positions and positions.get("status") == "success":
                for p in positions.get("data", []):
                    if str(p.get("securityId")) == str(security_id):
                        return abs(int(p.get("netQty", 0)))
        except Exception as e:
            logger.error(f"get_net_qty error: {e}")
        return 0

    def close_position(self, buy_back_price, reason):
        """Buy back to close the short position. PnL = sell_price - buy_back_price."""
        entry_sell = self.current_position['entry']
        pnl        = (entry_sell - buy_back_price) * self.lot_size
        self.realized_pnl += pnl
        self.unrealized_pnl = 0

        logger.info(
            f"🟢 BUY BACK | {self.current_position['type']} "
            f"Sold @ ₹{entry_sell:.2f} → BuyBack @ ₹{buy_back_price:.2f} "
            f"| PnL: ₹{round(pnl, 2):,} | {reason}"
        )

        old_pos = self.current_position
        self.in_position      = False
        self.current_position = None

        # ── Live execution: cancel SL, buy back ─────────────────────────
        if not self.paper_trade and old_pos.get('security_id'):
            try:
                security_id = old_pos['security_id']

                # Cancel pending SL BUY orders
                for order in getattr(self, 'live_sl_orders', []):
                    try:
                        self.dhan.cancel_order(order_id=order['id'])
                        logger.info(f"Cancelled SL BUY order: {order['id']}")
                    except Exception as e:
                        logger.error(f"Cancel SL error: {e}")
                self.live_sl_orders = []
                time.sleep(1.0)

                # Verify short qty still open
                true_qty = self.get_net_qty_from_broker(security_id)
                if true_qty > 0:
                    remaining = true_qty
                    max_chunk = 1690
                    while remaining > 0:
                        qty = min(remaining, max_chunk)
                        resp = self.dhan.place_order(
                            security_id=str(security_id),
                            exchange_segment=self.dhan.FNO,
                            transaction_type=self.dhan.BUY,
                            quantity=qty,
                            order_type=self.dhan.MARKET,
                            product_type=self.dhan.MARGIN,
                            price=0
                        )
                        logger.info(f"-> Live BUY BACK chunk ({qty}): {resp}")
                        remaining -= qty
                else:
                    logger.info("BUY BACK skip: 0 net qty (SL already triggered on broker)")
            except Exception as e:
                logger.error(f"Live BUY BACK failed: {e}")

        # ── Save to DB ───────────────────────────────────────────────────
        try:
            import pandas as pd
            from sqlalchemy import create_engine
            uri = os.getenv("POSTGRES_URI")
            if uri:
                engine = create_engine(uri)
                record = {
                    'Run_Date':     datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'Strategy_Name': "v4_gamma_sell",
                    'Run_Mode':     'Forward Test' if self.paper_trade else 'Live Trade',
                    'Date':         datetime.datetime.now().strftime("%Y-%m-%d"),
                    'Entry_Time':   old_pos['time'].strftime("%H:%M:%S"),
                    'Exit_Time':    datetime.datetime.now().strftime("%H:%M:%S"),
                    'Option_Type':  'C' if old_pos['type'] == 'CE' else 'P',
                    'Strike':       f"{old_pos.get('strike','?')}-{self.target_expiry}-{old_pos['type']}",
                    'Action':       'SELL',
                    'Qty':          self.lot_size,
                    'Buy_Price':    round(buy_back_price, 2),
                    'Peak_Price':   round(entry_sell, 2),
                    'Sell_Price':   round(entry_sell, 2),
                    'PNL':          round(pnl, 2),
                    'ROI%':         round((entry_sell - buy_back_price) / entry_sell * 100, 2) if entry_sell > 0 else 0,
                    'Capital_ROI%': round((pnl / 500000) * 100, 2),
                    'Reason':       reason,
                    'Win':          pnl > 0,
                    'Parameters':   f"SELL|leg_spike=20%|target=35%|sl=40%|bench=09:20|entry=09:30-11:30"
                }
                df = pd.DataFrame([record])
                try:
                    existing = pd.read_sql("SELECT * FROM historical_backtests", con=engine)
                    df = pd.concat([existing, df], ignore_index=True)
                except Exception:
                    pass
                df.to_sql("historical_backtests", con=engine, if_exists='replace', index=False)
                logger.info(f"✅ Trade saved to DB as v4_gamma_sell ({record['Run_Mode']})")
        except Exception as e:
            logger.error(f"DB save failed: {e}")


def main():
    """Standalone runner for testing."""
    logger.info("Starting Gamma Spike SELL Strategy...")
    from dotenv import load_dotenv
    load_dotenv()
    CLIENT_ID    = os.getenv("DHAN_CLIENT_ID")
    ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
    strategy     = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)

    # Set today's expiry — app.py passes this dynamically
    strategy.target_expiry = datetime.datetime.now().strftime("%Y-%m-%d")
    logger.info(f"Target expiry set to: {strategy.target_expiry}")

    try:
        while True:
            strategy.run_iteration(strategy.target_expiry)
            time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Strategy stopped by user.")

if __name__ == "__main__":
    main()
