"""
NiftyZScoreStrategy — Live trading engine for Z-score mean-reversion on NIFTY spot.

Parameters mirror ZScoreParams from backtest/zscore_nifty_backtester.py so that
backtested settings can be dropped in directly.

Best-performing params (1-year afternoon backtest, profit factor 12.78):
    lookback=20, entry_z=2.0, exit_z=0.5, stop_z=2.3,
    use_adx_filter=True, adx_period=14, max_adx=25.0,
    max_loss_per_trade=5000, max_daily_loss=7500, max_trades_per_day=3,
    session_start="13:30", session_end="15:00"
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dhanhq import dhanhq

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
INTRADAY_URL = "https://api.dhan.co/v2/charts/intraday"

# NIFTY index futures security ID on Dhan (NSE_FNO, FUTIDX, nearest expiry)
# The security_id must be looked up fresh each expiry; set via params or override.
NIFTY_FUT_SECURITY_ID = "13"   # placeholder — override with live security_id


# ---------------------------------------------------------------------------
# Parameters (mirrors ZScoreParams in backtester)
# ---------------------------------------------------------------------------

@dataclass
class ZScoreStrategyParams:
    # Signal
    lookback: int = 20
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 2.3

    # Sizing
    qty: int = 1                        # number of lots (1 lot = option_lot_size units)
    option_lot_size: int = 25           # NIFTY standard options lot size

    # ADX regime filter
    use_adx_filter: bool = True
    adx_period: int = 14
    max_adx: float = 25.0               # skip entry if ADX > this (trending market)

    # Risk controls
    max_loss_per_trade: float = 5000.0  # 0 = disabled; hard rupee stop per trade
    max_daily_loss: float = 7500.0      # 0 = disabled; skip entries after this day loss
    max_trades_per_day: int = 3         # 0 = disabled; cap entries per calendar day

    # Options mode
    use_options: bool = True            # True = trade ATM CE/PE; False = trade futures
    nifty_index_id: int = 13           # Dhan index id for NIFTY (13 = NIFTY 50)

    # Session window (IST, HH:MM)
    session_start: str = "13:30"
    session_end: str = "15:00"

    # Capital tracking only (not used for risk)
    initial_capital: float = 500000.0

    # Live order routing — security_id for NIFTY futures nearest expiry
    # Look up from Dhan master CSV before each session
    nifty_security_id: str = NIFTY_FUT_SECURITY_ID


# ---------------------------------------------------------------------------
# Live Strategy
# ---------------------------------------------------------------------------

class NiftyZScoreStrategy:
    """
    Runs one iteration per tick — wire run_iteration() to apscheduler/threading.

    Paper mode (default): in-memory position tracking, no real orders placed.
    Live mode: pass paper_trade=False + client_id. Orders are placed via Dhan
               NIFTY futures (NSE, INTRA, MARKET). Set params.nifty_security_id
               to the correct NIFTY nearest-expiry futures security_id before use.
    """

    def __init__(
        self,
        access_token: str,
        client_id: str = "",
        params: ZScoreStrategyParams = None,
        paper_trade: bool = True,
    ):
        self.access_token = access_token
        self.client_id = client_id
        self.params = params or ZScoreStrategyParams()
        self.paper_trade = paper_trade

        self._headers = {
            "access-token": self.access_token,
            "Content-Type": "application/json",
        }

        # Rolling price buffer for z-score
        self._prices: deque = deque(maxlen=self.params.lookback)

        # Current open position
        self.current_position: dict | None = None

        # PnL tracking
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0

        # Daily state (reset each new calendar day)
        self._today: str = ""
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0

        self.paused: bool = False

        # Dhan SDK client (used only in live mode)
        self._dhan: dhanhq | None = None
        if not self.paper_trade:
            self._dhan = dhanhq(str(client_id), str(access_token))

        # Track live order IDs for cancellation
        self._live_order_id: str | None = None
        # Cache of ATM option chain per-day { "YYYY-MM-DD": { spot, ce_id, pe_id, ce_price, pe_price, expiry } }
        self._option_chain_cache: dict = {}

    # ------------------------------------------------------------------
    # Option chain — ATM CE/PE price + security_id
    # ------------------------------------------------------------------

    def _get_atm_option_data(self) -> dict | None:
        """
        Fetch ATM CE and PE prices + security_ids from Dhan option_chain.
        Uses expiry_manager's target_expiry if set, else uses the nearest weekly.
        Returns dict: { spot, ce_price, pe_price, ce_id, pe_id, atm_strike, expiry }
        or None on failure.
        """
        today = datetime.now(IST).strftime("%Y-%m-%d")
        # Reuse cache for the same day to avoid repeated API calls each tick
        if today in self._option_chain_cache:
            return self._option_chain_cache[today]

        if self._dhan is None:
            # Paper mode — use Dhan API directly via requests
            return self._fetch_option_data_via_api(today)

        try:
            # Determine expiry (year-month-day format Dhan expects)
            expiry = getattr(self, "_expiry", None)
            if not expiry:
                # Fallback: nearest weekly Tuesday
                from strategy.expiry_manager import ExpiryManager
                expiry = ExpiryManager(self.access_token, str(self.params.nifty_index_id)).get_current_expiry()

            idx_name = "IDX_I"
            oc_resp = self._dhan.option_chain(self.params.nifty_index_id, idx_name, expiry)

            if not oc_resp or oc_resp.get("status") != "success":
                logger.warning("option_chain API failed: %s", oc_resp)
                return None

            raw = oc_resp.get("data", {})
            data = raw.get("data", raw) if isinstance(raw, dict) and "data" in raw else raw

            spot = float(data.get("last_price", 0))
            if spot <= 0:
                return None

            strikes = [float(s) for s in data.get("oc", {}).keys()]
            atm_strike = min(strikes, key=lambda x: abs(x - spot))
            chain = data["oc"][f"{atm_strike:.6f}"]

            result = {
                "spot": spot,
                "atm_strike": atm_strike,
                "ce_price": float(chain.get("ce", {}).get("last_price", 0)),
                "pe_price": float(chain.get("pe", {}).get("last_price", 0)),
                "ce_id": str(chain.get("ce", {}).get("security_id", "")),
                "pe_id": str(chain.get("pe", {}).get("security_id", "")),
                "expiry": expiry,
            }
            self._option_chain_cache[today] = result
            logger.info(
                "ATM option chain: NIFTY=%.2f strike=%.0f CE=%.2f(id=%s) PE=%.2f(id=%s)",
                spot, atm_strike, result["ce_price"], result["ce_id"],
                result["pe_price"], result["pe_id"],
            )
            return result
        except Exception as exc:
            logger.error("_get_atm_option_data failed: %s", exc)
            return None

    def _fetch_option_data_via_api(self, today: str) -> dict | None:
        """Fetch option chain price via Dhan rolling option API (paper mode / fallback)."""
        url = "https://api.dhan.co/v2/charts/rollingoption"
        results = {}
        for opt_type in ("CALL", "PUT"):
            payload = {
                "exchangeSegment": "NSE_FNO",
                "interval": "5",
                "securityId": "13",
                "instrument": "OPTIDX",
                "expiryFlag": "WEEK",
                "expiryCode": 1,
                "strike": "ATM",
                "drvOptionType": opt_type,
                "requiredData": ["close", "strike"],
                "fromDate": today,
                "toDate": today,
            }
            try:
                r = requests.post(url, json=payload, headers=self._headers, timeout=10)
                if r.status_code != 200:
                    continue
                raw = r.json()
                series = raw.get("data", {}).get("ce" if opt_type == "CALL" else "pe", {})
                if series and series.get("close"):
                    price = float(series["close"][-1])
                    strike = float(series["strike"][-1]) if series.get("strike") else 0
                    results[opt_type] = {"price": price, "strike": strike}
            except Exception as exc:
                logger.debug("_fetch_option_data_via_api %s: %s", opt_type, exc)

        if "CALL" not in results and "PUT" not in results:
            return None

        result = {
            "spot": results.get("CALL", results.get("PUT", {})).get("strike", 0),
            "atm_strike": results.get("CALL", results.get("PUT", {})).get("strike", 0),
            "ce_price": results.get("CALL", {}).get("price", 0),
            "pe_price": results.get("PUT", {}).get("price", 0),
            "ce_id": "",
            "pe_id": "",
            "expiry": "",
        }
        self._option_chain_cache[today] = result
        return result

    def _get_current_option_price(self, opt_type: str, security_id: str) -> float | None:
        """
        Get the latest price of an open option position.
        opt_type: 'CE' or 'PE'. Uses fresh option chain data (not cached).
        """
        today = datetime.now(IST).strftime("%Y-%m-%d")
        # Clear cache to force fresh fetch for mid-trade price
        self._option_chain_cache.pop(today, None)
        chain = self._get_atm_option_data()
        if chain is None:
            return None
        return chain["ce_price"] if opt_type == "CE" else chain["pe_price"]

    # ------------------------------------------------------------------
    # Dhan data fetch
    # ------------------------------------------------------------------

    def _fetch_intraday_candles(self, date_str: str) -> pd.DataFrame:
        """Fetch 5-min NIFTY index candles for date_str (YYYY-MM-DD)."""
        payload = {
            "securityId": "13",
            "exchangeSegment": "IDX_I",
            "instrument": "INDEX",
            "interval": "5",
            "oi_data_required": False,
            "fromDate": date_str,
            "toDate": date_str,
        }
        try:
            r = requests.post(INTRADAY_URL, json=payload, headers=self._headers, timeout=15)
            if r.status_code != 200:
                logger.warning("Dhan API %s: %s", r.status_code, r.text[:200])
                return pd.DataFrame()
            data = r.json()
            if not isinstance(data, dict) or not data.get("close"):
                return pd.DataFrame()
            df = pd.DataFrame({
                "datetime": pd.to_datetime(data["timestamp"], unit="s", utc=True)
                            .tz_convert(IST).tz_localize(None),
                "open":  data["open"],
                "high":  data["high"],
                "low":   data["low"],
                "close": data["close"],
            })
            return df.set_index("datetime").sort_index()
        except Exception as exc:
            logger.error("_fetch_intraday_candles error: %s", exc)
            return pd.DataFrame()

    def fetch_spot(self) -> float | None:
        """Return latest NIFTY spot from today's intraday candles."""
        today = datetime.now(IST).strftime("%Y-%m-%d")
        df = self._fetch_intraday_candles(today)
        if df.empty:
            return None
        return float(df["close"].iloc[-1])

    # ------------------------------------------------------------------
    # ADX (Wilder EMA — identical to backtester)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_adx(df: pd.DataFrame, period: int = 14) -> float | None:
        if not {"high", "low", "close"}.issubset(df.columns):
            return None
        if len(df) < period * 2:
            return None

        high  = df["high"]
        low   = df["low"]
        close = df["close"]

        plus_dm  = high.diff()
        minus_dm = -low.diff()
        plus_dm  = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        a        = 1 / period
        atr      = tr.ewm(alpha=a, adjust=False, min_periods=period).mean()
        plus_di  = 100 * plus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=a, adjust=False, min_periods=period).mean() / atr
        denom    = (plus_di + minus_di).replace(0, float("nan"))
        dx       = ((plus_di - minus_di).abs() / denom) * 100
        adx      = dx.ewm(alpha=a, adjust=False, min_periods=period).mean()

        val = adx.iloc[-1]
        return float(val) if pd.notna(val) else None

    def _get_live_adx(self) -> float | None:
        today = datetime.now(IST).strftime("%Y-%m-%d")
        df = self._fetch_intraday_candles(today)
        if df.empty:
            return None
        return self._compute_adx(df, period=self.params.adx_period)

    # ------------------------------------------------------------------
    # Z-score (pure Python, no pandas overhead per tick)
    # ------------------------------------------------------------------

    def _compute_zscore(self) -> float | None:
        if len(self._prices) < self.params.lookback:
            return None
        prices = list(self._prices)
        mean_val = sum(prices) / len(prices)
        variance = sum((x - mean_val) ** 2 for x in prices) / (len(prices) - 1)
        std = variance ** 0.5
        if std == 0:
            return 0.0
        return (prices[-1] - mean_val) / std

    # ------------------------------------------------------------------
    # Session window
    # ------------------------------------------------------------------

    def _in_session(self) -> bool:
        now = datetime.now(IST).strftime("%H:%M")
        return self.params.session_start <= now <= self.params.session_end

    # ------------------------------------------------------------------
    # Daily state management
    # ------------------------------------------------------------------

    def _check_reset_daily(self):
        today = datetime.now(IST).strftime("%Y-%m-%d")
        if today != self._today:
            if self._today:
                logger.info(
                    "Day roll %s → %s | prev_pnl=%.2f trades=%d",
                    self._today, today, self._daily_pnl, self._daily_trades,
                )
            self._today = today
            self._daily_pnl = 0.0
            self._daily_trades = 0
            # Force-close any overnight carry
            if self.current_position is not None:
                spot = self.fetch_spot()
                if spot:
                    self._exit_position(spot, "EOD Square Off")

    # ------------------------------------------------------------------
    # Broker hooks (override for live routing)
    # ------------------------------------------------------------------

    def place_live_order(self, side: str, price: float):
        """Place a live ATM option BUY order via Dhan (CE for LONG, PE for SHORT)."""
        if self._dhan is None:
            logger.error("place_live_order: dhan client not initialised")
            return
        if not self.params.use_options:
            # Futures fallback (original behaviour)
            transaction = self._dhan.BUY if side == "LONG" else self._dhan.SELL
            security_id = self.params.nifty_security_id
            exchange = self._dhan.NSE
        else:
            # BUY CE on LONG, BUY PE on SHORT
            chain = self.current_position.get("_option_chain") if self.current_position else None
            if not chain:
                logger.error("place_live_order: option chain missing on position")
                return
            security_id = chain["ce_id"] if side == "LONG" else chain["pe_id"]
            if not security_id:
                logger.error("place_live_order: missing security_id for %s option", side)
                return
            transaction = self._dhan.BUY
            exchange = self._dhan.NSE_FNO
        try:
            resp = self._dhan.place_order(
                security_id=str(security_id),
                exchange_segment=exchange,
                transaction_type=transaction,
                quantity=self.params.qty * self.params.option_lot_size,
                order_type=self._dhan.MARKET,
                product_type=self._dhan.INTRA,
                price=0,
            )
            logger.info("[LIVE ORDER] BUY %s option side=%s qty=%d | resp=%s",
                        "CE" if side == "LONG" else "PE", side,
                        self.params.qty * self.params.option_lot_size, resp)
            if resp and resp.get("status") == "success":
                self._live_order_id = resp.get("data", {}).get("orderId")
        except Exception as exc:
            logger.error("place_live_order failed: %s", exc)

    def close_live_order(self, side: str, price: float, reason: str):
        """Close/sell the open option position via market SELL order."""
        if self._dhan is None:
            logger.error("close_live_order: dhan client not initialised")
            return
        if not self.params.use_options:
            # Futures fallback
            transaction = self._dhan.SELL if side == "LONG" else self._dhan.BUY
            security_id = self.params.nifty_security_id
            exchange = self._dhan.NSE
        else:
            chain = self.current_position.get("_option_chain") if self.current_position else None
            if not chain:
                logger.error("close_live_order: option chain missing on position")
                return
            security_id = chain["ce_id"] if side == "LONG" else chain["pe_id"]
            if not security_id:
                logger.error("close_live_order: missing security_id for %s option", side)
                return
            transaction = self._dhan.SELL
            exchange = self._dhan.NSE_FNO
        try:
            time.sleep(0.5)
            resp = self._dhan.place_order(
                security_id=str(security_id),
                exchange_segment=exchange,
                transaction_type=transaction,
                quantity=self.params.qty * self.params.option_lot_size,
                order_type=self._dhan.MARKET,
                product_type=self._dhan.INTRA,
                price=0,
            )
            logger.info(
                "[LIVE CLOSE] SELL %s option side=%s @ %.2f | reason=%s | resp=%s",
                "CE" if side == "LONG" else "PE", side, price, reason, resp,
            )
            self._live_order_id = None
        except Exception as exc:
            logger.error("close_live_order failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal position management
    # ------------------------------------------------------------------

    def _enter_position(self, side: str, price: float):
        opt_type = "CE" if side == "LONG" else "PE"
        option_price = price  # fallback = spot
        chain = None

        if self.params.use_options:
            chain = self._get_atm_option_data()
            if chain is None:
                logger.warning("Cannot fetch option chain — skipping entry")
                return
            option_price = chain["ce_price"] if side == "LONG" else chain["pe_price"]
            if option_price <= 0:
                logger.warning("Zero option price for %s — skipping entry", opt_type)
                return

        self.current_position = {
            "side": side,
            "opt_type": opt_type,
            "entry_spot": price,
            "entry_price": option_price,   # option premium at entry
            "entry_time": datetime.now(IST).strftime("%H:%M:%S"),
            "qty": self.params.qty,
            "_option_chain": chain,        # store for live order routing
        }
        self._daily_trades += 1
        logger.info(
            "ENTRY %s %s @ %.2f (spot=%.2f) | trade#%d today | daily_pnl=%.2f",
            side, opt_type, option_price, price, self._daily_trades, self._daily_pnl,
        )
        if not self.paper_trade:
            self.place_live_order(side, option_price)

    def _exit_position(self, price: float, reason: str):
        if not self.current_position:
            return
        side      = self.current_position["side"]
        opt_type  = self.current_position["opt_type"]
        entry_opt = self.current_position["entry_price"]
        qty       = self.current_position["qty"]
        lot_units = self.params.option_lot_size if self.params.use_options else 1

        # Fetch current option price for exit
        exit_opt = price  # fallback = spot
        if self.params.use_options:
            cur_opt = self._get_current_option_price(opt_type, "")
            if cur_opt is not None and cur_opt > 0:
                exit_opt = cur_opt
            else:
                exit_opt = entry_opt  # neutral if unavailable

        pnl = (exit_opt - entry_opt) * qty * lot_units  # always BUY option → sell at exit
        self.realized_pnl   += pnl
        self._daily_pnl     += pnl
        self.unrealized_pnl  = 0.0

        logger.info(
            "EXIT %s %s @ %.2f (entry=%.2f) | pnl=%.2f | reason=%s | total_realized=%.2f",
            side, opt_type, exit_opt, entry_opt, pnl, reason, self.realized_pnl,
        )
        if not self.paper_trade:
            self.close_live_order(side, exit_opt, reason)

        self._save_trade_to_sf(
            side=side, opt_type=opt_type,
            entry=entry_opt, exit_price=exit_opt,
            pnl=pnl, reason=reason,
        )

        self.current_position = None

    # ------------------------------------------------------------------
    # Main tick (call every ~1 min from scheduler)
    # ------------------------------------------------------------------

    def run_iteration(self):
        """Single strategy tick. Wire to apscheduler/threading.Timer every minute."""
        if self.paused:
            return

        self._check_reset_daily()

        # Outside session — auto square-off if open
        if not self._in_session():
            if self.current_position is not None:
                spot = self.fetch_spot()
                if spot:
                    self._exit_position(spot, "EOD Square Off")
            return

        spot = self.fetch_spot()
        if spot is None:
            logger.warning("fetch_spot returned None")
            return

        self._prices.append(spot)
        z = self._compute_zscore()

        if z is None:
            logger.info("Warmup %d/%d bars", len(self._prices), self.params.lookback)
            return

        # --- Manage open position ---
        if self.current_position is not None:
            side     = self.current_position["side"]
            opt_type = self.current_position["opt_type"]
            entry    = self.current_position["entry_price"]
            qty      = self.current_position["qty"]
            lot_units = self.params.option_lot_size if self.params.use_options else 1

            # Get current option price for live P&L and stop checks
            if self.params.use_options:
                cur_opt = self._get_current_option_price(opt_type, "")
                cur_price = cur_opt if cur_opt and cur_opt > 0 else entry
            else:
                cur_price = spot

            live_pnl = (cur_price - entry) * qty * lot_units
            self.unrealized_pnl = live_pnl

            should_exit, reason = False, ""

            if side == "LONG":
                if z >= -self.params.exit_z:
                    should_exit, reason = True, "Mean Reversion Exit"
                elif z <= -self.params.stop_z:
                    should_exit, reason = True, "Z-Stop Loss"
            else:
                if z <= self.params.exit_z:
                    should_exit, reason = True, "Mean Reversion Exit"
                elif z >= self.params.stop_z:
                    should_exit, reason = True, "Z-Stop Loss"

            # Per-trade rupee stop
            if not should_exit and self.params.max_loss_per_trade > 0:
                if live_pnl <= -self.params.max_loss_per_trade:
                    should_exit, reason = True, "Rupee Stop Loss"

            if should_exit:
                self._exit_position(spot, reason)
            return

        # --- Look for new entry ---

        # Daily loss cap
        if self.params.max_daily_loss > 0 and self._daily_pnl <= -self.params.max_daily_loss:
            logger.info("Daily loss cap hit (₹%.0f) — skipping entries", self._daily_pnl)
            return

        # Daily trade count cap
        if self.params.max_trades_per_day > 0 and self._daily_trades >= self.params.max_trades_per_day:
            logger.info("Max %d trades/day reached — skipping entry", self._daily_trades)
            return

        # ADX regime filter
        if self.params.use_adx_filter:
            adx = self._get_live_adx()
            if adx is None:
                logger.info("ADX not yet computable — skipping entry")
                return
            if adx > self.params.max_adx:
                logger.info("ADX=%.2f > %.2f — trending market, skip", adx, self.params.max_adx)
                return

        # Z-score entry signal
        if z <= -self.params.entry_z:
            self._enter_position("LONG", spot)
        elif z >= self.params.entry_z:
            self._enter_position("SHORT", spot)

    # ------------------------------------------------------------------
    # Salesforce save (paper/forward-test trades only)
    # ------------------------------------------------------------------

    def _save_trade_to_sf(self, side: str, opt_type: str, entry: float, exit_price: float, pnl: float, reason: str):
        """Persist completed forward-test trade to Salesforce historical_backtests__c."""
        try:
            import os
            from simple_salesforce import Salesforce
            from datetime import timezone

            sf = Salesforce(
                username=os.getenv("SF_USERNAME"),
                password=os.getenv("SF_PASSWORD"),
                security_token=os.getenv("SF_SECURITY_TOKEN"),
                domain=os.getenv("SF_DOMAIN", "login"),
            )
            now_utc = datetime.now(IST).astimezone(timezone.utc)
            capital = self.params.initial_capital
            record = {
                "Run_Date__c":       now_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "Strategy_Name__c":  "zscore_nifty",
                "Run_Mode__c":       "Forward Test" if self.paper_trade else "Live Trade",
                "Date__c":           datetime.now(IST).strftime("%Y-%m-%d"),
                "Entry_Time__c":     self.current_position["entry_time"] if self.current_position else "",
                "Exit_Time__c":      datetime.now(IST).strftime("%H:%M:%S"),
                "Option_Type__c":    opt_type,
                "Action__c":         side,
                "Qty__c":            self.params.qty,
                "Buy_Price__c":      round(entry, 2),
                "Sell_Price__c":     round(exit_price, 2),
                "Peak_Price__c":     round(exit_price, 2),
                "PNL__c":            round(pnl, 2),
                "ROI__c":            round((pnl / capital) * 100, 4) if capital else 0,
                "Capital_ROI__c":    round((pnl / capital) * 100, 4) if capital else 0,
                "Reason__c":         reason,
                "Win__c":            pnl > 0,
                "Parameters__c": (
                    f"lookback={self.params.lookback}|entry_z={self.params.entry_z}|"
                    f"exit_z={self.params.exit_z}|stop_z={self.params.stop_z}|"
                    f"adx_filter={self.params.use_adx_filter}|max_adx={self.params.max_adx}|"
                    f"rupee_stop={self.params.max_loss_per_trade}|daily_cap={self.params.max_daily_loss}|"
                    f"max_trades={self.params.max_trades_per_day}|session={self.params.session_start}-{self.params.session_end}"
                ),
            }
            sf.historical_backtests__c.create(record)
            logger.info("Trade saved to Salesforce | side=%s %s pnl=%.2f", side, opt_type, pnl)
        except Exception as exc:
            logger.error("_save_trade_to_sf failed: %s", exc)

    # ------------------------------------------------------------------
    # Status snapshot
    # ------------------------------------------------------------------

    def status(self) -> dict:
        return {
            "paper_trade":    self.paper_trade,
            "session":        f"{self.params.session_start}–{self.params.session_end}",
            "in_session":     self._in_session(),
            "position":       self.current_position,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
            "realized_pnl":   round(self.realized_pnl, 2),
            "daily_pnl":      round(self._daily_pnl, 2),
            "daily_trades":   self._daily_trades,
            "params": {
                "lookback":           self.params.lookback,
                "entry_z":            self.params.entry_z,
                "exit_z":             self.params.exit_z,
                "stop_z":             self.params.stop_z,
                "qty":                self.params.qty,
                "use_adx_filter":     self.params.use_adx_filter,
                "max_adx":            self.params.max_adx,
                "max_loss_per_trade": self.params.max_loss_per_trade,
                "max_daily_loss":     self.params.max_daily_loss,
                "max_trades_per_day": self.params.max_trades_per_day,
            },
        }
