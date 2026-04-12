import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ZScoreParams:
    lookback: int = 30
    entry_z: float = 1.5
    exit_z: float = 0.3
    stop_z: float = 2.5
    qty: int = 1                    # number of lots for options (1 lot = option_lot_size units)
    initial_capital: float = 500000.0
    use_adx_filter: bool = False
    adx_period: int = 14
    max_adx: float = 20.0
    max_loss_per_trade: float = 0.0   # 0 = disabled; rupee stop-loss per trade
    max_daily_loss: float = 0.0       # 0 = disabled; skip rest of day if daily loss exceeds this
    max_trades_per_day: int = 0       # 0 = disabled; cap new entries per calendar day
    # Options mode
    use_options: bool = True          # True = trade CE/PE; False = trade spot P&L
    option_lot_size: int = 25         # NIFTY standard lot size (25 units per lot)


class NiftyZScoreBacktester:
    """Backtester for NIFTY z-score mean-reversion strategy.

    Signal source: NIFTY spot 5-min candles (z-score on close).
    Trade instrument: ATM CALL (LONG signal) or ATM PUT (SHORT signal).

    When use_options=True (default), pass an option_fetcher to run().
    The option_fetcher must implement:
        fetch(date_str: str, opt_type: str) -> dict[datetime, {close, strike}]
    where opt_type is 'C' (call) or 'P' (put).

    Falls back to synthetic spot P&L when option_fetcher is None.
    """

    def __init__(self, params=None):
        self.params = params or ZScoreParams()
        self.results = []
        self.current_capital = self.params.initial_capital

    @staticmethod
    def _compute_adx(df, period=14):
        if not {"high", "low", "close"}.issubset(df.columns):
            return pd.Series(index=df.index, dtype="float64")

        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)).replace([float("inf")], 0.0) * 100
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _prep_df(self, df):
        out = df.copy()
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"])
            out = out.set_index("datetime")
        out = out.sort_index()
        if "close" not in out.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        out["mean"] = out["close"].rolling(self.params.lookback).mean()
        out["std"] = out["close"].rolling(self.params.lookback).std(ddof=1)
        out["z"] = (out["close"] - out["mean"]) / out["std"]
        if self.params.use_adx_filter:
            out["adx"] = self._compute_adx(out, period=self.params.adx_period)
        return out

    # ------------------------------------------------------------------
    # Option price lookup helper
    # ------------------------------------------------------------------

    @staticmethod
    def _get_option_price(option_fetcher, date_str: str, opt_type: str, ts) -> float | None:
        """Look up the same-bar option close price; return None if unavailable."""
        if option_fetcher is None:
            return None
        try:
            bars = option_fetcher.fetch(date_str, opt_type)
            bar_key = ts.replace(second=0, microsecond=0)
            entry = bars.get(bar_key)
            if entry:
                return float(entry["close"])
            # Try exact timestamp without microseconds
            for k, v in bars.items():
                if abs((k - bar_key).total_seconds()) <= 30:
                    return float(v["close"])
        except Exception as exc:
            logger.debug("_get_option_price error: %s", exc)
        return None

    def run(self, df, option_fetcher=None):
        """
        Run the backtester on a NIFTY spot DataFrame.

        Args:
            df: DataFrame with NIFTY spot 5-min candles (index=datetime, col=close)
            option_fetcher: optional object with .fetch(date_str, opt_type) for option prices.
                            Required when params.use_options=True for real P&L.
                            Falls back to synthetic spot P&L if None.
        """
        data = self._prep_df(df)
        # effective trade qty
        lot_units = self.params.option_lot_size if self.params.use_options else 1
        trade_qty = self.params.qty * lot_units

        position = None
        daily_pnl: dict = {}    # date_str -> cumulative pnl for that day
        daily_trades: dict = {} # date_str -> trade count for that day
        prev_day: str = ""

        for ts, row in data.iterrows():
            spot = float(row["close"])
            z = row["z"]
            cur_day = ts.strftime("%Y-%m-%d")

            # Force-close any open position at the day boundary (intraday square-off)
            if prev_day and cur_day != prev_day and position is not None:
                opt_type = position["opt_type"]      # 'C' or 'P' (or None for spot mode)
                entry_opt = position["entry_opt_price"]
                # Try to get closing option price from previous day's last bar
                exit_opt = None
                if self.params.use_options and option_fetcher and opt_type:
                    exit_opt = self._get_option_price(option_fetcher, prev_day, opt_type, ts)
                if exit_opt is None:
                    # Fallback: use spot-based approximation
                    exit_opt = entry_opt  # net zero on forced EOD (conservative)
                pnl = round((exit_opt - entry_opt) * trade_qty, 2)
                cap_before = self.current_capital
                self.current_capital += pnl
                daily_pnl[prev_day] = daily_pnl.get(prev_day, 0.0) + pnl
                daily_trades[prev_day] = daily_trades.get(prev_day, 0) + 1
                self.results.append({
                    "Date": prev_day,
                    "Entry_Time": position["entry_time"].strftime("%H:%M:%S"),
                    "Exit_Time": "15:00:00",
                    "Option_Type": "CE" if opt_type == "C" else ("PE" if opt_type == "P" else "SPOT"),
                    "Action": position["side"],
                    "Qty": self.params.qty,
                    "Entry_Price": round(entry_opt, 2),
                    "Exit_Price": round(exit_opt, 2),
                    "Strike": position.get("strike", ""),
                    "ZScore": round(float(row["z"]) if pd.notna(row["z"]) else 0.0, 4),
                    "ADX": None,
                    "PNL": pnl,
                    "Capital_ROI%": round((pnl / cap_before) * 100, 4) if cap_before else 0,
                    "Reason": "EOD Square Off",
                    "Win": pnl > 0,
                })
                position = None
            prev_day = cur_day

            if pd.isna(z):
                continue

            if self.params.use_adx_filter:
                adx_val = row.get("adx")
                if pd.isna(adx_val):
                    continue
            else:
                adx_val = None

            if position is None:
                day_key = ts.strftime("%Y-%m-%d")

                # Daily max-loss gate
                if self.params.max_daily_loss > 0:
                    if daily_pnl.get(day_key, 0.0) <= -self.params.max_daily_loss:
                        continue

                # Daily trade count gate
                if self.params.max_trades_per_day > 0:
                    if daily_trades.get(day_key, 0) >= self.params.max_trades_per_day:
                        continue

                if self.params.use_adx_filter and adx_val > self.params.max_adx:
                    continue

                target_side = None
                if z <= -self.params.entry_z:
                    target_side = "LONG"    # price too low → mean revert up → buy CALL
                elif z >= self.params.entry_z:
                    target_side = "SHORT"   # price too high → mean revert down → buy PUT

                if target_side is None:
                    continue

                # Resolve option price at entry bar
                opt_type = "C" if target_side == "LONG" else "P"
                if self.params.use_options and option_fetcher:
                    entry_opt_price = self._get_option_price(option_fetcher, day_key, opt_type, ts)
                    if entry_opt_price is None:
                        logger.debug("No option price for %s %s @ %s — skip", opt_type, day_key, ts)
                        continue
                    strike_info = ""
                    try:
                        bars = option_fetcher.fetch(day_key, opt_type)
                        bar_key = ts.replace(second=0, microsecond=0)
                        for k, v in bars.items():
                            if abs((k - bar_key).total_seconds()) <= 30:
                                strike_info = str(v.get("strike", "ATM"))
                                break
                    except Exception:
                        pass
                else:
                    # Spot fallback: use spot price as "entry price"
                    entry_opt_price = spot
                    opt_type = None
                    strike_info = ""

                position = {
                    "side": target_side,
                    "entry_spot": spot,
                    "entry_opt_price": entry_opt_price,
                    "opt_type": opt_type,
                    "strike": strike_info,
                    "entry_time": ts,
                }
                continue

            # Managing open position
            side = position["side"]
            opt_type = position["opt_type"]

            # Fetch current option price for live P&L and stop checks
            if self.params.use_options and option_fetcher and opt_type:
                cur_opt_price = self._get_option_price(option_fetcher, cur_day, opt_type, ts)
                if cur_opt_price is None:
                    cur_opt_price = position["entry_opt_price"]  # hold if API gap
            else:
                cur_opt_price = spot  # spot fallback

            entry_opt = position["entry_opt_price"]
            live_pnl = (cur_opt_price - entry_opt) * trade_qty  # always BUY option

            should_exit = False
            reason = ""

            # Z-score exit / stop signals (same as before, based on spot z)
            if side == "LONG":
                if z >= -self.params.exit_z:
                    should_exit = True
                    reason = "Mean Reversion Exit"
                elif z <= -self.params.stop_z:
                    should_exit = True
                    reason = "Z-Stop Loss"
            else:
                if z <= self.params.exit_z:
                    should_exit = True
                    reason = "Mean Reversion Exit"
                elif z >= self.params.stop_z:
                    should_exit = True
                    reason = "Z-Stop Loss"

            # Per-trade rupee stop-loss
            if not should_exit and self.params.max_loss_per_trade > 0:
                if live_pnl <= -self.params.max_loss_per_trade:
                    should_exit = True
                    reason = "Rupee Stop Loss"

            if should_exit:
                pnl = round(live_pnl, 2)
                cap_before = self.current_capital
                self.current_capital += pnl

                day_key = ts.strftime("%Y-%m-%d")
                daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pnl
                daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

                self.results.append({
                    "Date": ts.strftime("%Y-%m-%d"),
                    "Entry_Time": position["entry_time"].strftime("%H:%M:%S"),
                    "Exit_Time": ts.strftime("%H:%M:%S"),
                    "Option_Type": "CE" if opt_type == "C" else ("PE" if opt_type == "P" else "SPOT"),
                    "Action": side,
                    "Qty": self.params.qty,
                    "Entry_Price": round(entry_opt, 2),
                    "Exit_Price": round(cur_opt_price, 2),
                    "Strike": position.get("strike", ""),
                    "ZScore": round(float(z), 4),
                    "ADX": round(float(adx_val), 2) if adx_val is not None else None,
                    "PNL": pnl,
                    "Capital_ROI%": round((pnl / cap_before) * 100, 4) if cap_before else 0,
                    "Reason": reason,
                    "Win": pnl > 0,
                })
                position = None

        # Force close any open position on final bar
        if position is not None and not data.empty:
            ts = data.index[-1]
            spot = float(data.iloc[-1]["close"])
            opt_type = position["opt_type"]
            entry_opt = position["entry_opt_price"]
            day_key = ts.strftime("%Y-%m-%d")

            if self.params.use_options and option_fetcher and opt_type:
                exit_opt = self._get_option_price(option_fetcher, day_key, opt_type, ts)
                if exit_opt is None:
                    exit_opt = entry_opt
            else:
                exit_opt = spot

            pnl = round((exit_opt - entry_opt) * trade_qty, 2)
            cap_before = self.current_capital
            self.current_capital += pnl

            self.results.append({
                "Date": ts.strftime("%Y-%m-%d"),
                "Entry_Time": position["entry_time"].strftime("%H:%M:%S"),
                "Exit_Time": ts.strftime("%H:%M:%S"),
                "Option_Type": "CE" if opt_type == "C" else ("PE" if opt_type == "P" else "SPOT"),
                "Action": position["side"],
                "Qty": self.params.qty,
                "Entry_Price": round(entry_opt, 2),
                "Exit_Price": round(exit_opt, 2),
                "Strike": position.get("strike", ""),
                "ZScore": round(float(data.iloc[-1]["z"]) if pd.notna(data.iloc[-1]["z"]) else 0.0, 4),
                "ADX": round(float(data.iloc[-1]["adx"]), 2) if self.params.use_adx_filter and "adx" in data.columns and pd.notna(data.iloc[-1]["adx"]) else None,
                "PNL": pnl,
                "Capital_ROI%": round((pnl / cap_before) * 100, 4) if cap_before else 0,
                "Reason": "Force Exit (Last Bar)",
                "Win": pnl > 0,
            })

        return self.results

    def summary(self):
        if not self.results:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "return_pct": 0.0,
            }

        df = pd.DataFrame(self.results)
        total_pnl = float(df["PNL"].sum())
        total_trades = len(df)
        win_rate = float(df["Win"].mean() * 100)
        return_pct = ((self.current_capital - self.params.initial_capital) / self.params.initial_capital) * 100

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(float(return_pct), 2),
        }


    @staticmethod
    def _compute_adx(df, period=14):
        if not {"high", "low", "close"}.issubset(df.columns):
            return pd.Series(index=df.index, dtype="float64")

        high = df["high"]
        low = df["low"]
        close = df["close"]

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / atr
        dx = ((plus_di - minus_di).abs() / (plus_di + minus_di)).replace([float("inf")], 0.0) * 100
        return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    def _prep_df(self, df):
        out = df.copy()
        if "datetime" in out.columns:
            out["datetime"] = pd.to_datetime(out["datetime"])
            out = out.set_index("datetime")
        out = out.sort_index()
        if "close" not in out.columns:
            raise ValueError("DataFrame must contain a 'close' column")
        out["mean"] = out["close"].rolling(self.params.lookback).mean()
        out["std"] = out["close"].rolling(self.params.lookback).std(ddof=1)
        out["z"] = (out["close"] - out["mean"]) / out["std"]
        if self.params.use_adx_filter:
            out["adx"] = self._compute_adx(out, period=self.params.adx_period)
        return out

        position = None
        daily_pnl: dict = {}    # date_str -> cumulative pnl for that day
        daily_trades: dict = {} # date_str -> trade count for that day
        prev_day: str = ""

        for ts, row in data.iterrows():
            price = float(row["close"])
            z = row["z"]
            cur_day = ts.strftime("%Y-%m-%d")

            # Force-close any open position at the day boundary (intraday square-off)
            if prev_day and cur_day != prev_day and position is not None:
                side = position["side"]
                entry = position["entry_price"]
                close_price = price  # open of new day — approximate with current bar
                pnl = (close_price - entry) * self.params.qty if side == "LONG" else (entry - close_price) * self.params.qty
                cap_before = self.current_capital
                self.current_capital += pnl
                daily_pnl[prev_day] = daily_pnl.get(prev_day, 0.0) + pnl
                daily_trades[prev_day] = daily_trades.get(prev_day, 0) + 1
                self.results.append({
                    "Date": prev_day,
                    "Entry_Time": position["entry_time"].strftime("%H:%M:%S"),
                    "Exit_Time": "15:00:00",
                    "Action": side,
                    "Qty": self.params.qty,
                    "Entry_Price": round(entry, 2),
                    "Exit_Price": round(close_price, 2),
                    "ZScore": round(float(row["z"]) if pd.notna(row["z"]) else 0.0, 4),
                    "ADX": None,
                    "PNL": round(float(pnl), 2),
                    "Capital_ROI%": round((pnl / cap_before) * 100, 4) if cap_before else 0,
                    "Reason": "EOD Square Off",
                    "Win": pnl > 0,
                })
                position = None
            prev_day = cur_day

            if pd.isna(z):
                continue

            if self.params.use_adx_filter:
                adx_val = row.get("adx")
                if pd.isna(adx_val):
                    continue
            else:
                adx_val = None

            if position is None:
                day_key = ts.strftime("%Y-%m-%d")

                # Daily max-loss gate
                if self.params.max_daily_loss > 0:
                    if daily_pnl.get(day_key, 0.0) <= -self.params.max_daily_loss:
                        continue

                # Daily trade count gate
                if self.params.max_trades_per_day > 0:
                    if daily_trades.get(day_key, 0) >= self.params.max_trades_per_day:
                        continue

                if self.params.use_adx_filter and adx_val > self.params.max_adx:
                    continue
                if z <= -self.params.entry_z:
                    position = {
                        "side": "LONG",
                        "entry_price": price,
                        "entry_time": ts,
                    }
                elif z >= self.params.entry_z:
                    position = {
                        "side": "SHORT",
                        "entry_price": price,
                        "entry_time": ts,
                    }
                continue

            side = position["side"]
            should_exit = False
            reason = ""

            if side == "LONG":
                if z >= -self.params.exit_z:
                    should_exit = True
                    reason = "Mean Reversion Exit"
                elif z <= -self.params.stop_z:
                    should_exit = True
                    reason = "Z-Stop Loss"
            else:
                if z <= self.params.exit_z:
                    should_exit = True
                    reason = "Mean Reversion Exit"
                elif z >= self.params.stop_z:
                    should_exit = True
                    reason = "Z-Stop Loss"

            # Per-trade rupee stop-loss
            if not should_exit and self.params.max_loss_per_trade > 0:
                entry = position["entry_price"]
                live_pnl = (price - entry) * self.params.qty if side == "LONG" else (entry - price) * self.params.qty
                if live_pnl <= -self.params.max_loss_per_trade:
                    should_exit = True
                    reason = "Rupee Stop Loss"

            if should_exit:
                entry = position["entry_price"]
                pnl = (price - entry) * self.params.qty if side == "LONG" else (entry - price) * self.params.qty
                cap_before = self.current_capital
                self.current_capital += pnl

                # Accumulate daily pnl and trade count
                day_key = ts.strftime("%Y-%m-%d")
                daily_pnl[day_key] = daily_pnl.get(day_key, 0.0) + pnl
                daily_trades[day_key] = daily_trades.get(day_key, 0) + 1

                self.results.append(
                    {
                        "Date": ts.strftime("%Y-%m-%d"),
                        "Entry_Time": position["entry_time"].strftime("%H:%M:%S"),
                        "Exit_Time": ts.strftime("%H:%M:%S"),
                        "Action": side,
                        "Qty": self.params.qty,
                        "Entry_Price": round(entry, 2),
                        "Exit_Price": round(price, 2),
                        "ZScore": round(float(z), 4),
                        "ADX": round(float(adx_val), 2) if adx_val is not None else None,
                        "PNL": round(float(pnl), 2),
                        "Capital_ROI%": round((pnl / cap_before) * 100, 4) if cap_before else 0,
                        "Reason": reason,
                        "Win": pnl > 0,
                    }
                )
                position = None

        # Force close any open position on final bar
        if position is not None and not data.empty:
            ts = data.index[-1]
            price = float(data.iloc[-1]["close"])
            side = position["side"]
            entry = position["entry_price"]
            pnl = (price - entry) * self.params.qty if side == "LONG" else (entry - price) * self.params.qty
            cap_before = self.current_capital
            self.current_capital += pnl

            self.results.append(
                {
                    "Date": ts.strftime("%Y-%m-%d"),
                    "Entry_Time": position["entry_time"].strftime("%H:%M:%S"),
                    "Exit_Time": ts.strftime("%H:%M:%S"),
                    "Action": side,
                    "Qty": self.params.qty,
                    "Entry_Price": round(entry, 2),
                    "Exit_Price": round(price, 2),
                    "ZScore": round(float(data.iloc[-1]["z"]) if pd.notna(data.iloc[-1]["z"]) else 0.0, 4),
                    "ADX": round(float(data.iloc[-1]["adx"]), 2) if self.params.use_adx_filter and "adx" in data.columns and pd.notna(data.iloc[-1]["adx"]) else None,
                    "PNL": round(float(pnl), 2),
                    "Capital_ROI%": round((pnl / cap_before) * 100, 4) if cap_before else 0,
                    "Reason": "Force Exit (Last Bar)",
                    "Win": pnl > 0,
                }
            )

        return self.results

    def summary(self):
        if not self.results:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "return_pct": 0.0,
            }

        df = pd.DataFrame(self.results)
        total_pnl = float(df["PNL"].sum())
        total_trades = len(df)
        win_rate = float(df["Win"].mean() * 100)
        return_pct = ((self.current_capital - self.params.initial_capital) / self.params.initial_capital) * 100

        return {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "total_pnl": round(total_pnl, 2),
            "return_pct": round(float(return_pct), 2),
        }
