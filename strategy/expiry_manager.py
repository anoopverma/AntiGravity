"""
expiry_manager.py

SOLID design:
  (S) Each class has exactly one job.
  (O) Open/Closed: add scripts → SCRIPT_REGISTRY only; new expiry patterns → new Fetcher subclass.
  (L) All fetchers honour the same ExpiryFetcherBase contract and are interchangeable.
  (I) Thin single-method interface per fetcher.
  (D) ExpiryManager depends on ExpiryFetcherBase abstractions injected at runtime.

Expiry rules:
  ┌──────────────┬─────────────────────────────────┬────────────────────────────┐
  │ Script       │ Rule                            │ Holiday shift              │
  ├──────────────┼─────────────────────────────────┼────────────────────────────┤
  │ NIFTY 50     │ Weekly Tuesday                  │ Previous trading day       │
  │ BANKNIFTY    │ Monthly – last Tuesday of month │ Previous trading day       │
  │ SENSEX       │ Weekly Thursday                 │ Previous trading day       │
  │ FINNIFTY     │ Monthly – last Tuesday of month │ Previous trading day       │
  └──────────────┴─────────────────────────────────┴────────────────────────────┘
"""

from __future__ import annotations

import abc
import calendar
import datetime
import logging
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# (S) Enum: expiry schedule types
# ---------------------------------------------------------------------------

class ExpiryType(Enum):
    WEEKLY_BACKFILL  = auto()   # Next occurrence of weekday; holiday → prev trading day
    MONTHLY_LAST     = auto()   # Last weekday of month; holiday → prev trading day


# ---------------------------------------------------------------------------
# (S) Value objects / DTOs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScriptConfig:
    """Immutable descriptor for one tradeable index."""
    name: str
    dhan_id: int
    expiry_type: ExpiryType
    expiry_weekday: int   # 0=Mon … 6=Sun
    day_label: str        # Short human label, e.g. "Tue"


@dataclass(frozen=True)
class ExpiryRecord:
    """Output DTO — carries data, knows how to serialise itself."""
    script: str
    expiry_date: datetime.date
    day_label: str
    source: str   # "api" | "estimated" | "holiday-adjusted"

    def to_dict(self) -> dict[str, str]:
        return {
            "script":     self.script,
            "expiry":     self.expiry_date.strftime(f"%d %b %Y ({self.day_label})"),
            "expiry_iso": self.expiry_date.isoformat(),
            "day_label":  self.day_label,
            "source":     self.source,
        }


# ---------------------------------------------------------------------------
# (O) Registry — add a new script here; nothing else needs editing
# ---------------------------------------------------------------------------

SCRIPT_REGISTRY: list[ScriptConfig] = [
    ScriptConfig("NIFTY 50",  13, ExpiryType.WEEKLY_BACKFILL,  1, "Tue"),  # Weekly Tue
    ScriptConfig("BANKNIFTY", 25, ExpiryType.MONTHLY_LAST,     1, "Tue"),  # Last Tue of month
    ScriptConfig("SENSEX",    51, ExpiryType.WEEKLY_BACKFILL,  3, "Thu"),  # Weekly Thu
    ScriptConfig("FINNIFTY",  27, ExpiryType.MONTHLY_LAST,     1, "Tue"),  # Last Tue of month
]


# ---------------------------------------------------------------------------
# Shared helper — Dhan API trading-day validation
# ---------------------------------------------------------------------------

def _is_trading_day(dhan_client: Any, dhan_id: int, date: datetime.date) -> bool:
    """Returns True if Dhan confirms a valid option chain for this date.
    Falls back to True on any error to avoid false negatives.
    """
    try:
        resp = dhan_client.option_chain(dhan_id, "IDX_I", date.isoformat())
        return (
            resp.get("status") == "success"
            and resp.get("data", {}).get("status") != "failed"
        )
    except Exception as exc:
        logger.warning("Trading-day check failed (id=%s, date=%s): %s", dhan_id, date, exc)
        return True  # assume valid; avoid false negatives


def _backfill_to_trading_day(
    candidate: datetime.date,
    not_before: datetime.date,
    dhan_client: Optional[Any],
    dhan_id: int,
    max_backtrack: int = 4,
) -> tuple[datetime.date, str]:
    """Step back from candidate until a trading day is found or we hit not_before.
    Returns (date, source_label).
    """
    for offset in range(max_backtrack + 1):
        day = candidate - datetime.timedelta(days=offset)
        if day < not_before:
            break
        if dhan_client is None or _is_trading_day(dhan_client, dhan_id, day):
            source = "estimated" if dhan_client is None else "api"
            if offset > 0:
                source = "holiday-adjusted"
                logger.info("Holiday detected on %s → shifted back to %s", candidate, day)
            return day, source

    # Last resort: return unvalidated candidate
    logger.warning("Could not validate trading day near %s; falling back to candidate.", candidate)
    return candidate, "estimated"


# ---------------------------------------------------------------------------
# (I / L) Abstract fetcher contract
# ---------------------------------------------------------------------------

class ExpiryFetcherBase(abc.ABC):
    """Single-method interface: ScriptConfig + reference date → ExpiryRecord."""

    @abc.abstractmethod
    def resolve_expiry(self, cfg: ScriptConfig, reference: datetime.date) -> ExpiryRecord:
        ...


# ---------------------------------------------------------------------------
# Fetcher 1 — Weekly with backfill  (NIFTY 50: Tue, SENSEX: Thu)
# ---------------------------------------------------------------------------

class WeeklyBackfillFetcher(ExpiryFetcherBase):
    """Finds the next occurrence of cfg.expiry_weekday and steps back on holidays."""

    def __init__(self, dhan_client: Optional[Any] = None) -> None:
        self._dhan = dhan_client

    def resolve_expiry(self, cfg: ScriptConfig, reference: datetime.date) -> ExpiryRecord:
        days_ahead = (cfg.expiry_weekday - reference.weekday()) % 7
        target = reference + datetime.timedelta(days=days_ahead)

        expiry_date, source = _backfill_to_trading_day(
            candidate=target,
            not_before=reference,
            dhan_client=self._dhan,
            dhan_id=cfg.dhan_id,
        )
        return ExpiryRecord(
            script=cfg.name,
            expiry_date=expiry_date,
            day_label=expiry_date.strftime("%a"),
            source=source,
        )


# ---------------------------------------------------------------------------
# Fetcher 2 — Monthly last-weekday with backfill  (BANKNIFTY, FINNIFTY: last Tue)
# ---------------------------------------------------------------------------

class MonthlyLastWeekdayFetcher(ExpiryFetcherBase):
    """Finds the last cfg.expiry_weekday of the current (or next) month and backfills on holidays."""

    def __init__(self, dhan_client: Optional[Any] = None) -> None:
        self._dhan = dhan_client

    @staticmethod
    def _last_weekday_of_month(year: int, month: int, weekday: int) -> datetime.date:
        last_day = calendar.monthrange(year, month)[1]
        last_date = datetime.date(year, month, last_day)
        days_back = (last_date.weekday() - weekday) % 7
        return last_date - datetime.timedelta(days=days_back)

    def resolve_expiry(self, cfg: ScriptConfig, reference: datetime.date) -> ExpiryRecord:
        target = self._last_weekday_of_month(reference.year, reference.month, cfg.expiry_weekday)

        # If this month's expiry is already past, roll to next month
        if target < reference:
            if reference.month == 12:
                target = self._last_weekday_of_month(reference.year + 1, 1, cfg.expiry_weekday)
            else:
                target = self._last_weekday_of_month(reference.year, reference.month + 1, cfg.expiry_weekday)

        expiry_date, source = _backfill_to_trading_day(
            candidate=target,
            not_before=reference,
            dhan_client=self._dhan,
            dhan_id=cfg.dhan_id,
        )
        return ExpiryRecord(
            script=cfg.name,
            expiry_date=expiry_date,
            day_label=expiry_date.strftime("%a"),
            source=source,
        )


# ---------------------------------------------------------------------------
# Fetcher 3 — Pure weekday arithmetic (no network; offline fallback)
# ---------------------------------------------------------------------------

class WeekdayFallbackFetcher(ExpiryFetcherBase):
    """Offline estimate using simple weekday arithmetic. No API calls.
    NOTE: inaccurate on holidays; use only when Dhan client is unavailable.
    """

    def resolve_expiry(self, cfg: ScriptConfig, reference: datetime.date) -> ExpiryRecord:
        days_ahead = (cfg.expiry_weekday - reference.weekday()) % 7
        expiry_date = reference + datetime.timedelta(days=days_ahead)
        return ExpiryRecord(
            script=cfg.name,
            expiry_date=expiry_date,
            day_label=cfg.day_label,
            source="estimated",
        )


# ---------------------------------------------------------------------------
# (D) Orchestrator — fully data-driven dispatch via ExpiryType on ScriptConfig
# ---------------------------------------------------------------------------

class ExpiryManager:
    """
    Orchestrates expiry resolution for all registered scripts.
    Fetcher is chosen per-script based on ScriptConfig.expiry_type (data-driven, O/C-compliant).
    """

    _FETCHER_MAP = {
        ExpiryType.WEEKLY_BACKFILL: WeeklyBackfillFetcher,
        ExpiryType.MONTHLY_LAST:   MonthlyLastWeekdayFetcher,
    }

    def __init__(
        self,
        registry: Optional[list[ScriptConfig]] = None,
        dhan_client: Optional[Any] = None,
        # Legacy compatibility: ignored if dhan_client is provided
        fetcher: Optional[ExpiryFetcherBase] = None,
    ) -> None:
        self._registry: list[ScriptConfig] = registry if registry is not None else SCRIPT_REGISTRY
        self._dhan = dhan_client

    def _fetcher_for(self, cfg: ScriptConfig) -> ExpiryFetcherBase:
        """Instantiate the correct fetcher for this script's ExpiryType."""
        fetcher_cls = self._FETCHER_MAP.get(cfg.expiry_type)
        if fetcher_cls:
            return fetcher_cls(dhan_client=self._dhan)
        # Fallback for any unmapped type
        return WeekdayFallbackFetcher()

    def get_upcoming_expiries(
        self, reference: Optional[datetime.date] = None
    ) -> list[ExpiryRecord]:
        today = reference if reference is not None else datetime.date.today()
        return [self._fetcher_for(cfg).resolve_expiry(cfg, today) for cfg in self._registry]

    def to_json(self, reference: Optional[datetime.date] = None) -> list[dict[str, str]]:
        """Convenience serialiser — returns list of dicts ready for JSON response."""
        return [r.to_dict() for r in self.get_upcoming_expiries(reference)]
