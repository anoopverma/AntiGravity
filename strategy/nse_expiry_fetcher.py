"""
nse_expiry_fetcher.py

Fetches real option chain expiry dates from NSE India using a headless browser
session (Playwright) to bypass Akamai bot protection.

SOLID:
  (S) Single responsibility: only fetches & parses NSE expiry dates.
  (O) Open/Closed: swap out transport by subclassing NseSessionBase.
  (D) Dependency Inversion: NseExpiryFetcher accepts an NseSessionBase.
"""

from __future__ import annotations

import abc
import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

NSE_SYMBOLS: dict[str, str] = {
    "NIFTY 50":   "NIFTY",
    "BANKNIFTY":  "BANKNIFTY",
    "FINNIFTY":   "FINNIFTY",
    "MIDCPNIFTY": "MIDCPNIFTY",
    "SENSEX":     "SENSEX",    # BSE — not available on NSE chain; handled separately
}


# ---------------------------------------------------------------------------
# (I) Abstract session contract
# ---------------------------------------------------------------------------

class NseSessionBase(abc.ABC):
    @abc.abstractmethod
    def fetch_expiry_dates(self, symbol: str) -> list[str]:
        """Return list of expiry date strings in 'DD-Mon-YYYY' format."""
        ...


# ---------------------------------------------------------------------------
# Playwright-based session (real browser, bypasses bot protection)
# ---------------------------------------------------------------------------

class PlaywrightNseSession(NseSessionBase):
    """Uses Playwright to open a real browser session and call NSE API."""

    def __init__(self, headless: bool = True) -> None:
        self._headless = headless

    def fetch_expiry_dates(self, symbol: str) -> list[str]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            raise RuntimeError(
                "playwright not installed. Run: pip install playwright && playwright install chromium"
            )

        url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=self._headless)
            ctx = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                )
            )
            page = ctx.new_page()

            # Warm up session — NSE requires a real visit before API call
            page.goto("https://www.nseindia.com", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1500)
            page.goto("https://www.nseindia.com/option-chain", wait_until="domcontentloaded", timeout=20000)
            page.wait_for_timeout(1000)

            # Call the API using fetch() inside the browser context (inherits cookies/headers)
            result = page.evaluate(f"""
                async () => {{
                    const r = await fetch('{url}', {{
                        credentials: 'include',
                        headers: {{
                            'Accept': 'application/json, */*',
                            'Referer': 'https://www.nseindia.com/option-chain',
                        }}
                    }});
                    return await r.json();
                }}
            """)
            browser.close()

        expiries: list[str] = result.get("records", {}).get("expiryDates", [])
        logger.info("NSE [%s]: %d expiry dates fetched.", symbol, len(expiries))
        return expiries


# ---------------------------------------------------------------------------
# High-level helper: parse NSE date strings → nearest upcoming date
# ---------------------------------------------------------------------------

def nearest_upcoming(expiry_strings: list[str], reference: datetime.date) -> Optional[datetime.date]:
    """
    Convert NSE date strings ('10-Mar-2026', '17-Mar-2026', …) to date objects
    and return the first one that is >= reference.
    """
    parsed: list[datetime.date] = []
    for s in expiry_strings:
        try:
            parsed.append(datetime.datetime.strptime(s.strip(), "%d-%b-%Y").date())
        except ValueError:
            logger.warning("Could not parse NSE date: %r", s)

    upcoming = [d for d in sorted(parsed) if d >= reference]
    return upcoming[0] if upcoming else None


# ---------------------------------------------------------------------------
# Fetcher implementation compatible with ExpiryFetcherBase contract
# ---------------------------------------------------------------------------

class NseExpiryFetcher:
    """
    Fetches real expiry dates for NSE-listed indices directly from NSE India.
    Returns None for scripts not listed on NSE (e.g. SENSEX → BSE only).

    Usage:
        fetcher = NseExpiryFetcher()
        date = fetcher.get_next_expiry("NIFTY 50", datetime.date.today())
    """

    def __init__(self, session: Optional[NseSessionBase] = None) -> None:
        self._session: NseSessionBase = session or PlaywrightNseSession()

    def get_next_expiry(
        self, script_name: str, reference: datetime.date
    ) -> Optional[datetime.date]:
        """Return the nearest upcoming expiry date for the given script, or None."""
        symbol = NSE_SYMBOLS.get(script_name)
        if not symbol or script_name == "SENSEX":
            logger.info("NSE fetcher: %s is not an NSE index, skipping.", script_name)
            return None

        try:
            expiry_strings = self._session.fetch_expiry_dates(symbol)
            result = nearest_upcoming(expiry_strings, reference)
            if result:
                logger.info("NSE [%s]: next expiry = %s", script_name, result)
            else:
                logger.warning("NSE [%s]: no upcoming expiry found.", script_name)
            return result
        except Exception as exc:
            logger.error("NseExpiryFetcher failed for %s: %s", script_name, exc)
            return None
