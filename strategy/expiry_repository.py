"""
expiry_repository.py

Responsibility: Persist and retrieve ExpiryRecord objects to/from PostgreSQL.
Follows SOLID:
  (S) Single Responsibility : only handles DB I/O for expiry records.
  (O) Open/Closed           : switch DB engine by subclassing ExpiryRepositoryBase.
  (L) Liskov Substitution   : PostgresExpiryRepository is a drop-in for the base contract.
  (I) Interface Segregation : Thin interface — upsert + fetch_latest only.
  (D) Dependency Inversion  : Accepts a SQLAlchemy engine, not a raw connection string.
"""

from __future__ import annotations

import abc
import datetime
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# DDL for the dedicated table – idempotent (CREATE TABLE IF NOT EXISTS)
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS script_expiries (
    id            SERIAL PRIMARY KEY,
    script_name   VARCHAR(32)  NOT NULL,
    expiry_date   DATE         NOT NULL,
    day_label     VARCHAR(4)   NOT NULL,
    source        VARCHAR(16)  NOT NULL DEFAULT 'estimated',
    fetched_at    TIMESTAMP    NOT NULL DEFAULT NOW(),
    UNIQUE (script_name, expiry_date)
);
"""

UPSERT_SQL = """
INSERT INTO script_expiries (script_name, expiry_date, day_label, source, fetched_at)
VALUES (:script_name, :expiry_date, :day_label, :source, NOW())
ON CONFLICT (script_name, expiry_date)
DO UPDATE SET
    day_label  = EXCLUDED.day_label,
    source     = EXCLUDED.source,
    fetched_at = NOW();
"""

FETCH_LATEST_SQL = """
SELECT DISTINCT ON (script_name)
    script_name,
    expiry_date,
    day_label,
    source,
    fetched_at
FROM script_expiries
ORDER BY script_name, fetched_at DESC;
"""


# ---------------------------------------------------------------------------
# (I) Abstract contract
# ---------------------------------------------------------------------------

class ExpiryRepositoryBase(abc.ABC):

    @abc.abstractmethod
    def upsert(self, records: list[dict]) -> int:
        """Persist expiry records. Returns number of rows affected."""
        ...

    @abc.abstractmethod
    def fetch_latest(self) -> list[dict]:
        """Return most-recently saved expiry per script."""
        ...


# ---------------------------------------------------------------------------
# (L / D) Concrete PostgreSQL implementation
# ---------------------------------------------------------------------------

class PostgresExpiryRepository(ExpiryRepositoryBase):
    """Stores expiry records in a PostgreSQL table using an injected engine."""

    def __init__(self, engine) -> None:
        self._engine = engine
        self._ensure_table()

    def _ensure_table(self) -> None:
        """Create the table if it does not already exist (idempotent)."""
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                conn.execute(text(CREATE_TABLE_SQL))
                conn.commit()
            logger.info("script_expiries table ready.")
        except Exception as exc:
            logger.error("Failed to create script_expiries table: %s", exc)
            raise

    def upsert(self, records: list[dict]) -> int:
        """
        Upsert a list of dicts with keys:
            script_name, expiry_date (date|str), day_label, source
        Returns the count of rows inserted/updated.
        """
        if not records:
            return 0
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                result = conn.execute(
                    text(UPSERT_SQL),
                    [
                        {
                            "script_name": r["script_name"],
                            "expiry_date": str(r["expiry_date"]),   # accepts YYYY-MM-DD
                            "day_label":   r["day_label"],
                            "source":      r.get("source", "estimated"),
                        }
                        for r in records
                    ],
                )
                conn.commit()
                count = result.rowcount
            logger.info("Upserted %d expiry record(s) into script_expiries.", count)
            return count
        except Exception as exc:
            logger.error("Failed to upsert expiry records: %s", exc)
            raise

    def fetch_latest(self) -> list[dict]:
        """Return the most-recently stored expiry per script as a list of dicts."""
        try:
            from sqlalchemy import text
            with self._engine.connect() as conn:
                rows = conn.execute(text(FETCH_LATEST_SQL)).mappings().all()
            return [
                {
                    "script_name": row["script_name"],
                    "expiry_date": str(row["expiry_date"]),
                    "day_label":   row["day_label"],
                    "source":      row["source"],
                    "fetched_at":  row["fetched_at"].isoformat() if row["fetched_at"] else None,
                }
                for row in rows
            ]
        except Exception as exc:
            logger.error("Failed to fetch expiry records: %s", exc)
            raise
