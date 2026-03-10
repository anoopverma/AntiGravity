"""
expiry_populator.py

Populates script_expiries table with future expiries for the next ~4 months.
Follows user rules:
- NIFTY 50: Weekly Tue, holiday -> previous trading day
- SENSEX: Weekly Thu, holiday -> previous trading day
- BANKNIFTY: Last Tue of Month, holiday -> previous trading day
- FINNIFTY: Last Tue of Month, holiday -> previous trading day

Holidays 2026:
- Mar 26 (Thu), Mar 31 (Tue)
- Apr 03 (Fri), Apr 14 (Tue)
- May 01 (Fri), May 28 (Thu)
- Jun 26 (Fri)
+ Sat/Sun
"""
import datetime
import calendar
import os
import logging
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

HOLIDAYS_2026 = {
    datetime.date(2026, 3, 26),
    datetime.date(2026, 3, 31),
    datetime.date(2026, 4, 3),
    datetime.date(2026, 4, 14),
    datetime.date(2026, 5, 1),
    datetime.date(2026, 5, 28),
    datetime.date(2026, 6, 26),
}

def is_trading_holiday(date: datetime.date) -> bool:
    # Weekend or declared holiday
    return date.weekday() >= 5 or date in HOLIDAYS_2026

def get_previous_trading_day(date: datetime.date) -> datetime.date:
    temp = date
    while is_trading_holiday(temp):
        temp -= datetime.timedelta(days=1)
    return temp

def get_last_weekday_of_month(year: int, month: int, weekday: int) -> datetime.date:
    last_day = calendar.monthrange(year, month)[1]
    last_date = datetime.date(year, month, last_day)
    days_back = (last_date.weekday() - weekday) % 7
    return last_date - datetime.timedelta(days=days_back)

def populate():
    uri = os.getenv("POSTGRES_URI")
    if not uri:
        logger.error("POSTGRES_URI not found in env.")
        return

    engine = create_engine(uri)
    
    # 1. Define Scripts
    # NIFTY 50: Weekly Tue (1)
    # SENSEX: Weekly Thu (3)
    # BANKNIFTY: Monthly Last Tue (1)
    # FINNIFTY: Monthly Last Tue (1)
    
    start_date = datetime.date.today()
    end_date = datetime.date(2026, 6, 30)
    
    records = []
    
    # --- Weekly Scripts (Nifty & Sensex) ---
    curr = start_date
    while curr <= end_date:
        # NIFTY 50 (Weekly Tue)
        if curr.weekday() == 1:
            expiry = get_previous_trading_day(curr)
            records.append({
                "script_name": "NIFTY 50",
                "expiry_date": expiry,
                "day_label": expiry.strftime("%a"),
                "source": "bulk_insert"
            })
        # SENSEX (Weekly Thu)
        if curr.weekday() == 3:
            expiry = get_previous_trading_day(curr)
            records.append({
                "script_name": "SENSEX",
                "expiry_date": expiry,
                "day_label": expiry.strftime("%a"),
                "source": "bulk_insert"
            })
        curr += datetime.timedelta(days=1)
        
    # --- Monthly Scripts (BankNifty & FinNifty) ---
    for year in [2026]:
        for month in [3, 4, 5, 6]:
            # Last Tue (1)
            target = get_last_weekday_of_month(year, month, 1)
            # If target has passed since start_date or is in month >= start, add it
            if target >= start_date:
                expiry = get_previous_trading_day(target)
                for script in ["BANKNIFTY", "FINNIFTY"]:
                    records.append({
                        "script_name": script,
                        "expiry_date": expiry,
                        "day_label": expiry.strftime("%a"),
                        "source": "bulk_insert"
                    })
    
    # Upsert logic to match the existing schema
    UPSERT_SQL = """
    INSERT INTO script_expiries (script_name, expiry_date, day_label, source, fetched_at)
    VALUES (:script_name, :expiry_date, :day_label, :source, NOW())
    ON CONFLICT (script_name, expiry_date)
    DO UPDATE SET
        day_label  = EXCLUDED.day_label,
        source     = EXCLUDED.source,
        fetched_at = NOW();
    """
    
    with engine.connect() as conn:
        logger.info(f"Inserting/Updating {len(records)} expiry records...")
        for r in records:
            conn.execute(text(UPSERT_SQL), r)
        conn.commit()
    
    logger.info("Done.")

if __name__ == "__main__":
    populate()
