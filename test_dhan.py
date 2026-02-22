import os
import json
from dotenv import load_dotenv
from dhanhq import dhanhq, DhanContext

load_dotenv()
dhan = dhanhq(DhanContext(os.getenv('DHAN_CLIENT_ID'), os.getenv('DHAN_ACCESS_TOKEN')))

print("Testing NIFTY Options Data for Expiry")
# Let's try to get historical quote for a specific Nifty Call Option
# Nifty 12 Feb 2026 CE 22000 (We need the correct security ID, usually obtainable from scrip master)
req = dhan.intraday_minute_data(
    security_id='35002', # Example ID
    exchange_segment=dhan.NSE_FNO,
    instrument_type='OPTIDX',
    from_date='2026-02-12',
    to_date='2026-02-12'
)
print(req)
