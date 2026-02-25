import os
from sqlalchemy import create_engine
from sqlalchemy import text
from dotenv import load_dotenv

load_dotenv()
uri = os.getenv("POSTGRES_URI")
if uri:
    engine = create_engine(uri)
    with engine.connect() as conn:
        conn.execute(text("DELETE FROM historical_backtests WHERE \"Strategy_Name\" NOT IN ('v4_trailing_sl', 'gamma_spike');"))
        conn.commit()
    print("Postgres table cleaned. Only mapped strategies remain!")
else:
    print("No POSTGRES_URI found.")
