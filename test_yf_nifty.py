import yfinance as yf
print("Testing Yahoo Finance 5-min intervals")
nifty = yf.Ticker('^NSEI')
# Fetch 5-min data for the last 60 days (max allowed by YF)
df = nifty.history(period="60d", interval="5m")
print(f"Returned {len(df)} rows")
if not df.empty:
    print(df.head(2))
    print(df.tail(2))
