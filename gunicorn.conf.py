import os

# Bind to the PORT env var injected by Render (defaults to 5002 locally)
bind = f"0.0.0.0:{os.environ.get('PORT', '5002')}"

# Workers: 2 is enough for a single-user dashboard on Render free tier
workers = 2

# Threads per worker for handling concurrent requests
threads = 4

# Timeout: backtest calls can take a while — give them 2 minutes
timeout = 120

# Keep-alive for persistent connections
keepalive = 5

# Log level
loglevel = "info"

# Access log to stdout so Render captures it
accesslog = "-"
errorlog  = "-"
