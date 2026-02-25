import os
import threading
import time
import logging
import pandas as pd
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv
from sqlalchemy import create_engine
from flask import (
    Flask, render_template, jsonify, request,
    make_response, redirect, url_for, session
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "antigravity-secret-key-change-in-prod")

# ── Auth helpers ─────────────────────────────────────────────────────────────
DASHBOARD_USER = os.getenv("DASHBOARD_USERNAME", "admin")
DASHBOARD_PASS = os.getenv("DASHBOARD_PASSWORD", "antigravity2024")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

# ── Dhan API client ──────────────────────────────────────────────────────────
# Lazy/safe init: app must boot even without keys (Render dashboard-only mode)
CLIENT_ID   = os.getenv("DHAN_CLIENT_ID", "")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")

dhan = None

def init_dhan_local():
    """Method 1: Init Dhan client using personal ACCESS_TOKEN from env (for local use)"""
    global dhan
    try:
        from dhanhq import dhanhq as _DhanHQ, DhanContext as _DhanCtx
        if CLIENT_ID and ACCESS_TOKEN:
            dhan = _DhanHQ(_DhanCtx(str(CLIENT_ID), str(ACCESS_TOKEN)))
            # Update strategy if it's already booted
            if 'strategy' in globals() and strategy is not None:
                strategy.dhan = dhan
            logger.info("Dhan client initialised successfully via Local Access Token.")
        else:
            logger.warning("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set — local trading disabled.")
    except Exception as e:
        logger.warning(f"Dhan client init failed (local method): {e}")

def init_dhan_oauth(token):
    """Method 2: Init Dhan client using Token from OAuth Callback (for deployment)"""
    global dhan
    try:
        from dhanhq import dhanhq as _DhanHQ, DhanContext as _DhanCtx
        if CLIENT_ID and token:
            dhan = _DhanHQ(_DhanCtx(str(CLIENT_ID), str(token)))
            # Update strategy if it's already booted
            if 'strategy' in globals() and strategy is not None:
                strategy.dhan = dhan
            logger.info("Dhan client initialised successfully via OAuth Callback Token.")
    except Exception as e:
        logger.warning(f"Dhan client init failed (OAuth method): {e}")

# Bootup local client initially if token is present
init_dhan_local()

# ── Strategy ─────────────────────────────────────────────────────────────────
strategy = None
try:
    from strategy.v4_trailing_sl_strategy import NiftyV4TrailingSLStrategy
    expiry_date = "2026-03-02"
    strategy = NiftyV4TrailingSLStrategy(expiry_date)
    if dhan:
        strategy.dhan = dhan
    logger.info("Strategy initialised successfully.")
except Exception as e:
    logger.warning(f"Strategy init failed (trading disabled): {e}")
    expiry_date = "2026-03-02"

current_broker  = "Dhan"
strategy_thread = None


def strategy_loop():
    """Background thread that drives the strategy."""
    global expiry_date, strategy
    logger.info(f"Background Strategy Thread Started. Broker: {current_broker}.")
    while strategy and getattr(strategy, 'running', False):
        try:
            if not getattr(strategy, 'paused', False):
                strategy.run_iteration()
        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}")
        for _ in range(60):
            if not strategy or not getattr(strategy, 'running', False):
                break
            time.sleep(1)
    logger.info("Background Strategy Thread Stopped.")


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    error = None
    username = ""
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if username == DASHBOARD_USER and password == DASHBOARD_PASS:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))
        error = "Invalid username or password. Please try again."
    return render_template("login.html", error=error, username=username)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("login_page"))


@app.route('/')
@login_required
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


@app.route('/backtest')
@login_required
def backtest():
    response = make_response(render_template('backtest_runner.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response


# ── Dhan OAuth flow ───────────────────────────────────────────────────────────

@app.route('/dhan/connect')
@login_required
def dhan_connect():
    """Redirect user to Dhan login page to obtain a tokenId."""
    import urllib.parse
    api_key      = os.getenv("DHAN_API_KEY", "")
    redirect_uri = urllib.parse.quote(request.host_url.rstrip('/') + '/dhan/callback', safe='')
    dhan_login_url = (
        f"https://api.dhan.co/partner/oauth/authorize"
        f"?client_id={api_key}&redirect_uri={redirect_uri}&response_type=code"
    )
    return redirect(dhan_login_url)


@app.route('/dhan/callback', methods=['GET', 'POST'])
@login_required
def dhan_callback():
    """
    Dhan redirects here after user login.
    GET  → tokenId arrives as a query param; exchange it for access_token.
    POST → manual exchange form submission.
    """
    from datetime import datetime as _dt
    import requests as _req

    token_id     = request.args.get('tokenId') or request.form.get('token_id', '')
    access_token = request.args.get('access_token', '')   # some flows pass directly
    error_msg    = None

    # If access_token came directly, we're done
    if access_token:
        return render_template(
            'dhan_callback.html',
            status='success',
            access_token=access_token,
            client_id=os.getenv("DHAN_CLIENT_ID", CLIENT_ID),
            connected_at=_dt.now().strftime('%d %b %Y, %I:%M %p'),
        )

    # Exchange tokenId → access_token via Dhan API
    if token_id:
        try:
            resp = _req.post(
                'https://api.dhan.co/partner/oauth/token',
                json={
                    'tokenId':      token_id,
                    'clientId':     os.getenv("DHAN_API_KEY", ""),
                    'clientSecret': os.getenv("DHAN_CLIENT_SECRET", ""),
                },
                timeout=10,
            )
            data = resp.json()
            if resp.ok and data.get('access_token'):
                return render_template(
                    'dhan_callback.html',
                    status='success',
                    access_token=data['access_token'],
                    client_id=data.get('dhanClientId', os.getenv("DHAN_CLIENT_ID", CLIENT_ID)),
                    connected_at=_dt.now().strftime('%d %b %Y, %I:%M %p'),
                )
            else:
                # Show pending UI so user can try manually
                return render_template(
                    'dhan_callback.html',
                    status='pending',
                    token_id=token_id,
                )
        except Exception as e:
            error_msg = str(e)

    return render_template(
        'dhan_callback.html',
        status='error',
        error_message=error_msg or "No tokenId or access_token received from Dhan.",
    )


@app.route('/dhan/save-token', methods=['POST'])
@login_required
def dhan_save_token():
    """Hot-reload the Dhan client with a new access token (no restart needed)."""
    global dhan
    data = request.get_json(force=True)
    new_token = data.get('access_token', '').strip()
    if not new_token:
        return jsonify({"status": "error", "message": "No token provided"}), 400
    try:
        init_dhan_oauth(new_token)
        return jsonify({"status": "success", "message": "Token saved and client reloaded."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "broker":        current_broker,
        "running":       getattr(strategy, 'running', False)      if strategy else False,
        "paused":        getattr(strategy, 'paused',  False)      if strategy else False,
        "in_position":   getattr(strategy, 'in_position', False)  if strategy else False,
        "unrealized_pnl":getattr(strategy, 'unrealized_pnl', 0)  if strategy else 0,
        "realized_pnl":  getattr(strategy, 'realized_pnl',   0)  if strategy else 0,
    })


@app.route('/api/start', methods=['POST'])
def start():
    global strategy_thread
    if not strategy:
        return jsonify({"status": "error", "message": "Strategy not initialised"}), 503
    if strategy_thread is None or not strategy_thread.is_alive():
        strategy.running = True
        strategy.paused  = False
        strategy_thread  = threading.Thread(target=strategy_loop, daemon=True)
        strategy_thread.start()
        return jsonify({"status": "success", "message": "Strategy Started"})
    return jsonify({"status": "error", "message": "Strategy already running"}), 400


@app.route('/api/pause', methods=['POST'])
def pause():
    if not strategy:
        return jsonify({"status": "error", "message": "Strategy not initialised"}), 503
    strategy.paused = True
    return jsonify({"status": "success", "message": "Strategy Paused"})


@app.route('/api/resume', methods=['POST'])
def resume():
    if not strategy:
        return jsonify({"status": "error", "message": "Strategy not initialised"}), 503
    strategy.paused = False
    return jsonify({"status": "success", "message": "Strategy Resumed"})


@app.route('/api/stop', methods=['POST'])
def stop_bot():
    if not strategy:
        return jsonify({"status": "error", "message": "Strategy not initialised"}), 503
    strategy.running = False
    return jsonify({"status": "success", "message": "Strategy Stopped"})


@app.route('/api/close_all_positions', methods=['POST'])
def close_all_positions():
    if not dhan:
        return jsonify({"status": "error", "message": "Dhan client not initialised"}), 503
    try:
        dhan.cancel_all_orders()
        pos_resp = dhan.get_positions()
        if pos_resp.get("status") == "success":
            for p in pos_resp.get("data", []):
                if int(p.get("netQty", 0)) != 0:
                    dhan.place_order(
                        security_id=p["securityId"],
                        exchange_segment=p["exchangeSegment"],
                        transaction_type=dhan.SELL if int(p["netQty"]) > 0 else dhan.BUY,
                        quantity=abs(int(p["netQty"])),
                        order_type=dhan.MARKET,
                        product_type=p["productType"],
                    )
        if strategy:
            strategy.running    = False
            strategy.in_position = False
        return jsonify({"status": "success", "message": "All orders cancelled and positions closed."})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/backtests', methods=['GET'])
def get_backtests():
    uri = os.getenv("POSTGRES_URI", "")
    if not uri:
        return jsonify({"status": "error", "message": "POSTGRES_URI not configured"}), 503
    try:
        engine = create_engine(uri)
        df = pd.read_sql(
            'SELECT * FROM historical_backtests ORDER BY "Date" DESC, "Entry_Time" DESC',
            con=engine,
        )
        return jsonify({"status": "success", "data": df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/run-backtest', methods=['POST'])
@login_required
def run_backtest():
    """Acts as a Quick Dhan API Tester for the UI."""
    data = request.json or {}
    strategy = data.get('strategy', 'Unknown')
    
    if not dhan:
        return jsonify({"status": "error", "message": "Dhan API not connected. Please connect via settings/env."}), 400
    
    try:
        # Just to verify connection
        res = dhan.get_fund_limits()
        if res.get("status") == "success":
            return jsonify({
                "status": "success",
                "message": "Dhan API is actively connected and responding successfully!",
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "net_pnl": 0,
                "capital": data.get('capital', 500000)
            })
        else:
            return jsonify({"status": "error", "message": f"Dhan API Verification Failed: {res}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500



@app.route('/api/test-strategy', methods=['POST'])
def test_strategy():
    data = request.json or {}
    strategy = data.get('strategy', 'ALL')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    # Mapping frontend strategy keys to python files
    scripts = []
    if strategy == "ALL":
        scripts = ["backtest_dhan_5min.py", "backtest_gamma.py", "backtest_v4.py"]
    elif strategy == "v4_trailing_sl":
        scripts = ["backtest_v4.py"]
    elif strategy == "gamma_spike":
        scripts = ["backtest_gamma.py"]
    else:
        # DB strategies like V5_IV15_48W can be covered by the main dhan backtest 
        scripts = ["backtest_dhan_5min.py"]

    import subprocess
    env = os.environ.copy()
    if start_date:
        env['BACKTEST_START_DATE'] = start_date
    if end_date:
        env['BACKTEST_END_DATE'] = end_date

    count = 0
    for script in scripts:
        if os.path.exists(script):
            subprocess.Popen(["python", script], env=env)
            count += 1

    if count == 0:
        return jsonify({"status": "error", "message": f"No backtest scripts found for {strategy}"}), 404

    return jsonify({"status": "success", "message": f"Started {count} background job(s) for testing strategy: {strategy}."})


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
