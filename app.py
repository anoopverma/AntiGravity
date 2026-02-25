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
CLIENT_ID   = os.getenv("DHAN_CLIENT_ID") or ""
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN") or ""

dhan = None
ACTIVE_TOKEN = ACCESS_TOKEN  # Track the current active token for subprocesses

def init_dhan():
    """Init Dhan client using DHAN_ACCESS_TOKEN from env (Local .env or Render Env Variables)"""
    global dhan
    try:
        from dhanhq import dhanhq as _DhanHQ
        if CLIENT_ID and ACCESS_TOKEN:
            dhan = _DhanHQ(str(CLIENT_ID), str(ACCESS_TOKEN))
            # Update active strategies if already booted
            if 'active_strategies' in globals() and active_strategies is not None:
                for strat in active_strategies:
                    strat.dhan = dhan
            logger.info("Dhan client initialised successfully via Environment Access Token.")
        else:
            logger.warning("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set — local trading disabled.")
    except Exception as e:
        logger.warning(f"Dhan client init failed: {e}")

# Bootup local client initially if token is present
init_dhan()

# ── Engine State ─────────────────────────────────────────────────────────────
active_strategies = []
running_flag = False
paused_flag = False
current_broker = "Dhan"
strategy_thread = None
expiry_date = "2026-03-02"

logger.info("Engine configured. Standing by for start.")


def strategy_loop():
    """Background thread that drives active strategies."""
    global expiry_date, active_strategies, running_flag, paused_flag, current_broker
    logger.info(f"Background Strategy Thread Started. Broker: {current_broker}.")
    
    while running_flag:
        now_ist = pd.Timestamp.now('Asia/Kolkata')
        if now_ist.hour == 15 and now_ist.minute == 31:
            logger.info("Auto-stopping engines at 3:31 PM IST.")
            running_flag = False
            paused_flag = False
            active_strategies.clear()
            break

        try:
            if not paused_flag:
                for strat in active_strategies:
                    # check if the strategy is individually paused
                    if getattr(strat, '_is_paused', False):
                        continue
                        
                    # Depending on strategy class, call run_iteration appropriately
                    if hasattr(strat, 'run_iteration'):
                        import inspect
                        sig = inspect.signature(strat.run_iteration)
                        if len(sig.parameters) > 0:
                            strat.run_iteration(expiry_date)
                        else:
                            strat.run_iteration()
        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}")
            
        # Tick for 60 seconds unless stopped
        for _ in range(60):
            if not running_flag:
                break
                
            now_ist = pd.Timestamp.now('Asia/Kolkata')
            if now_ist.hour == 15 and now_ist.minute == 31:
                logger.info("Auto-stopping engines at 3:31 PM IST.")
                running_flag = False
                paused_flag = False
                active_strategies.clear()
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



@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    global running_flag, paused_flag, active_strategies
    
    in_pos = any(getattr(s, 'in_position', False) for s in active_strategies) if active_strategies else False
    u_pnl = sum(getattr(s, 'unrealized_pnl', 0) for s in active_strategies) if active_strategies else 0
    r_pnl = sum(getattr(s, 'realized_pnl', 0) for s in active_strategies) if active_strategies else 0
    
    names = []
    for s in active_strategies:
        base_name = s.__class__.__name__.replace("Nifty", "").replace("Strategy", "")
        mode = "P" if getattr(s, "paper_trade", False) else "L"
        state = "[PAUSED]" if getattr(s, "_is_paused", False) else ""
        names.append(f"{base_name}[{mode}]{state}")
    
    return jsonify({
        "broker":        current_broker,
        "running":       running_flag,
        "paused":        paused_flag,
        "in_position":   in_pos,
        "unrealized_pnl": u_pnl,
        "realized_pnl":  r_pnl,
        "active_names":  ", ".join(names) if names else ""
    })


@app.route('/api/start', methods=['POST'])
def start():
    global strategy_thread, running_flag, paused_flag, active_strategies, expiry_date
    data = request.get_json(silent=True) or {}
    live_strategies = data.get('live', [])
    paper_strategies = data.get('paper', [])
    
    if not dhan:
        return jsonify({"status": "error", "message": "Dhan not initialised"}), 503
        
    if not live_strategies and not paper_strategies:
        return jsonify({"status": "error", "message": "No strategies selected"}), 400
        
    if strategy_thread is None or not strategy_thread.is_alive() or not running_flag:
        # Initial boot
        loaded_names = []
        active_strategies.clear()
        
        # Instantiate requested strategies
        def load_strategy(strat_id, is_paper):
            if strat_id == "v4_gamma":
                try:
                    from strategy.v4_trailing_sl_strategy import NiftyV4TrailingSLStrategy
                    s1 = NiftyV4TrailingSLStrategy(expiry_date)
                    s1.dhan = dhan
                    s1.paper_trade = is_paper
                    active_strategies.append(s1)
                    loaded_names.append(f"V4[{'P' if is_paper else 'L'}]")
                except Exception as e:
                    logger.error(f"Failed to load V4: {e}")
                    
            elif strat_id == "gamma_blast":
                try:
                    from strategy.gamma_spike_strategy import NiftyGammaSpikeStrategy
                    s2 = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
                    s2.dhan = dhan
                    s2.paper_trade = is_paper
                    active_strategies.append(s2)
                    loaded_names.append(f"GammaBlast[{'P' if is_paper else 'L'}]")
                except Exception as e:
                    logger.error(f"Failed to load Gamma Blast: {e}")
                    
        for s in live_strategies:
            load_strategy(s, False)
        for s in paper_strategies:
            load_strategy(s, True)
        
        if not active_strategies:
            return jsonify({"status": "error", "message": "Failed to load instances"}), 500

        running_flag = True
        paused_flag = False
        
        logger.info(f"Starting Engine with Live: {live_strategies} | Paper: {paper_strategies}")
        strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
        strategy_thread.start()
        
        names_str = ", ".join(loaded_names)
        return jsonify({"status": "success", "message": f"Strategy Engine Started [{names_str}]"})
        
    else:
        # Engine is already running. We dynamically add the new strategies if not already present.
        loaded_names = []
        already_loaded = []
        
        def is_already_loaded(strat_id, is_paper):
            expected_class = "NiftyV4TrailingSLStrategy" if strat_id == "v4_gamma" else "NiftyGammaSpikeStrategy"
            for s in active_strategies:
                if s.__class__.__name__ == expected_class and getattr(s, 'paper_trade', False) == is_paper:
                    return True
            return False

        def append_strategy(strat_id, is_paper):
            if is_already_loaded(strat_id, is_paper):
                mode = 'P' if is_paper else 'L'
                name = "V4" if strat_id == "v4_gamma" else "GammaBlast"
                already_loaded.append(f"{name}[{mode}]")
                return

            if strat_id == "v4_gamma":
                try:
                    from strategy.v4_trailing_sl_strategy import NiftyV4TrailingSLStrategy
                    s1 = NiftyV4TrailingSLStrategy(expiry_date)
                    s1.dhan = dhan
                    s1.paper_trade = is_paper
                    active_strategies.append(s1)
                    loaded_names.append(f"V4[{'P' if is_paper else 'L'}]")
                except Exception as e:
                    logger.error(f"Failed to load V4: {e}")
                    
            elif strat_id == "gamma_blast":
                try:
                    from strategy.gamma_spike_strategy import NiftyGammaSpikeStrategy
                    s2 = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
                    s2.dhan = dhan
                    s2.paper_trade = is_paper
                    active_strategies.append(s2)
                    loaded_names.append(f"GammaBlast[{'P' if is_paper else 'L'}]")
                except Exception as e:
                    logger.error(f"Failed to load Gamma Blast: {e}")

        for s in live_strategies:
            append_strategy(s, False)
        for s in paper_strategies:
            append_strategy(s, True)

        if not loaded_names and already_loaded:
            return jsonify({"status": "error", "message": f"Strategies ({', '.join(already_loaded)}) are already running!"}), 400
        elif not loaded_names:
            return jsonify({"status": "error", "message": "Failed to load instances"}), 500

        logger.info(f"Added strategies Live: {live_strategies} | Paper: {paper_strategies} to running Engine.")
        names_str = ", ".join(loaded_names)
        return jsonify({"status": "success", "message": f"Added strategies [{names_str}] to existing Engine"})



@app.route('/api/pause', methods=['POST'])
def pause():
    global paused_flag, running_flag, active_strategies
    if not running_flag:
        return jsonify({"status": "error", "message": "Engine is not running"}), 503
    
    data = request.get_json(silent=True) or {}
    target = data.get('target', 'all')
    
    effect_count = 0
    if target == 'all':
        paused_flag = True
        return jsonify({"status": "success", "message": "Strategy Engine Paused (ALL)"})
    else:
        # Pause specific
        is_paper_target = (target == 'paper')
        for strat in active_strategies:
            if getattr(strat, 'paper_trade', False) == is_paper_target:
                strat._is_paused = True # mark internally
                effect_count += 1
        return jsonify({"status": "success", "message": f"Paused {target.upper()} strategies"})


@app.route('/api/resume', methods=['POST'])
def resume():
    global paused_flag, running_flag, active_strategies
    if not running_flag:
        return jsonify({"status": "error", "message": "Engine is not running"}), 503

    data = request.get_json(silent=True) or {}
    target = data.get('target', 'all')
    
    if target == 'all':
        paused_flag = False
        for strat in active_strategies:
            strat._is_paused = False
        return jsonify({"status": "success", "message": "Strategy Engine Resumed (ALL)"})
    else:
        is_paper_target = (target == 'paper')
        for strat in active_strategies:
            if getattr(strat, 'paper_trade', False) == is_paper_target:
                strat._is_paused = False
        
        # If any strategies remain unpaused, ensure global is False
        if any(not getattr(s, '_is_paused', False) for s in active_strategies):
            paused_flag = False
        
        return jsonify({"status": "success", "message": f"Resumed {target.upper()} strategies"})


@app.route('/api/stop', methods=['POST'])
def stop_bot():
    global running_flag, paused_flag, active_strategies
    if not running_flag:
        return jsonify({"status": "error", "message": "Engine is already stopped"}), 400
        
    data = request.get_json(silent=True) or {}
    target = data.get('target', 'all')
    
    if target == 'all':
        running_flag = False
        paused_flag = False
        active_strategies = []
        logger.info("Cleared state: ALL Live and Paper strategies stopped.")
        return jsonify({"status": "success", "message": "All BOTS Stopped Completely"})
    else:
        is_paper_target = (target == 'paper')
        logger.info(f"Cleared state: Stopping {'Paper' if is_paper_target else 'Live'} strategies.")
        # Retain strategies that do NOT belong to the targeted section
        active_strategies = [s for s in active_strategies if getattr(s, 'paper_trade', False) != is_paper_target]
        
        if not active_strategies:
            running_flag = False
            paused_flag = False
            return jsonify({"status": "success", "message": f"Stopped {target.upper()} strategies. Engine stopped as no strategies remain."})
            
        return jsonify({"status": "success", "message": f"Stopped {target.upper()} strategies"})




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
        global running_flag
        running_flag = False
        for strat in active_strategies:
            strat.in_position = False
            strat.current_position = None
            strat.unrealized_pnl = 0
            
        return jsonify({"status": "success", "message": "All orders cancelled and positions closed."})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/positions', methods=['GET'])
@login_required
def get_positions():
    if not dhan:
        return jsonify({"status": "error", "message": "Dhan client not initialised"}), 503
    try:
        pos_resp = dhan.get_positions()
        positions = []
        if pos_resp.get("status") == "success":
            for p in pos_resp.get("data", []):
                net_qty = int(p.get("netQty", 0))
                if net_qty == 0:
                    continue
                buy_avg  = float(p.get("buyAvg",  0) or 0)
                sell_avg = float(p.get("sellAvg", 0) or 0)
                ltp      = float(p.get("ltp",     0) or 0)
                unrealized = (ltp - buy_avg) * net_qty if net_qty > 0 else (sell_avg - ltp) * abs(net_qty)
                positions.append({
                    "script":    p.get("tradingSymbol", p.get("securityId", "Unknown")),
                    "direction": "BUY" if net_qty > 0 else "SELL",
                    "qty":       abs(net_qty),
                    "avg_price": buy_avg if net_qty > 0 else sell_avg,
                    "ltp":       ltp,
                    "unrealized_pnl": round(unrealized, 2),
                })
        return jsonify({"status": "success", "data": positions})
    except Exception as e:
        logger.error(f"Positions error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/backtests', methods=['GET'])
@login_required
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



@app.route('/api/delete-backtest', methods=['POST'])
@login_required
def delete_backtest():
    """Delete database records for a specified strategy and date range."""
    data = request.json or {}
    strategy = data.get('strategy', 'ALL')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    uri = os.getenv("POSTGRES_URI", "")
    if not uri:
        return jsonify({"status": "error", "message": "POSTGRES_URI not configured"}), 503

    try:
        from sqlalchemy import text
        engine = create_engine(uri)
        with engine.connect() as conn:
            query = 'DELETE FROM historical_backtests WHERE 1=1'
            params = {}

            if strategy != 'ALL':
                query += ' AND "Strategy_Name" = :strategy'
                params['strategy'] = strategy
            
            if start_date:
                query += ' AND "Date" >= :start_date'
                params['start_date'] = start_date

            if end_date:
                query += ' AND "Date" <= :end_date'
                params['end_date'] = end_date

            result = conn.execute(text(query), params)
            conn.commit()

            return jsonify({
                "status": "success", 
                "message": f"Deleted {result.rowcount} record(s) from database."
            })
    except Exception as e:
        logger.error(f"Error deleting backtest data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/test-strategy', methods=['POST'])
def test_strategy():
    data = request.json or {}
    strategy = data.get('strategy', 'ALL')
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    # Always use the main dhan script because it has the DB saving logic
    scripts = ["backtest_dhan_5min.py"]

    import subprocess
    env = os.environ.copy()
    env["RUN_BACKTEST_STRATEGY"] = strategy
    
    # Inject active credentials into subprocess environment
    if CLIENT_ID:
        env['DHAN_CLIENT_ID'] = str(CLIENT_ID)
    if ACTIVE_TOKEN:
        env['DHAN_ACCESS_TOKEN'] = str(ACTIVE_TOKEN)

    if start_date:
        env['BACKTEST_START_DATE'] = start_date
    if end_date:
        env['BACKTEST_END_DATE'] = end_date

    count = 0
    for script in scripts:
        if os.path.exists(script):
            try:
                subprocess.run(["python", script], env=env, check=True)
                count += 1
            except subprocess.CalledProcessError as e:
                return jsonify({"status": "error", "message": f"Error running {script}: {e}"}), 500

    if count == 0:
        return jsonify({"status": "error", "message": f"No backtest scripts found for {strategy}"}), 404

    return jsonify({"status": "success", "message": f"Completed {count} backtest(s) for strategy: {strategy}. Data saved to database."})


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
