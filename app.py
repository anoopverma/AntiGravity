import os
import threading
import time
import logging
import pandas as pd
from datetime import datetime
from functools import wraps
from dotenv import load_dotenv
from flask import (
    Flask, render_template, jsonify, request,
    make_response, redirect, url_for, session
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("server_log.txt"),
        logging.StreamHandler()
    ]
)
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
        from dhanhq.dhan_context import DhanContext
        if CLIENT_ID and ACCESS_TOKEN:
            context = DhanContext(str(CLIENT_ID), str(ACCESS_TOKEN))
            dhan = _DhanHQ(context)
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
expiry_date = "2026-03-24" # Default fallback
try:
    from strategy.expiry_manager import ExpiryManager
    if dhan:
        exp_records = ExpiryManager(dhan_client=dhan).get_upcoming_expiries()
        nifty_rec = next((r for r in exp_records if r.script == "NIFTY 50"), None)
        if nifty_rec:
            expiry_date = nifty_rec.expiry_date.strftime("%Y-%m-%d")
            logger.info(f"Auto-detected next Nifty expiry: {expiry_date}")
except Exception as e:
    logger.warning(f"Failed to auto-detect expiry: {e}. Using fallback {expiry_date}")

logger.info("Engine configured. Standing by for start.")


def strategy_loop():
    """Background thread that drives active strategies."""
    global expiry_date, active_strategies, running_flag, paused_flag, current_broker
    logger.info(f"Background Strategy Thread Started. Broker: {current_broker}.")
    
    import datetime
    while running_flag:
        try:
            # Simple IST check without pandas overhead
            now_ist = datetime.datetime.now() # assume OS is IST, or just check hour/min
            if now_ist.hour == 15 and now_ist.minute == 31:
                logger.info("Auto-stopping engines at 3:31 PM IST.")
                running_flag = False
                paused_flag = False
                active_strategies.clear()
                break

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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200

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



@app.route('/api/system_logs', methods=['GET'])
@login_required
def get_system_logs():
    try:
        with open('server_log.txt', 'r') as f:
            lines = f.readlines()
            
        # Filter out werkzeug HTTP request logs
        filtered_lines = [line for line in lines if '"GET /' not in line and '"POST /' not in line]
        return jsonify({"status": "success", "logs": filtered_lines[-100:]})
    except Exception as e:
        return jsonify({"status": "error", "message": "No logs found"})

@app.route('/api/clear_logs', methods=['POST'])
@login_required
def clear_logs():
    try:
        open('server_log.txt', 'w').close()   # truncate the file
        logger.info("🗑 System logs cleared by user.")
        return jsonify({"status": "success", "message": "Logs cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

@app.route('/api/expiries', methods=['GET'])
@login_required
def get_expiries():
    from strategy.expiry_manager import ExpiryManager
    try:
        records = ExpiryManager(dhan_client=dhan).get_upcoming_expiries()
        data    = [r.to_dict() for r in records]
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Failed to calculate expiries: {e}")
        return jsonify({"status": "error", "message": str(e)})


def _build_expiry_calendar(days_ahead=120):
    """Build a forward expiry calendar without database persistence."""
    import datetime as dt
    from strategy.expiry_manager import ExpiryManager

    manager = ExpiryManager(dhan_client=dhan)
    today = dt.date.today()
    seen = set()
    rows = []

    for offset in range(days_ahead + 1):
        ref_date = today + dt.timedelta(days=offset)
        for rec in manager.get_upcoming_expiries(reference=ref_date):
            key = (rec.script, rec.expiry_date.isoformat())
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec.to_dict())

    rows.sort(key=lambda r: (r.get("expiry_iso", ""), r.get("script", "")))
    return rows


@app.route('/api/expiries/history', methods=['GET'])
@login_required
def get_expiries_history():
    """Backward-compatible endpoint returning generated expiry calendar."""
    try:
        return jsonify({"status": "success", "data": _build_expiry_calendar()})
    except Exception as e:
        logger.error(f"Failed to read expiry history: {e}")
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/expiries/all', methods=['GET'])
@login_required
def get_expiries_all():
    """Return all upcoming expiry records for the next few months."""
    try:
        data = _build_expiry_calendar()
        return jsonify({"status": "success", "data": data})
    except Exception as e:
        logger.error(f"Failed to read expiry list: {e}")
        return jsonify({"status": "error", "message": str(e)})

@app.route('/expiries')
@login_required
def expiries_page():
    response = make_response(render_template('expiry.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response



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
    # Ensure overrides is a dict or None
    raw_overrides = data.get('overrides')
    overrides = raw_overrides if isinstance(raw_overrides, dict) else None
    
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
                    active_expiry = expiry_date
                    if overrides and overrides.get('expiry'):
                        active_expiry = overrides.get('expiry')
                        
                    s1 = NiftyV4TrailingSLStrategy(active_expiry)
                    s1.dhan = dhan
                    s1.paper_trade = is_paper
                    
                    if overrides:
                        if overrides.get('qty'): 
                            try: s1.lot_size = int(overrides.get('qty'))
                            except: pass
                        if overrides.get('index_id'): s1.index_id = str(overrides.get('index_id'))
                        if overrides.get('base_time'): s1.manual_base_time = str(overrides.get('base_time'))
                        s1.force_run = True

                    active_strategies.append(s1)
                    mode_label = 'P' if is_paper else 'L'
                    loaded_names.append(f"V4[{mode_label}]")
                except Exception as e:
                    logger.error(f"Failed to load V4: {e}")
                    
            elif strat_id == "gamma_blast":
                try:
                    from strategy.gamma_spike_strategy import NiftyGammaSpikeStrategy
                    s2 = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
                    s2.paper_trade = is_paper

                    # Apply overrides — same pattern as v4_gamma
                    if overrides and overrides.get('expiry'):
                        s2.target_expiry = overrides.get('expiry')
                    else:
                        s2.target_expiry = expiry_date

                    if overrides:
                        if overrides.get('qty'):
                            try: s2.lot_size = int(overrides.get('qty'))
                            except: pass
                        if overrides.get('index_id'): s2.index_id = int(overrides.get('index_id'))
                        if overrides.get('base_time'):
                            s2.benchmark_hour, s2.benchmark_min = map(int, overrides.get('base_time').split(':'))
                        s2.force_run = True   # run on any day/time when overridden

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
                    active_expiry = expiry_date
                    if overrides and overrides.get('expiry'):
                        active_expiry = overrides.get('expiry')
                        
                    s1 = NiftyV4TrailingSLStrategy(active_expiry)
                    s1.dhan = dhan
                    s1.paper_trade = is_paper
                    
                    if overrides:
                        if overrides.get('qty'): 
                            try: s1.lot_size = int(overrides.get('qty'))
                            except: pass
                        if overrides.get('index_id'): s1.index_id = str(overrides.get('index_id'))
                        if overrides.get('base_time'): s1.manual_base_time = str(overrides.get('base_time'))
                        s1.force_run = True

                    active_strategies.append(s1)
                    mode_label = 'P' if is_paper else 'L'
                    loaded_names.append(f"V4[{mode_label}]")
                except Exception as e:
                    logger.error(f"Failed to load V4: {e}")
                    
            elif strat_id == "gamma_blast":
                try:
                    from strategy.gamma_spike_strategy import NiftyGammaSpikeStrategy
                    s2 = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
                    s2.paper_trade = is_paper

                    if overrides and overrides.get('expiry'):
                        s2.target_expiry = overrides.get('expiry')
                    else:
                        s2.target_expiry = expiry_date

                    if overrides:
                        if overrides.get('qty'):
                            try: s2.lot_size = int(overrides.get('qty'))
                            except: pass
                        if overrides.get('index_id'): s2.index_id = int(overrides.get('index_id'))
                        if overrides.get('base_time'):
                            s2.benchmark_hour, s2.benchmark_min = map(int, overrides.get('base_time').split(':'))
                        s2.force_run = True

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
    try:
        from simple_salesforce import Salesforce

        sf_instance = (os.getenv("SF_INSTANCE") or "").strip()
        if sf_instance.startswith("https://"):
            sf_instance = sf_instance[len("https://"):]
        elif sf_instance.startswith("http://"):
            sf_instance = sf_instance[len("http://"):]
        sf_instance = sf_instance.rstrip("/")

        sf_kwargs = {
            "username": os.getenv("SF_USERNAME"),
            "password": os.getenv("SF_PASSWORD"),
            "security_token": os.getenv("SF_SECURITY_TOKEN", ""),
            "version": os.getenv("SF_API_VERSION", "59.0"),
        }
        if sf_instance:
            sf_kwargs["instance"] = sf_instance
        else:
            sf_kwargs["domain"] = os.getenv("SF_DOMAIN", "login")

        sf = Salesforce(**sf_kwargs)

        query = """
            SELECT Run_Date__c,
                   Run_Mode__c,
                   Strategy_Name__c,
                   Trade_Date__c,
                   Strike__c,
                   Option_Type__c,
                   Action__c,
                   Qty__c,
                   Entry_Time__c,
                   Exit_Time__c,
                   Buy_Price__c,
                   Peak_Price__c,
                   Sell_Price__c,
                   Total_PNL__c,
                   Total_Return_Percentage__c,
                   Capital_ROI_Pct__c,
                   Reason__c,
                   Parameters__c
            FROM historical_backtests__c
            ORDER BY Trade_Date__c DESC, Entry_Time__c DESC
            LIMIT 5000
        """

        results = sf.query_all(query)

        def _fmt_time(v):
            if not v:
                return "-"
            s = str(v)
            if "T" in s:
                return s.split("T", 1)[1][:8]
            return s

        rows = []
        for rec in results.get("records", []):
            rows.append({
                "Run_Date": rec.get("Run_Date__c"),
                "Run_Mode": rec.get("Run_Mode__c") or "backtest",
                "Strategy_Name": rec.get("Strategy_Name__c"),
                "Date": rec.get("Trade_Date__c"),
                "Strike": rec.get("Strike__c"),
                "Option_Type": rec.get("Option_Type__c"),
                "Action": rec.get("Action__c"),
                "Qty": rec.get("Qty__c"),
                "Entry_Time": _fmt_time(rec.get("Entry_Time__c")),
                "Exit_Time": _fmt_time(rec.get("Exit_Time__c")),
                "Buy_Price": rec.get("Buy_Price__c"),
                "Peak_Price": rec.get("Peak_Price__c"),
                "Sell_Price": rec.get("Sell_Price__c"),
                "PNL": rec.get("Total_PNL__c"),
                "ROI%": rec.get("Total_Return_Percentage__c"),
                "Capital_ROI%": rec.get("Capital_ROI_Pct__c"),
                "Reason": rec.get("Reason__c"),
                "Parameters": rec.get("Parameters__c"),
            })

        return jsonify({"status": "success", "data": rows})
    except Exception as e:
        logger.error("Failed loading backtest stats from Salesforce: %s", e)
        return jsonify({
            "status": "error",
            "message": f"Salesforce fetch failed: {str(e)}"
        }), 500


@app.route('/api/run-backtest', methods=['POST'])
@login_required
def run_backtest():
    """Execute selected strategy backtest and sync to Salesforce.
    
    Request JSON:
    {
        "strategy": "gamma_blast" or "v4_gamma",
        "capital": 500000,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
    """
    data = request.json or {}
    strategy = data.get('strategy', 'gamma_blast')
    capital = data.get('capital', 500000)
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    if not strategy or strategy not in ['gamma_blast', 'v4_gamma']:
        return jsonify({
            "status": "error",
            "message": f"Invalid strategy: {strategy}. Must be 'gamma_blast' or 'v4_gamma'"
        }), 400
    
    try:
        from backtest_orchestrator import BacktestOrchestrator
        
        logger.info(f"Starting backtest: strategy={strategy}, capital={capital}")
        
        orchestrator = BacktestOrchestrator()
        
        # Build kwargs for strategy
        kwargs = {'capital': capital}
        if start_date:
            kwargs['start_date'] = start_date
        if end_date:
            kwargs['end_date'] = end_date
        
        # Run and sync
        result = orchestrator.run_and_sync(strategy, **kwargs)
        
        if "error" in result:
            return jsonify({
                "status": "error",
                "message": result.get("error", "Unknown error"),
                "strategy": strategy
            }), 400
        
        return jsonify({
            "status": "success",
            "strategy": strategy,
            "execution": result.get("execution"),
            "sync": result.get("sync"),
            "overall_status": result.get("overall_status", "unknown")
        })
        
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "strategy": strategy
        }), 500



@app.route('/api/delete-backtest', methods=['POST'])
@login_required
def delete_backtest():
    """Delete Salesforce backtest records for a specified filter set."""
    data = request.json or {}
    strategy = data.get('strategy', 'ALL')
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    run_mode = data.get('run_mode', 'ALL')
    run_date = data.get('run_date', 'ALL')

    try:
        from simple_salesforce import Salesforce

        sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN", ""),
            domain=os.getenv("SF_DOMAIN", "login"),
            version=os.getenv("SF_API_VERSION", "59.0"),
        )

        def _escape_soql_string(value):
            return str(value).replace("\\", "\\\\").replace("'", "\\'")

        # Build SOQL WHERE clause from active filters
        where_clauses = []

        if strategy != 'ALL':
            where_clauses.append(f"Strategy_Name__c = '{_escape_soql_string(strategy)}'")

        if run_mode != 'ALL':
            where_clauses.append(f"Run_Mode__c = '{_escape_soql_string(run_mode)}'")

        if run_date != 'ALL':
            where_clauses.append(f"Run_Date__c = {run_date}")

        if start_date:
            where_clauses.append(f"Trade_Date__c >= {start_date}")

        if end_date:
            where_clauses.append(f"Trade_Date__c <= {end_date}")

        where_sql = " AND ".join(where_clauses) if where_clauses else "Id != null"
        soql_query = f"SELECT Id FROM historical_backtests__c WHERE {where_sql}"
        sf_records = sf.query_all(soql_query)

        if not sf_records.get('records'):
            return jsonify({
                "status": "success",
                "message": "No Salesforce records matched the selected filters."
            })

        record_ids = [rec['Id'] for rec in sf_records['records']]
        bulk_obj = getattr(sf.bulk, 'historical_backtests__c')
        delete_records = [{'Id': rid} for rid in record_ids]
        delete_result = bulk_obj.delete(delete_records)

        sf_deleted = sum(1 for r in delete_result if r.get('success', False))
        sf_failed = len(delete_result) - sf_deleted

        logger.info("Deleted %d Salesforce backtest records (%d failed)", sf_deleted, sf_failed)

        return jsonify({
            "status": "success",
            "message": f"Deleted Salesforce records: {sf_deleted} (failed: {sf_failed})"
        })
    except Exception as e:
        logger.error(f"Error deleting backtest data: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/api/test-strategy', methods=['POST'])
@login_required
def test_strategy():
    """Execute selected strategy backtest and sync to Salesforce.
    
    Request JSON:
    {
        "strategy": "gamma_blast" or "v4_gamma" (default: ALL for all available),
        "capital": 500000,
        "start_date": "2025-01-01",
        "end_date": "2025-12-31"
    }
    
    Returns results summary and Salesforce sync status.
    """
    data = request.json or {}
    strategy = data.get('strategy', 'gamma_blast')
    
    # Handle 'ALL' by running all strategies
    strategies_to_run = ['gamma_blast', 'v4_gamma'] if strategy == 'ALL' else [strategy]
    
    # Validate strategy names
    valid_strategies = ['gamma_blast', 'v4_gamma']
    for strat in strategies_to_run:
        if strat not in valid_strategies:
            return jsonify({
                "status": "error",
                "message": f"Invalid strategy: {strat}. Must be one of: {', '.join(valid_strategies)}"
            }), 400
    
    capital = data.get('capital', 500000)
    start_date = data.get('start_date')
    end_date = data.get('end_date')
    
    try:
        from backtest_orchestrator import BacktestOrchestrator
        
        all_results = {
            "status": "success",
            "strategies_run": [],
            "message": f"Backtest completed for {len(strategies_to_run)} strategy/strategies"
        }
        
        # Build kwargs for strategy
        kwargs = {'capital': capital}
        if start_date:
            kwargs['start_date'] = start_date
        if end_date:
            kwargs['end_date'] = end_date
        
        # Run all selected strategies
        for strat in strategies_to_run:
            logger.info(f"Running strategy: {strat}")
            orchestrator = BacktestOrchestrator()
            
            try:
                result = orchestrator.run_and_sync(strat, **kwargs)
                all_results["strategies_run"].append({
                    "strategy": strat,
                    "execution": result.get("execution"),
                    "sync": result.get("sync"),
                    "status": "success" if "error" not in result else "error"
                })
            except Exception as e:
                logger.error(f"Strategy {strat} failed: {e}", exc_info=True)
                all_results["strategies_run"].append({
                    "strategy": strat,
                    "status": "error",
                    "error": str(e)
                })
        
        # Check if any strategy had errors
        has_errors = any(r.get("status") == "error" for r in all_results.get("strategies_run", []))
        if has_errors and len(strategies_to_run) > 1:
            all_results["status"] = "partial"
        elif has_errors and len(strategies_to_run) == 1:
            all_results["status"] = "error"
        
        return jsonify(all_results)
        
    except Exception as e:
        logger.error(f"Backtest execution failed: {e}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": str(e),
            "strategies": strategies_to_run
        }), 500


# ── Strategy Performance & Analytics ─────────────────────────────────────────

@app.route('/api/strategy-performance', methods=['GET'])
@login_required
def get_strategy_performance():
    """Get performance statistics for all strategies from Salesforce."""
    try:
        from simple_salesforce import Salesforce
        
        sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN", ""),
            domain=os.getenv("SF_DOMAIN", "login"),
            version=os.getenv("SF_API_VERSION", "59.0"),
        )
        
        # Get all backtest records from Salesforce
        query = """
            SELECT Strategy_Name__c, 
                   COUNT() as Total_Trades,
                   SUM(Total_PNL__c) as Total_PNL,
                   SUM(CASE WHEN Win__c = true THEN 1 ELSE 0 END) as Wins,
                   AVG(Total_Return_Percentage__c) as Avg_Return_Pct,
                   MAX(Total_Return_Percentage__c) as Max_Return,
                   MIN(Total_Return_Percentage__c) as Min_Return
            FROM historical_backtests__c
            GROUP BY Strategy_Name__c
            ORDER BY MAX(Trade_Date__c) DESC
        """
        
        results = sf.query_all(query)
        
        strategies = []
        for record in results.get('records', []):
            strategies.append({
                "name": record.get('Strategy_Name__c', 'Unknown'),
                "total_trades": record.get('Total_Trades', 0),
                "total_pnl": float(record.get('Total_PNL', 0) or 0),
                "wins": record.get('Wins', 0),
                "win_rate": round((record.get('Wins', 0) / max(record.get('Total_Trades', 1), 1)) * 100, 2),
                "avg_return": round(float(record.get('Avg_Return_Pct', 0) or 0), 2),
                "max_return": round(float(record.get('Max_Return', 0) or 0), 2),
                "min_return": round(float(record.get('Min_Return', 0) or 0), 2),
            })
        
        return jsonify({
            "status": "success",
            "data": strategies
        })
    except Exception as e:
        logger.error(f"Failed to fetch strategy performance: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/strategy-backtest/<strategy>', methods=['GET'])
@login_required
def get_strategy_backtest(strategy):
    """Get latest backtest results for a specific strategy from Salesforce."""
    try:
        from simple_salesforce import Salesforce
        
        sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN", ""),
            domain=os.getenv("SF_DOMAIN", "login"),
            version=os.getenv("SF_API_VERSION", "59.0"),
        )
        
        # Get latest trades for the strategy
        query = f"""
            SELECT Run_Date__c, Trade_Date__c, Entry_Time__c, Exit_Time__c,
                   Option_Type__c, Action__c, Qty__c, Buy_Price__c, Sell_Price__c,
                   Peak_Price__c, Total_PNL__c, Total_Return_Percentage__c, 
                   Capital_ROI_Pct__c, Reason__c, Win__c
            FROM historical_backtests__c
            WHERE Strategy_Name__c = '{strategy}'
            ORDER BY Trade_Date__c DESC, Entry_Time__c DESC
            LIMIT 50
        """
        
        results = sf.query_all(query)
        
        trades = []
        for record in results.get('records', []):
            trades.append({
                "run_date": record.get('Run_Date__c'),
                "trade_date": record.get('Trade_Date__c'),
                "entry_time": record.get('Entry_Time__c'),
                "exit_time": record.get('Exit_Time__c'),
                "option_type": record.get('Option_Type__c'),
                "action": record.get('Action__c'),
                "qty": record.get('Qty__c'),
                "buy_price": float(record.get('Buy_Price__c', 0) or 0),
                "sell_price": float(record.get('Sell_Price__c', 0) or 0),
                "peak_price": float(record.get('Peak_Price__c', 0) or 0),
                "pnl": float(record.get('Total_PNL__c', 0) or 0),
                "return_pct": float(record.get('Total_Return_Percentage__c', 0) or 0),
                "capital_roi_pct": float(record.get('Capital_ROI_Pct__c', 0) or 0),
                "reason": record.get('Reason__c'),
                "win": record.get('Win__c'),
            })
        
        # Calculate summary stats
        total_pnl = sum(t['pnl'] for t in trades)
        total_trades = len(trades)
        wins = sum(1 for t in trades if t['win'])
        win_rate = round((wins / total_trades * 100), 2) if total_trades > 0 else 0
        avg_return = round(sum(t['return_pct'] for t in trades) / total_trades, 2) if total_trades > 0 else 0
        
        return jsonify({
            "status": "success",
            "strategy": strategy,
            "summary": {
                "total_trades": total_trades,
                "total_pnl": round(total_pnl, 2),
                "wins": wins,
                "win_rate": win_rate,
                "avg_return": avg_return,
                "latest_run_date": trades[0]['run_date'] if trades else None
            },
            "recent_trades": trades[:10]  # Return top 10 most recent
        })
    except Exception as e:
        logger.error(f"Failed to fetch backtest for strategy {strategy}: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/expiry-strategy-stats/<expiry_date>', methods=['GET'])
@login_required
def get_expiry_strategy_stats(expiry_date):
    """Get strategy performance for a specific expiry date."""
    try:
        from simple_salesforce import Salesforce
        
        sf = Salesforce(
            username=os.getenv("SF_USERNAME"),
            password=os.getenv("SF_PASSWORD"),
            security_token=os.getenv("SF_SECURITY_TOKEN", ""),
            domain=os.getenv("SF_DOMAIN", "login"),
            version=os.getenv("SF_API_VERSION", "59.0"),
        )
        
        # Get trades for this expiry date
        query = f"""
            SELECT Strategy_Name__c, Option_Type__c, Action__c, 
                   COUNT() as Total,
                   SUM(Total_PNL__c) as PNL,
                   SUM(CASE WHEN Win__c = true THEN 1 ELSE 0 END) as Wins
            FROM historical_backtests__c
            WHERE Trade_Date__c = {expiry_date}
            GROUP BY Strategy_Name__c, Option_Type__c, Action__c
        """
        
        results = sf.query_all(query)
        
        stats = []
        for record in results.get('records', []):
            total = record.get('Total', 0)
            wins = record.get('Wins', 0)
            stats.append({
                "strategy": record.get('Strategy_Name__c'),
                "option_type": record.get('Option_Type__c'),
                "action": record.get('Action__c'),
                "total_trades": total,
                "wins": wins,
                "win_rate": round((wins / total * 100), 2) if total > 0 else 0,
                "total_pnl": float(record.get('PNL', 0) or 0),
            })
        
        return jsonify({
            "status": "success",
            "expiry_date": expiry_date,
            "data": stats
        })
    except Exception as e:
        logger.error(f"Failed to fetch expiry stats: {e}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == '__main__':
    port = int(os.getenv("PORT", 5002))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
