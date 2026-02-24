import os
import threading
import time
import logging
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine
from flask import Flask, render_template, jsonify, request, make_response
from dhanhq import dhanhq
from v4_trailing_sl_strategy import NiftyV4TrailingSLStrategy

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Dhan API client
CLIENT_ID = os.getenv("DHAN_CLIENT_ID")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN")
dhan = dhanhq(str(CLIENT_ID), str(ACCESS_TOKEN))

# Global Strategy Instance and State
current_broker = "Dhan"
expiry_date = "2026-03-02" 
strategy = NiftyV4TrailingSLStrategy(expiry_date)
strategy_thread = None

def strategy_loop():
    """Background thread to run the strategy loop."""
    global expiry_date, strategy
    logger.info(f"Background Strategy Thread Started. Broker: {current_broker}.")
    
    while getattr(strategy, 'running', False):
        try:
            if not getattr(strategy, 'paused', False):
                strategy.run_iteration()
        except Exception as e:
            logger.error(f"Error in strategy iteration: {e}")
        
        # Sleep in short increments to remain responsive to the 'running' flag
        for _ in range(60): 
            if not getattr(strategy, 'running', False):
                break
            time.sleep(1)
            
    logger.info("Background Strategy Thread Stopped.")

@app.route('/')
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return response

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "broker": current_broker,
        "running": getattr(strategy, 'running', False),
        "paused": getattr(strategy, 'paused', False),
        "in_position": getattr(strategy, 'in_position', False),
        "unrealized_pnl": getattr(strategy, 'unrealized_pnl', 0),
        "realized_pnl": getattr(strategy, 'realized_pnl', 0)
    })

@app.route('/api/start', methods=['POST'])
def start():
    global strategy_thread
    if strategy_thread is None or not strategy_thread.is_alive():
        strategy.running = True
        strategy.paused = False
        strategy_thread = threading.Thread(target=strategy_loop, daemon=True)
        strategy_thread.start()
        return jsonify({"status": "success", "message": "Strategy Started"})
    return jsonify({"status": "error", "message": "Strategy already running"}), 400

@app.route('/api/pause', methods=['POST'])
def pause():
    strategy.paused = True
    return jsonify({"status": "success", "message": "Strategy Paused"})

@app.route('/api/resume', methods=['POST'])
def resume():
    strategy.paused = False
    return jsonify({"status": "success", "message": "Strategy Resumed"})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    strategy.running = False
    return jsonify({"status": "success", "message": "Strategy Stopped"})

@app.route('/api/close_all_positions', methods=['POST'])
def close_all_positions():
    """Cancel all orders and close all open positions."""
    try:
        # 1. Cancel all pending orders first
        dhan.cancel_all_orders()
        
        # 2. Get and close all open positions
        pos_resp = dhan.get_positions()
        if pos_resp.get("status") == "success":
            positions = pos_resp.get("data", [])
            for p in positions:
                if int(p.get("netQty", 0)) != 0:
                    dhan.place_order(
                        security_id=p["securityId"],
                        exchange_segment=p["exchangeSegment"],
                        transaction_type=dhan.SELL if int(p["netQty"]) > 0 else dhan.BUY,
                        quantity=abs(int(p["netQty"])),
                        order_type=dhan.MARKET,
                        product_type=p["productType"]
                    )
        
        # Stop bot logic
        strategy.running = False
        strategy.in_position = False
        return jsonify({"status": "success", "message": "All orders cancelled and positions closed."})
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/backtests', methods=['GET'])
def get_backtests():
    uri = os.getenv("POSTGRES_URI", "postgresql://postgres:Aidni%40%23123@localhost:5432/postgres")
    try:
        engine = create_engine(uri)
        df = pd.read_sql("SELECT * FROM historical_backtests ORDER BY \"Date\" DESC, \"Entry_Time\" DESC", con=engine)
        return jsonify({"status": "success", "data": df.to_dict(orient='records')})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
