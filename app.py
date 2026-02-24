import os
import threading
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from flask import Flask, render_template, jsonify, request, make_response
from dhanhq import dhanhq
from gamma_spike_strategy import NiftyGammaSpikeStrategy, CLIENT_ID, ACCESS_TOKEN

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Dhan API client for position management
dhan = dhanhq(str(os.getenv("DHAN_CLIENT_ID")), str(os.getenv("DHAN_ACCESS_TOKEN")))

# Global Strategy Instance and State
current_broker = "Dhan"
strategy = NiftyGammaSpikeStrategy(CLIENT_ID, ACCESS_TOKEN)
strategy_thread = None
expiry_date = "2026-03-02" # Updated strictly to matching Dhan valid expiries

# Initialize missing properties for the frontend
strategy.running = False
strategy.paused = False
strategy.unrealized_pnl = 0

def strategy_loop():
    """Background thread to run the strategy loop."""
    global expiry_date, strategy
    logger.info(f"Background Strategy Thread Started. Broker: {current_broker}.")
    
    while getattr(strategy, 'running', False):
        try:
            if not getattr(strategy, 'paused', False):
                strategy.run_iteration(expiry_date)
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
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        "broker": current_broker,
        "running": getattr(strategy, 'running', False),
        "paused": getattr(strategy, 'paused', False),
        "in_position": getattr(strategy, 'in_position', False),
        "unrealized_pnl": getattr(strategy, 'unrealized_pnl', 0)
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
    """Fully stop the strategy bot - sets running=False, waits for thread to exit."""
    global strategy_thread
    strategy.running = False
    strategy.paused = False
    strategy.in_position = False
    
    # Wait for the background thread to actually terminate (max 5 seconds)
    if strategy_thread is not None and strategy_thread.is_alive():
        strategy_thread.join(timeout=5)
        if strategy_thread.is_alive():
            logger.warning("Strategy thread did not stop within 5s timeout.")
        else:
            logger.info("Strategy thread stopped cleanly.")
    
    strategy_thread = None
    logger.info("🛑 Strategy Bot STOPPED via dashboard.")
    return jsonify({"status": "success", "message": "Strategy Bot Stopped. All monitoring halted."})

@app.route('/api/close_all_positions', methods=['POST'])
def close_all_positions():
    """
    1. Cancel all pending/transit orders.
    2. Fetch all open positions from Dhan and place MARKET counter-orders to square off.
    BUY positions get a SELL order, SELL positions get a BUY order.
    """
    try:
        # STEP 1: Cancel all pending orders
        orders_response = dhan.get_order_list()
        if orders_response.get("status") == "success":
            orders = orders_response.get("data", [])
            cancelled_orders = 0
            for order in orders:
                status = order.get("orderStatus", "")
                if status in ["PENDING", "TRANSIT"]:
                    order_id = order.get("orderId")
                    cancel_resp = dhan.cancel_order(order_id)
                    if cancel_resp.get("status") == "success":
                        cancelled_orders += 1
                        logger.info(f"🚫 Cancelled pending order: {order_id}")
            if cancelled_orders > 0:
                logger.info(f"Total {cancelled_orders} pending orders cancelled.")

        # STEP 2: Square off positions
        positions_response = dhan.get_positions()
        
        if positions_response.get("status") != "success":
            return jsonify({
                "status": "error", 
                "message": f"Failed to fetch positions: {positions_response}"
            }), 500
        
        positions = positions_response.get("data", [])
        
        if not positions:
            return jsonify({"status": "success", "message": "No open positions found.", "closed": 0})
        
        # Filter only positions with non-zero net quantity
        open_positions = [p for p in positions if int(p.get("netQty", 0)) != 0]
        
        if not open_positions:
            return jsonify({"status": "success", "message": "No open positions to close.", "closed": 0})
        
        closed_count = 0
        errors = []
        
        for pos in open_positions:
            sec_id = str(pos.get("securityId", ""))
            exchange = pos.get("exchangeSegment", "")
            net_qty = int(pos.get("netQty", 0))
            pos_product_type = pos.get("productType", dhan.INTRA) # Extract original product type
            
            # Determine counter direction
            if net_qty > 0:
                # Long position → SELL to close
                txn_type = dhan.SELL
                qty = net_qty
            else:
                # Short position → BUY to close
                txn_type = dhan.BUY
                qty = abs(net_qty)
            
            try:
                order_resp = dhan.place_order(
                    security_id=sec_id,
                    exchange_segment=exchange,
                    transaction_type=txn_type,
                    quantity=qty,
                    order_type=dhan.MARKET,
                    product_type=pos_product_type, # Use matching product type
                    price=0
                )
                
                if order_resp.get("status") == "success":
                    closed_count += 1
                    logger.info(f"✅ Closed position: {sec_id} | Qty: {qty} | Direction: {'SELL' if txn_type == dhan.SELL else 'BUY'}")
                else:
                    err_msg = f"Order failed for {sec_id}: {order_resp}"
                    errors.append(err_msg)
                    logger.error(err_msg)
                    
            except Exception as order_err:
                err_msg = f"Exception closing {sec_id}: {order_err}"
                errors.append(err_msg)
                logger.error(err_msg)
        
        # Also stop the strategy bot after closing all positions
        strategy.running = False
        strategy.in_position = False
        
        result_msg = f"Closed {closed_count}/{len(open_positions)} positions at MARKET."
        if errors:
            result_msg += f" Errors: {'; '.join(errors)}"
        
        logger.info(f"🔴 CLOSE ALL POSITIONS executed: {result_msg}")
        return jsonify({"status": "success", "message": result_msg, "closed": closed_count, "errors": errors})
        
    except Exception as e:
        logger.error(f"Close all positions failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/backtests', methods=['GET'])
def get_backtests():
    try:
        uri = os.getenv("POSTGRES_URI", "postgresql://postgres:Aidni%40%23123@localhost:5432/postgres")
        engine = create_engine(uri)
        df = pd.read_sql_table("historical_backtests", con=engine)
        # Sort by Run_Date descending to show newest test runs first
        df = df.sort_values(by="Run_Date", ascending=False)
        return jsonify({"status": "success", "data": df.to_dict(orient="records")})
    except Exception as e:
        logger.error(f"Error fetching backtests: {e}")
        return jsonify({"status": "error", "message": str(e), "data": []}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=False, threaded=True)
