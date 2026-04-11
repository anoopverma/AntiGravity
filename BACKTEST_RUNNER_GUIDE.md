# Integrated Backtest Runner Configuration

## Overview

The backtest runner has been fully integrated with Salesforce to automatically sync backtest results whenever any strategy is executed. Choose a strategy from the web UI, configure parameters, and results will be automatically saved to Salesforce.

## How It Works

### Architecture

```
Web UI (backtest_runner.html)
    ↓
Flask API (/api/test-strategy)
    ↓
BacktestOrchestrator (backtest_orchestrator.py)
    ↓
Strategy Executor (gamma_blast / v4_gamma)
    ↓
Salesforce Bulk Insert
```

### Key Components

1. **BacktestOrchestrator** (`backtest_orchestrator.py`)
   - Unified orchestrator for all strategies
   - Handles Salesforce connection and sync
   - Supports both gamma_blast and v4_gamma strategies
   - Converts IST times to UTC before insertion

2. **API Endpoint** (`/api/test-strategy`)
   - Accepts POST requests with strategy selection
   - Executes selected strategy or all strategies
   - Returns detailed execution and sync results

3. **Frontend Integration** (`templates/backtest_runner.html`)
   - Strategy selector dropdown
   - Real-time execution logging
   - Result display and status indicators

## Using the Backtest Runner

### Web Interface

1. Navigate to **Backtests** tab in the AntiGravity dashboard
2. Select strategy from **Strategy** dropdown:
   - **Gamma Blast**: Gamma spike SELL strategy (48-week historical sweep)
   - **V4 Gamma**: Trailing SL strategy with momentum scaling
   - **All Strategies**: Run both strategies sequentially

3. Configure parameters:
   - **Run Mode**: Paper or Live (affects data tagging)
   - **Run Date**: Filter by existing backtest date
   - **Start Date**: Backtest data range start
   - **End Date**: Backtest data range end
   - **Initial Capital**: ₹ invested (default: ₹500,000)

4. Click **▶ DHAN BACK TEST** button
5. Watch real-time execution log:
   - Strategy execution progress
   - Total trades identified
   - Salesforce sync status
   - Number of records inserted

### Expected Output

```
🎯 Starting backtest: gamma_blast
📅 Date range: 2025-01-01 to 2025-12-31
💰 Initial Capital: ₹500,000
⏳ This may take a few moments...

✅ Backtest Execution Completed Successfully!
📊 Strategy: gamma_blast — Gamma Blast backtest completed and synced to Salesforce
   Total Trades: 27
   Salesforce Sync: All 27 records inserted successfully
```

## API Reference

### POST `/api/test-strategy`

Execute backtest and sync to Salesforce.

**Request Body:**
```json
{
  "strategy": "gamma_blast",
  "capital": 500000,
  "start_date": "2025-01-01",
  "end_date": "2025-12-31"
}
```

**Parameters:**
- `strategy` (string): "gamma_blast", "v4_gamma", or "ALL"
- `capital` (integer): Initial capital in rupees
- `start_date` (string): YYYY-MM-DD format
- `end_date` (string): YYYY-MM-DD format

**Response:**
```json
{
  "status": "success",
  "strategies_run": [
    {
      "strategy": "gamma_blast",
      "execution": {
        "status": "success",
        "strategy": "gamma_blast",
        "total_trades": 27,
        "message": "Gamma Blast backtest completed and synced to Salesforce"
      },
      "sync": {
        "status": "success",
        "message": "All 27 records inserted successfully",
        "inserted": 27
      },
      "status": "success"
    }
  ],
  "message": "Backtest completed for 1 strategy/strategies"
}
```

## Salesforce Data Fields

Each trade record is inserted into `historical_backtests__c` with:

| Field | Type | Description |
|-------|------|-------------|
| Strategy_Name__c | Text | Strategy identifier (gamma_blast, v4_gamma) |
| Trade_Date__c | Date | Date trade occurred |
| Entry_Time__c | DateTime | Entry time (UTC) |
| Exit_Time__c | DateTime | Exit time (UTC) |
| Option_Type__c | Text | CE or PE |
| Action__c | Text | SELL or BUY |
| Qty__c | Number | Quantity traded |
| Buy_Price__c | Number | Buy/Entry price |
| Sell_Price__c | Number | Sell/Exit price |
| Peak_Price__c | Number | Peak premium during trade |
| Total_PNL__c | Number | Profit/Loss in rupees |
| Total_Return_Percentage__c | Number | Return % |
| Capital_ROI_Pct__c | Number | % of capital at risk |
| Run_Mode__c | Text | "backtest" |
| Win__c | Checkbox | Win/Loss flag |
| Reason__c | Text | Exit reason |
| Strike__c | Text | Strike price |
| Parameters__c | Text | Strategy parameters used |

## Timezone Handling

- **Source**: IST (Asia/Kolkata, UTC+5:30)
- **Storage**: UTC (Salesforce standard)
- **Conversion**: Automatic via `_to_sf_datetime()` function
- **Example**: 09:45 IST → 04:15 UTC

All datetime fields are converted to UTC with Z suffix before insertion:
- Format: `2025-12-30T04:15:00.000Z`

## Running Programmatically

### Command Line

```bash
cd /Users/anoop/VSCODE/AntiGravity
source .venv/bin/activate
python backtest_orchestrator.py gamma_blast
```

### Python Script

```python
from backtest_orchestrator import BacktestOrchestrator

orchestrator = BacktestOrchestrator()
result = orchestrator.run_and_sync('gamma_blast', capital=500000)

print(result)
# {
#   "execution": {...},
#   "sync": {...},
#   "overall_status": "success"
# }
```

## Troubleshooting

### "Salesforce connection failed"
- Verify `.env` contains valid SF credentials:
  - `SF_USERNAME`
  - `SF_PASSWORD`
  - `SF_SECURITY_TOKEN`
  - `SF_DOMAIN`
  - `SF_API_VERSION`

### "Invalid strategy"
- Ensure strategy is either `gamma_blast` or `v4_gamma`
- `ALL` runs both strategies

### "No results to sync"
- Strategy executed but no trades found
- Adjust date range or parameters
- Check strategy output for Reason field

### Partial Sync Success (some records failed)
- Check Salesforce field validation
- Verify field types match insertion data
- Review error message in sync response

## Configuration

### Environment Variables

```bash
# Salesforce
SF_USERNAME=your@salesforce.com
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=your_security_token
SF_DOMAIN=login  # or test for sandbox
SF_API_VERSION=59.0

# Database (PostgreSQL)
POSTGRES_URI=postgresql://user:password@localhost/dbname

# Dashboard
DASHBOARD_USERNAME=admin
DASHBOARD_PASSWORD=antigravity2024
```

### Strategy Parameters (Gamma Blast)

- `LEG_EXPANSION`: 120% spike threshold
- `MIN_SELL_PRICE`: ₹20 minimum premium
- `BENCHMARK_TIME`: 09:20 AM capture
- `ENTRY_CUTOFF`: 11:30 AM no new sells
- `TARGET_PCT`: 35% profit target
- `SL_PCT`: 40% stop-loss
- `QTY`: 1,300 contracts
- `INITIAL_CAPITAL`: ₹500,000

## Advanced Usage

### Custom Strategy Integration

To add a new strategy to the runner:

1. Create backtest class with `run()` method
2. Add to `BacktestOrchestrator._run_<strategy>()`
3. Ensure output format matches trade record structure
4. Update frontend dropdown in `backtest_runner.html`

Example structure:
```python
class CustomBacktester:
    def run(self):
        self.results = [
            {
                "Date": "2025-01-06",
                "Entry_Time": "09:30:00",
                "Exit_Time": "15:30:00",
                "Option_Type": "CE",
                "Action": "SELL",
                "Buy_Price": 100.0,
                "Sell_Price": 80.0,
                "Peak_Price": 100.0,
                "PNL": 20000.0,
                "ROI%": 5.0,
                "Capital_ROI%": 4.0,
                "Win": True,
                "Reason": "Target Hit",
                "Strike": "22000",
                "Parameters": "custom_params"
            }
        ]
```

## Performance Tips

1. **Caching**: Backtest uses disk cache (~96 entries)
2. **Bulk API**: Salesforce uses Bulk API for 27+ record inserts
3. **Date Range**: Wider ranges (1yr+) take longer but improve statistics
4. **Capital**: Higher capital doesn't affect execution speed, only PnL values

## Support

For issues or questions:
1. Check server logs: `server_log.txt`
2. Verify all environment variables are set
3. Test Salesforce connection: `pg_to_salesforce.py`
4. Run individual strategy script for debugging
