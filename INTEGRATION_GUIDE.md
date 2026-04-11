# Integration Guide: Backtest Runner with Index & Expiry Pages

## Overview

The AntiGravity backtest runner is now fully integrated with the dashboard (index.html) and expiry calendar (expiry.html). Users can:

1. **Dashboard (Index)**: View strategy performance, run quick backtests before going live
2. **Backtests**: Execute full historical backtests with configurable parameters  
3. **Expiries**: Analyze strategy performance for specific expiry dates

## New Features

### 1. Dashboard Integration (index.html)

#### Strategy Performance Panel
A new "Strategy Performance (Latest Backtest)" card displays:
- Win rates and total P&L for each strategy
- Average returns and performance range (max/min)
- Trade counts from latest backtest runs
- Color-coded indicators (green = 80%+ win rate, yellow = 50%-80%, red = <50%)

**Features:**
```
📊 Strategy Performance (Latest Backtest)
├─ Performance cards for each strategy
├─ Refresh stats button
└─ Quick action buttons:
   ├─ ⚡ Quick Test: Gamma Blast
   ├─ ⚡ Quick Test: V4 Gamma
   └─ 📊 Full Backtest → (navigate to backtests page)
```

#### Quick Backtest from Dashboard
Users can:
1. Click "Quick Test: Gamma Blast" or "Quick Test: V4 Gamma"
2. Automatically run backtest with default parameters (2025 full year data)
3. See real-time execution feedback in system logs
4. Performance stats refresh after completion
5. Decide whether to go live based on results

**Benefits:**
- No need to navigate away to test strategies
- One-click execution
- Immediate performance visibility before going live
- Reduces risk of deploying untested strategies

### 2. Expiry Calendar Integration (expiry.html)

#### Enhanced Expiry Calendar Table
Each expiry date row now includes:
- Traditional expiry info (Script, Date, Day, Source)
- **New: "📊 Stats" action button**

#### Strategy Stats by Expiry Date
Clicking "📊 Stats" opens a panel showing:

```
📊 Strategy Performance for [Expiry Date]
├─ Summary for each strategy running on that date
│  ├─ Strategy name
│  ├─ Total trades executed
│  ├─ Win rate percentage (color-coded)
│  ├─ Total P&L for that expiry
│  └─ Breakdown by option type (CE/PE) and action
└─ Quick action buttons:
   ├─ ▶ Run Backtest for This Date
   └─ ✕ Close
```

**Features:**
- Analyze past performance for specific expiry dates
- See which strategies performed best on which expiries
- Group performance data by:
  - Strategy type (gamma_blast, v4_gamma, etc.)
  - Option type (Call/Put)
  - Action (buy/sell)
- Win rates shown with visual indicators

### 3. API Endpoints

Three new REST endpoints power the integration:

#### GET `/api/strategy-performance`
Get performance metrics for all strategies from Salesforce.

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "name": "gamma_blast",
      "total_trades": 27,
      "total_pnl": 774930,
      "wins": 26,
      "win_rate": 96.3,
      "avg_return": 5.42,
      "max_return": 15.3,
      "min_return": -0.5
    }
  ]
}
```

#### GET `/api/strategy-backtest/<strategy>`
Get latest backtest results and summary for a specific strategy.

**Response:**
```json
{
  "status": "success",
  "strategy": "gamma_blast",
  "summary": {
    "total_trades": 27,
    "total_pnl": 774930,
    "wins": 26,
    "win_rate": 96.3,
    "avg_return": 5.42,
    "latest_run_date": "2026-04-12"
  },
  "recent_trades": [...]
}
```

#### GET `/api/expiry-strategy-stats/<expiry_date>`
Get strategy performance aggregated for a specific expiry date.

**Response:**
```json
{
  "status": "success",
  "expiry_date": "2025-12-30",
  "data": [
    {
      "strategy": "gamma_blast",
      "option_type": "CE",
      "action": "SELL",
      "total_trades": 1,
      "wins": 1,
      "win_rate": 100.0,
      "total_pnl": 29445
    }
  ]
}
```

## Navigation Flow

### User Journey 1: Pre-Live Testing
1. User goes to Dashboard
2. Sees strategy performance cards
3. Clicks "Quick Test: Gamma Blast"
4. System runs backtest (takes ~1-2 minutes)
5. Logs show progress (trades found, Salesforce synced)
6. Performance stats update automatically
7. User checks win rate and P&L
8. If satisfied, goes to live trading section
9. If not satisfied, adjusts strategy or goes to full backtest

### User Journey 2: Expiry Analysis  
1. User goes to Expiry Calendar
2. Sees upcoming expiry dates
3. Clicks "📊 Stats" on date of interest
4. Panel shows strategy performance for that expiry
5. Can see which strategies worked best on that date
6. Optionally runs full backtest for that specific date
7. Uses insights to plan trading for upcoming expiries

### User Journey 3: Full Backtest
1. User clicks "Full Backtest →" on dashboard
2. Or navigates directly to Backtests tab
3. Configures strategy, date range, capital
4. Clicks "▶ DHAN BACK TEST"
5. Results stored in PostgreSQL and Salesforce
6. Can filter/export results
7. Returns to dashboard (performance stats now updated)

## Technical Architecture

### Data Flow

```
Dashboard/Expiry Page (HTML/JS)
    ↓
API Endpoints (app.py)
    ├─ /api/strategy-performance
    ├─ /api/strategy-backtest/<strategy>
    └─ /api/expiry-strategy-stats/<date>
    ↓
Salesforce (historical_backtests__c)
    ↓
Cache/Display in UI
```

### Technology Stack

- **Frontend**: Vanilla JavaScript, CSS Grid
- **Backend**: Flask + Python
- **Data Source**: Salesforce (real-time), PostgreSQL (historical)
- **Execution**: BacktestOrchestrator (independent process)

## Key Components

### 1. BacktestOrchestrator (`backtest_orchestrator.py`)
- Executes strategy backtests independently
- Converts IST times to UTC automatically
- Bulk inserts results to Salesforce
- Returns execution summary and sync status

### 2. Strategy Performance APIs (`app.py`)
- Query Salesforce using SOQL
- Aggregate statistics by strategy, expiry, option type
- Return formatted JSON for UI rendering

### 3. Enhanced Templates
- **index.html**: Strategy performance panel + quick backtest
- **expiry.html**: Expiry stats panel + action buttons
- **backtest_runner.html**: Updated nav links

## Salesforce Data Requirements

For the integration to work, historical_backtests__c must contain:

**Required Fields:**
- Strategy_Name__c (Text)
- Trade_Date__c (Date)
- Entry_Time__c (DateTime)
- Exit_Time__c (DateTime)
- Option_Type__c (Text)
- Action__c (Text)
- Total_PNL__c (Number)
- Total_Return_Percentage__c (Number)
- Win__c (Checkbox)
- Run_Date__c (Date)

## Configuration

### Environment Variables
```bash
# Salesforce
SF_USERNAME=your@email.com
SF_PASSWORD=your_password
SF_SECURITY_TOKEN=token
SF_DOMAIN=login (or test for sandbox)
SF_API_VERSION=59.0

# Database
POSTGRES_URI=postgresql://user:pass@host/db
```

### Date Ranges
- Default backtest: 2025-01-01 to 2025-12-31 (full year)
- Customizable via UI selectors
- Format: YYYY-MM-DD

### Strategy Names
- `gamma_blast`: Gamma spike SELL strategy
- `v4_gamma`: Trailing SL straddle strategy
- Case-sensitive in API calls

## Performance Metrics Explained

### Win Rate
- Percentage of trades that were profitable (Win = True)
- Formula: (Wins / Total Trades) × 100
- Color indicator: Green (≥80%), Yellow (50-80%), Red (<50%)

### Total P&L
- Sum of cash profit/loss across all trades
- In ₹ (Indian Rupees)
- Color: Green (positive), Red (negative)

### Average Return
- Mean return percentage per trade
- Formula: Sum(Total_Return_Percentage) / Total_Trades
- Shows consistency of strategy

### Return Range
- Max Return: Best performing trade
- Min Return: Worst performing trade
- Shows strategy volatility

## Limitations & Considerations

1. **Performance Data Depends on History**
   - Empty stats if no backtests have been run
   - Guide users to run at least one backtest first

2. **IST to UTC Conversion**
   - All times converted automatically
   - 5:30 hour offset applied
   - Ensures correct timestamp storage

3. **Salesforce API Rate Limits**
   - Production org: 15-20 req/sec
   - Sandbox: 5-10 req/sec
   - Cache results locally if possible

4. **Real-time Data Accuracy**
   - Performance stats reflect last backtest run only
   - Running live strategies doesn't update historical stats
   - Users need to run backtests to update performance metrics

## Future Enhancements

1. **Scheduled Backtests**
   - Run backtests on fixed schedule (daily/weekly)
   - Automatically update performance metrics

2. **Strategy Comparison**
   - Compare multiple strategies side-by-side
   - Filter by performance criteria

3. **Expiry-Specific Settings**
   - Override parameters per expiry date
   - Historical optimized settings

4. **Performance Alerts**
   - Alert when strategy performance drops below threshold
   - Notify before strategy goes live

5. **PDF Reports**
   - Export backtest results as PDF
   - Share performance reports with stakeholders

## Testing Checklist

- [ ] Dashboard loads with performance panel
- [ ] Quick backtest buttons trigger backtest execution
- [ ] Performance stats refresh after backtest
- [ ] Expiry calendar shows action buttons
- [ ] Expiry stats panel displays correctly
- [ ] Navigation links work between pages
- [ ] API endpoints return correct data
- [ ] Salesforce records created/updated properly
- [ ] Timezone conversion validated (IST → UTC)
- [ ] Error messages display gracefully

## Support & Debugging

### Common Issues

1. **"Salesforce connection failed"**
   - Verify SF credentials in .env
   - Check SF API version
   - Verify security token

2. **"No backtest data available"**
   - Run at least one backtest via UI
   - Check PostgreSQL connection
   - Verify Salesforce records exist

3. **"Performance stats not updating"**
   - Run backtest again
   - Check Salesforce query in logs
   - Verify field mappings

4. **"Expiry stats empty"**
   - Run backtest with past dates
   - Check Trade_Date__c format in Salesforce
   - Verify expiry date format (YYYY-MM-DD)

### Debug Mode
Enable verbose logging:
```python
logging.basicConfig(level=logging.DEBUG)
# Or set in .env: LOG_LEVEL=DEBUG
```

## Documentation Files

- **BACKTEST_RUNNER_GUIDE.md**: Backtest runner details
- **INTEGRATION_GUIDE.md** (this file): Dashboard & Expiry integration
- **app.py**: API endpoint documentation in docstrings
- **backtest_orchestrator.py**: Orchestrator documentation

## Summary

The integration brings strategic insights directly to the dashboard:
- ✅ Real-time performance visibility
- ✅ One-click backtest execution  
- ✅ Expiry-specific analytics
- ✅ Pre-live validation
- ✅ Reduced deployment risk

Users can now make data-driven decisions before going live with strategies.
