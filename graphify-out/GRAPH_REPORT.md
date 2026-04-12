# Graph Report - .  (2026-04-12)

## Corpus Check
- Corpus is ~24,081 words - fits in a single context window. You may not need a graph.

## Summary
- 270 nodes · 411 edges · 21 communities detected
- Extraction: 84% EXTRACTED · 16% INFERRED · 0% AMBIGUOUS · INFERRED: 64 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]

## God Nodes (most connected - your core abstractions)
1. `NiftyGammaSpikeStrategy` - 24 edges
2. `BacktestOrchestrator` - 22 edges
3. `NiftyV4TrailingSLStrategy` - 21 edges
4. `V4Backtester` - 18 edges
5. `ExpiryManager` - 17 edges
6. `NiftyTuesdayDhanBacktester` - 16 edges
7. `GammaSpikeBacktester` - 15 edges
8. `PostgresExpiryRepository` - 7 edges
9. `_collect_schema()` - 6 edges
10. `_build_deployment_zip()` - 6 edges

## Surprising Connections (you probably didn't know these)
- `Init Dhan client using DHAN_ACCESS_TOKEN from env (Local .env or Render Env Vari` --uses--> `BacktestOrchestrator`  [INFERRED]
  app.py → backtest_orchestrator.py
- `Background thread that drives active strategies.` --uses--> `BacktestOrchestrator`  [INFERRED]
  app.py → backtest_orchestrator.py
- `Build a forward expiry calendar without database persistence.` --uses--> `BacktestOrchestrator`  [INFERRED]
  app.py → backtest_orchestrator.py
- `Backward-compatible endpoint returning generated expiry calendar.` --uses--> `BacktestOrchestrator`  [INFERRED]
  app.py → backtest_orchestrator.py
- `Return all upcoming expiry records for the next few months.` --uses--> `BacktestOrchestrator`  [INFERRED]
  app.py → backtest_orchestrator.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.06
Nodes (27): _build_expiry_calendar(), close_all_positions(), delete_backtest(), get_expiries_all(), get_expiries_history(), get_expiry_strategy_stats(), get_positions(), get_strategy_backtest() (+19 more)

### Community 1 - "Community 1"
Cohesion: 0.09
Nodes (16): GammaSpikeBacktester, get_last_n_nifty_expiries(), backtest_gamma.py — Gamma Spike SELL Strategy Backtest Morning session: benchmar, BacktestOrchestrator, backtest_orchestrator.py — Unified backtest executor with Salesforce integration, Sync backtest results to Salesforce.                  Args:             results:, Execute strategy and sync results to Salesforce in one call.                  Ar, Manages strategy execution and Salesforce sync. (+8 more)

### Community 2 - "Community 2"
Cohesion: 0.09
Nodes (22): Enum, _backfill_to_trading_day(), ExpiryFetcherBase, ExpiryRecord, ExpiryType, _is_trading_day(), _last_weekday_of_month(), MonthlyLastWeekdayFetcher (+14 more)

### Community 3 - "Community 3"
Cohesion: 0.14
Nodes (10): NiftyTuesdayDhanBacktester, Manual ADX calculation using Wilder's Smoothing., Manual Supertrend calculation based on ATR., Manual RSI calculation using Wilder's Smoothing., Gamma Spike Strategy Backtest         ──────────────────────────────────────────, Estimate Black Scholes price for options (simplified for PnL tracking)., V5 Backtest incorporating all 10 improvements over V4:         #1  Confirmation, Get the date strings for the last N Tuesdays. (+2 more)

### Community 4 - "Community 4"
Cohesion: 0.13
Nodes (13): main(), NiftyGammaSpikeStrategy, Capture individual CE & PE prices at 9:20 AM as baseline., Called every ~60 seconds by app.py engine., Gamma Spike SELL Strategy — Morning Session     ────────────────────────────────, Sell whichever leg has spiked >= 20% from baseline benchmark., Check if target or SL hit; buy back to close the short., Record or execute a SELL (short) order. (+5 more)

### Community 5 - "Community 5"
Cohesion: 0.13
Nodes (22): _build_deployment_zip(), _build_object_xml(), _build_package_xml(), _build_permissionset_xml(), _collect_schema(), _deploy_and_wait(), _ensure_permissionset_assigned(), main() (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.16
Nodes (8): Main check loop - called every 1 min, Entry: individual CE or PE leg must expand >=17% from its 1:30 PM price., Trailing SL logic matching backtest exactly., Simulates or places a real order., Safely verify our open quantity directly from Broker to avoid naked short sellin, Simulates or closes a real position., Fetches spot, atm prices, and VIX from Dhan., Sets the 1:45 PM baseline for spot and straddle price.

### Community 7 - "Community 7"
Cohesion: 0.17
Nodes (9): nearest_upcoming(), NseExpiryFetcher, NseSessionBase, PlaywrightNseSession, nse_expiry_fetcher.py  Fetches real option chain expiry dates from NSE India usi, Convert NSE date strings ('10-Mar-2026', '17-Mar-2026', …) to date objects     a, Fetches real expiry dates for NSE-listed indices directly from NSE India.     Re, Return the nearest upcoming expiry date for the given script, or None. (+1 more)

### Community 8 - "Community 8"
Cohesion: 0.16
Nodes (7): ExpiryRepositoryBase, PostgresExpiryRepository, expiry_repository.py  Responsibility: Persist and retrieve ExpiryRecord objects, Return the most-recently stored expiry per script as a list of dicts., Stores expiry records in a PostgreSQL table using an injected engine., Create the table if it does not already exist (idempotent)., Upsert a list of dicts with keys:             script_name, expiry_date (date|str

### Community 9 - "Community 9"
Cohesion: 0.38
Nodes (5): build_zip(), deploy_zip(), main(), Deploy a metadata zip, poll until done, return True on success., Build a Metadata API deployment ZIP that:       1. Sets fieldPermissions (read +

### Community 10 - "Community 10"
Cohesion: 0.53
Nodes (5): get_last_weekday_of_month(), get_previous_trading_day(), is_trading_holiday(), populate(), expiry_populator.py  Populates script_expiries table with future expiries for th

### Community 11 - "Community 11"
Cohesion: 0.4
Nodes (2): A standard WhatsApp Alerter using the official Meta WhatsApp Cloud API.     To u, WhatsAppAlerter

### Community 12 - "Community 12"
Cohesion: 0.4
Nodes (2): A simple Telegram Alerter using the official Telegram Bot API.     To get your A, TelegramAlerter

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (0): 

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Return list of expiry date strings in 'DD-Mon-YYYY' format.

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (0): 

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Persist expiry records. Returns number of rows affected.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): Return most-recently saved expiry per script.

## Knowledge Gaps
- **70 isolated node(s):** `Build a Metadata API deployment ZIP that:       1. Sets fieldPermissions (read +`, `Deploy a metadata zip, poll until done, return True on success.`, `Return a Salesforce Metadata API field descriptor dict for a given     SQLAlchem`, `Convert an arbitrary string (table name or column name) to a valid     Salesforc`, `Human-readable label, max _MAX_LABEL chars.` (+65 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 13`** (2 nodes): `compare_adx.py`, `run_comparison()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `test_dhan.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 15`** (1 nodes): `clean_db.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (1 nodes): `gunicorn.conf.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Return list of expiry date strings in 'DD-Mon-YYYY' format.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Persist expiry records. Returns number of rows affected.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `Return most-recently saved expiry per script.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `BacktestOrchestrator` connect `Community 1` to `Community 0`?**
  _High betweenness centrality (0.135) - this node is a cross-community bridge._
- **Why does `ExpiryManager` connect `Community 0` to `Community 2`?**
  _High betweenness centrality (0.126) - this node is a cross-community bridge._
- **Why does `NiftyGammaSpikeStrategy` connect `Community 4` to `Community 0`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 11 inferred relationships involving `NiftyGammaSpikeStrategy` (e.g. with `Init Dhan client using DHAN_ACCESS_TOKEN from env (Local .env or Render Env Vari` and `Background thread that drives active strategies.`) actually correct?**
  _`NiftyGammaSpikeStrategy` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 13 inferred relationships involving `BacktestOrchestrator` (e.g. with `GammaSpikeBacktester` and `V4Backtester`) actually correct?**
  _`BacktestOrchestrator` has 13 INFERRED edges - model-reasoned connections that need verification._
- **Are the 11 inferred relationships involving `NiftyV4TrailingSLStrategy` (e.g. with `Init Dhan client using DHAN_ACCESS_TOKEN from env (Local .env or Render Env Vari` and `Background thread that drives active strategies.`) actually correct?**
  _`NiftyV4TrailingSLStrategy` has 11 INFERRED edges - model-reasoned connections that need verification._
- **Are the 9 inferred relationships involving `V4Backtester` (e.g. with `BacktestOrchestrator` and `backtest_orchestrator.py — Unified backtest executor with Salesforce integration`) actually correct?**
  _`V4Backtester` has 9 INFERRED edges - model-reasoned connections that need verification._