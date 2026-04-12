# AntiGravity — GitHub Copilot Instructions

## Knowledge Graph

A graphify knowledge graph is available for this project.
Before answering architecture questions or searching raw files, read:
`graphify-out/GRAPH_REPORT.md` for god nodes and community structure.

To query the graph precisely:
```bash
graphify query "your question" --graph graphify-out/graph.json
graphify path "NodeA" "NodeB" --graph graphify-out/graph.json
```

To rebuild the graph after significant changes:
```bash
source .venv/bin/activate
python -c "
import json
from graphify.detect import detect
from graphify.extract import collect_files, extract
from graphify.build import build_from_json
from graphify.cluster import cluster, score_all
from graphify.analyze import god_nodes, surprising_connections, suggest_questions
from graphify.report import generate
from graphify.export import to_json, to_html
from pathlib import Path

detection = json.loads(Path('graphify-out/.graphify_detect.json').read_text())
code_files_raw = detection.get('files', {}).get('code', [])
code_files = []
for f in code_files_raw:
    code_files.extend(collect_files(Path(f)) if Path(f).is_dir() else [Path(f)])
ast_result = extract(code_files)
G = build_from_json(ast_result)
communities = cluster(G)
cohesion = score_all(G, communities)
labels = {cid: 'Community ' + str(cid) for cid in communities}
questions = suggest_questions(G, communities, labels)
report = generate(G, communities, cohesion, labels, god_nodes(G), surprising_connections(G, communities), detection, {'input':0,'output':0}, '.', suggested_questions=questions)
Path('graphify-out/GRAPH_REPORT.md').write_text(report)
to_json(G, communities, 'graphify-out/graph.json')
to_html(G, communities, 'graphify-out/graph.html')
print(f'Rebuilt: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges')
"
```

---

## Architecture Overview (from graph — 270 nodes, 411 edges, 21 communities)

### God Nodes (most connected — touch these carefully)
1. `NiftyGammaSpikeStrategy` (24 edges) — live trading engine in `strategy/gamma_spike_strategy.py`
2. `BacktestOrchestrator` (22 edges) — central coordinator in `backtest_orchestrator.py`
3. `NiftyV4TrailingSLStrategy` (21 edges) — live V4 trailing SL engine in `strategy/v4_trailing_sl_strategy.py`
4. `V4Backtester` (18 edges) — backtesting engine in `backtest_v4.py`
5. `ExpiryManager` (17 edges) — expiry calendar manager in `strategy/expiry_manager.py`

### Key Clusters
| Community | What it is | Key files |
|---|---|---|
| Backtest+Salesforce | Backtest execution + SF sync | `backtest_orchestrator.py`, `backtest_gamma.py` |
| Live Strategy | Active trading strategies | `strategy/gamma_spike_strategy.py`, `strategy/v4_trailing_sl_strategy.py` |
| Expiry Management | NSE expiry fetching + persistence | `strategy/expiry_manager.py`, `strategy/expiry_repository.py`, `strategy/nse_expiry_fetcher.py` |
| Flask API | REST endpoints + auth | `app.py` |
| Alerting | Telegram + WhatsApp notifications | `strategy/telegram_alerter.py`, `strategy/whatsapp_alerter.py` |
| Salesforce FLS | Field-level security layout deploy | `fix_fls_layout.py` |

---

## Codebase Conventions

### Salesforce Sync
- All backtest data is stored in Salesforce object `historical_backtests__c`
- Only **`backtest_orchestrator.py`** should call `sync_to_salesforce()` — never call it directly from strategy backtester classes (causes duplicate inserts)
- `GammaSpikeBacktester.run()` must NOT call `self.save_to_salesforce()` — the orchestrator handles SF sync
- Use `sf.bulk.historical_backtests__c.insert(records)` for bulk writes

### API Routes (app.py)
- `GET /api/backtests` — fetches all records from Salesforce (up to 5000, DESC order)
- `POST /api/test-strategy` — triggers backtest via `BacktestOrchestrator.run_and_sync()`
- `POST /api/delete-backtest` — bulk deletes SF records by filter
- All routes are protected with `@login_required`
- Table filtering on the frontend is **client-side only** — no API calls on filter change

### Strategy Files
- Live strategies live in `strategy/` folder
- Backtesting equivalents live at root: `backtest_v4.py`, `backtest_gamma.py`, `backtest_dhan_5min.py`
- `OptionFetcher` in `backtest_v4.py` is shared — imported by other backtest files
- Cache for fetched DHAN data is stored in `/tmp/*.pkl` — safe to delete for fresh data

### Datetime Handling
- All datetimes are stored in Salesforce as UTC (`%Y-%m-%dT%H:%M:%S.000Z`)
- Always convert IST → UTC before writing to Salesforce
- Use `ZoneInfo("Asia/Kolkata")` for IST, not pytz

### Environment Variables
- All secrets loaded from `.env` via `load_dotenv(override=True)`
- Required: `DHAN_ACCESS_TOKEN`, `SF_USERNAME`, `SF_PASSWORD`, `SF_SECURITY_TOKEN`, `SF_DOMAIN`, `SF_INSTANCE`, `SF_API_VERSION`
- Never hardcode credentials; always use `os.getenv()`

### Error Handling
- Log with `logger.error(...)` not `print()` in production code
- Backtest failures per-strategy are captured and returned as `status: "partial"` — don't abort all strategies on one failure

---

## What NOT To Do
- Do not add a second `save_to_salesforce()` call in any backtest class — the orchestrator owns SF writes
- Do not add `simple_salesforce` imports to `backtest_v4.py` — it has no direct SF dependency by design
- Do not change expiry logic without checking `ExpiryManager` dependents (17 connected nodes)
- Do not bypass `@login_required` on any API route
