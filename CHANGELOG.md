# Changelog

## [Unreleased] — 2026-04-09

### Added
- **`pg_to_salesforce.py`** — New migration script that copies all PostgreSQL tables
  to Salesforce Custom Objects using the Metadata API (for schema creation) and
  the Bulk API (for data insertion).
  - Introspects every PostgreSQL table and maps column types to Salesforce field types
  - Builds and deploys a Metadata ZIP to create / update Custom Objects and fields
  - Bulk-inserts all rows via `simple-salesforce` Bulk API in configurable chunk sizes
  - Reads credentials from environment variables: `POSTGRES_URI`, `SF_USERNAME`,
    `SF_PASSWORD`, `SF_SECURITY_TOKEN`, `SF_DOMAIN`, `SF_API_VERSION`

### Changed — `pg_to_salesforce.py` (migration fixes, 2026-04-09)

- **Metadata deploy: base64 → temp ZIP file** — `sf.mdapi.deploy()` in the installed
  version of `simple-salesforce` expects a file path, not a raw base64 string.
  The script now writes the deployment ZIP to a `tempfile.NamedTemporaryFile` and
  passes its path; the temp file is always cleaned up in a `finally` block.

- **Tuple-safe deploy response parsing** — `sf.mdapi.deploy()` returns a
  `(async_process_id, state)` tuple in this library version (not an object with
  an `.id` attribute). `_deploy_and_wait` now unpacks tuples, dicts, and objects
  gracefully.

- **Tuple-safe deploy-status parsing** — `sf.mdapi.check_deploy_status()` likewise
  returns a 4-tuple `(state, state_detail, deployment_detail, unit_test_detail)`.
  The polling loop now reads `status[0]` for state and `status[2]["errors"]` for
  component failures instead of calling `.get("status")` on the raw response.

- **Explicit table → Salesforce object mapping** — Auto-generated `table_name__c`
  objects are replaced by a `TABLE_TO_OBJECT_API_BASE` dict that targets the
  pre-created org objects:

  | PostgreSQL table       | Salesforce Custom Object |
  |------------------------|--------------------------|
  | `v4_backtest_results`  | `Trade__c`               |
  | `historical_backtests` | `historical_backtests__c`|
  | `script_expiries`      | `Expiry__c`              |

- **Field compatibility guard** — Before building insert payloads, the script
  now calls `sf.<Object>.describe()` and only sends fields whose API names already
  exist on the target object; unmapped columns are logged as warnings and skipped
  rather than causing `INVALID_FIELD` bulk-insert failures.

- **Fallback field mapping (`TABLE_COLUMN_TO_FIELD_MAP`)** — Provides explicit
  source-column → Salesforce-field overrides for columns whose auto-generated
  `api_name__c` doesn't match the target object's field names:
  - `v4_backtest_results.Date` → `Trade_Time__c`
  - `v4_backtest_results.Type` → `Direction__c`
  - `v4_backtest_results.Entry_Price` → `Average_Price__c`
  - `script_expiries.expiry_date` → `Expiry_Date__c`
  - `script_expiries.source` → `Source__c`
  - `historical_backtests.Run_Date` → `Run_Date__c`
  - `historical_backtests.Strategy_Name` → `Strategy_Name__c`
  - `historical_backtests.PNL` → `Total_PNL__c`
  - `historical_backtests.ROI%` → `Total_Return_Percentage__c`

- **Salesforce field-type-aware date/datetime formatting** — Timestamp conversion
  now reads the actual Salesforce field type from `describe()` (`"datetime"` /
  `"date"`) instead of relying on the PostgreSQL-derived type, fixing
  `Cannot deserialize instance of datetime from VALUE_STRING` bulk errors.

- **`SF_SKIP_METADATA_DEPLOY` flag** — Defaulted to `true` so re-runs against
  pre-created objects skip the metadata deployment step and go straight to data
  insertion, reducing run time and avoiding unnecessary deploy jobs.

### Migration run results (2026-04-09)

| Table                  | SF Object            | Rows inserted | Rows failed |
|------------------------|----------------------|---------------|-------------|
| `v4_backtest_results`  | `Trade__c`           | 9             | 0           |
| `historical_backtests` | `historical_backtests__c` | 59      | 0           |
| `script_expiries`      | `Expiry__c`          | 43            | 0           |

> **Note:** Some PostgreSQL columns were not copied because matching custom fields
> do not yet exist on the target Salesforce objects. See `WARNING Skipping unmapped
> columns` in the run logs. To copy those columns, add the missing custom fields to
> the target objects in Salesforce Setup, add entries to `TABLE_COLUMN_TO_FIELD_MAP`,
> and re-run the script.

---

## [Unreleased] — 2026-04-09 (session 2)

### Changed — `backtest_gamma.py`
- **Removed PostgreSQL dependency** — Deleted `from sqlalchemy import create_engine`
  import and the `save_to_db()` function which wrote to the `historical_backtests`
  PostgreSQL table.
- **Added Salesforce insert** — New `save_to_salesforce()` function logs into SF
  via env vars and writes one summary record (run-level aggregates: strategy name,
  run date, total PNL, ROI, trade count, win rate) plus one record per individual
  trade to `historical_backtests__c`.

### Changed — `pg_to_salesforce.py` (schema extension, session 2)

- **Full column mapping (`TABLE_COLUMN_TO_FIELD_MAP` expanded)** — All 38 columns
  across the three PostgreSQL tables are now explicitly mapped to their Salesforce
  field API names. Previously only ~9 columns were mapped; the rest were skipped.

  | Table | New fields mapped |
  |-------|-------------------|
  | `v4_backtest_results` | `Entry_Time__c`, `Exit_Time__c`, `Peak_Price__c`, `Exit_Price__c`, `PnL_INR__c`, `ROI_Pct__c`, `Reason__c` |
  | `historical_backtests` | `Trade_Date__c`, `Entry_Time__c`, `Exit_Time__c`, `Option_Type__c`, `Action__c`, `Qty__c`, `Buy_Price__c`, `Peak_Price__c`, `Sell_Price__c`, `Reason__c`, `Win__c`, `Capital_ROI_Pct__c`, `Run_Mode__c`, `Strike__c`, `Trade_Type__c`, `Entry_Price__c`, `Exit_Price__c`, `PnL_INR__c`, `Parameters__c` |
  | `script_expiries` | `Day_Label__c`, `Script_Name__c`, `Fetched_At__c` |

- **Metadata deploy creates new custom fields** — `SF_SKIP_METADATA_DEPLOY` default
  changed back to `false`; schema extension now auto-creates any fields that don't
  yet exist on the target object using the Metadata API.

- **PermissionSet deployed alongside objects** — `_build_permissionset_xml()` and
  `PERMSET_NAME = "AntiGravity_Access"` added. The deployment ZIP now includes a
  `permissionsets/AntiGravity_Access.permissionset` file that grants read + edit
  FLS on every mapped custom field for all three objects. Without this, Salesforce
  creates the fields but they are invisible to `describe()` and API inserts.

- **`_ensure_permissionset_assigned()`** — Post-deploy step that queries the org for
  the `AntiGravity_Access` PermissionSet and assigns it to the running user if not
  already assigned, ensuring FLS is active before data migration begins.

- **Switched from Bulk API v1 to REST Collections API** — Data insert now uses
  `POST /services/data/v{version}/composite/sobjects` instead of
  `sf_bulk_obj.insert(chunk)`. The Bulk API v1 caches object schemas server-side
  and does not pick up newly deployed fields for several minutes; the REST
  Collections API is always in sync with `describe()`.

- **Pre-insert field visibility guard** — Before sending records, the code calls
  `describe()` and checks that every target field in `col_map` is visible. If any
  are missing it waits 15 s and retries once, then skips the table with an error
  rather than sending partial rows.

- **Blanket Python `date`/`datetime` serialisation** — All columns containing
  `datetime.date`, `datetime.datetime`, or `pandas.Timestamp` values are now
  converted to `"YYYY-MM-DD"` or `"YYYY-MM-DDTHH:MM:SS.000Z"` strings before
  JSON serialisation, eliminating `Object of type date is not JSON serializable`
  errors for `script_expiries`.

### Known issues / pending (session 2)

- `Expiry__c` fields were created with **lowercase API names** (`source__c`,
  `expiry_date__c`) by the prior partial deploy; the `TABLE_COLUMN_TO_FIELD_MAP`
  entries use Title-Case names. Needs reconciliation before `script_expiries` rows
  can be fully inserted with all columns populated.
- `415 Unsupported Media Type` on the composite/sobjects REST call — header or
  session object mismatch to be resolved next session.
- `backtest_v4.py` — still uses PostgreSQL; conversion to Salesforce pending.
- `app.py` — 5+ routes still read/write PostgreSQL; conversion pending.
- `templates/backtest_runner.html` — still references PostgreSQL in heading; pending.
