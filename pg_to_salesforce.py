#!/usr/bin/env python3
"""
pg_to_salesforce.py

Migrates ALL PostgreSQL tables in this project to Salesforce Custom Objects.

Steps
─────
  1. Introspect every table in the configured PostgreSQL database (schema + rows)
  2. Map each PostgreSQL column type to the correct Salesforce field type
  3. Build a Salesforce Metadata API deployment ZIP and deploy it
     → creates one Custom Object per table with all mapped Custom Fields
  4. Insert all data rows via the Salesforce Bulk API

Required environment variables (add to .env or Render Environment Variables)
─────────────────────────────────────────────────────────────────────────────
  POSTGRES_URI          – PostgreSQL connection string
                          e.g. postgresql://user:pass@host:5432/dbname

  SF_USERNAME           – Salesforce login e-mail
  SF_PASSWORD           – Salesforce password
  SF_SECURITY_TOKEN     – Salesforce security token (leave blank for IP-trusted orgs)
  SF_DOMAIN             – 'login' for production, 'test' for sandbox  (default: login)
  SF_API_VERSION        – Salesforce API version to use               (default: 59.0)

Usage
─────
  pip install simple-salesforce        # one-time
  python pg_to_salesforce.py

Notes
─────
  • Primary-key columns (SERIAL / auto-increment) are skipped; Salesforce
    generates its own Id automatically.
  • Column / table names are sanitised to valid Salesforce API names.
  • Custom Object API names get the mandatory __c suffix automatically.
  • A Name (AutoNumber) field is added to every Custom Object as the
    standard Name field required by Salesforce.
  • Re-running the script is safe: the metadata deploy is idempotent
    (existing objects/fields are updated, not duplicated), but the data
    insert will create additional records — truncate the Salesforce objects
    first if you need a clean reload.
"""

import io
import logging
import os
import re
import tempfile
import time
import zipfile

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from sqlalchemy import types as sa_types

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
POSTGRES_URI   = os.getenv("POSTGRES_URI")
SF_USERNAME    = os.getenv("SF_USERNAME")
SF_PASSWORD    = os.getenv("SF_PASSWORD")
SF_TOKEN       = os.getenv("SF_SECURITY_TOKEN", "")
SF_DOMAIN      = os.getenv("SF_DOMAIN", "login")
SF_API_VERSION = os.getenv("SF_API_VERSION", "59.0")

# Optional explicit table -> Salesforce custom object mapping (API base, no __c)
# This lets the migration target pre-created objects instead of generating
# table-name-based objects.
TABLE_TO_OBJECT_API_BASE = {
    "v4_backtest_results": "Trade",
    "historical_backtests": "historical_backtests",
    "script_expiries": "Expiry",
}

# Explicit per-table source columns to ignore completely.
# Ignored columns are not created as Salesforce fields and are not migrated.
TABLE_COLUMN_EXCLUDE = {
    "historical_backtests": {"Entry_Price", "Exit_Price"},
    "v4_backtest_results": {"Entry_Price", "Exit_Price"},
}

# Explicit source-column → Salesforce field API name for every column in every table.
# Columns listed here drive BOTH the metadata deploy (field creation) AND the data insert.
# All values must end with __c (custom fields); standard fields like Name are not settable.
TABLE_COLUMN_TO_FIELD_MAP = {
    # ── v4_backtest_results  →  Trade__c ─────────────────────────────────────
  
    # ── historical_backtests  →  historical_backtests__c ─────────────────────
    "historical_backtests": {
        "Run_Date":      "Run_Date__c",              # existing
        "Strategy_Name": "Strategy_Name__c",         # existing
        "PNL":           "Total_PNL__c",             # existing
        "ROI%":          "Total_Return_Percentage__c",# existing
        "Roi%":          "Total_Return_Percentage__c",# alias
        "Date":          "Trade_Date__c",            # NEW
        "Entry_Time":    "Entry_Time__c",            # NEW
        "Exit_Time":     "Exit_Time__c",             # NEW
        "Option_Type":   "Option_Type__c",           # NEW
        "Action":        "Action__c",                # NEW
        "Qty":           "Qty__c",                   # NEW
        "Buy_Price":     "Buy_Price__c",             # NEW
        "Peak_Price":    "Peak_Price__c",            # NEW
        "Sell_Price":    "Sell_Price__c",            # NEW
        "Reason":        "Reason__c",                # NEW
        "Win":           "Win__c",                   # NEW
        "Capital_ROI%":  "Capital_ROI_Pct__c",       # NEW
        "Capital Roi%":  "Capital_ROI_Pct__c",       # alias
        "Run_Mode":      "Run_Mode__c",              # NEW
        "Strike":        "Strike__c",                # NEW
        "Type":          "Trade_Type__c",            # NEW
        "PnL_INR":       "PnL_INR__c",              # NEW
        "Parameters":    "Parameters__c",            # NEW
    },
    # ── script_expiries  →  Expiry__c ────────────────────────────────────────
    "script_expiries": {
        "expiry_date": "Expiry_Date__c", # existing
        "source":      "Source__c",      # existing
        "script_name": "Script_Name__c", # NEW
        "day_label":   "Day_Label__c",   # NEW
        "fetched_at":  "Fetched_At__c",  # NEW
    },
}

# Explicit PostgreSQL column -> Salesforce field type overrides.
# Use this when source DB column types are too generic (e.g., text) but
# Salesforce fields must be strongly typed.
TABLE_COLUMN_TYPE_OVERRIDES = {
    "historical_backtests": {
        "Date": {"type": "Date"},
        "Buy_Price": {"type": "Number", "precision": 18, "scale": 2},
        "Sell_Price": {"type": "Number", "precision": 18, "scale": 2},
        "PnL_INR": {"type": "Number", "precision": 18, "scale": 2},
        "ROI%": {"type": "Number", "precision": 16, "scale": 2},
        "Roi%": {"type": "Number", "precision": 16, "scale": 2},
        "Capital_ROI%": {"type": "Number", "precision": 16, "scale": 2},
        "Capital Roi%": {"type": "Number", "precision": 16, "scale": 2},
        "Run_Date": {"type": "Date"},
        "Run_Mode": {"type": "Text", "length": "255"},
        "Strategy_Name": {"type": "Text"},
        "Strike": {"type": "Text"},
        "Entry_Time": {"type": "DateTime"},
        "Exit_Time": {"type": "DateTime"},
        "Action": {"type": "Text", "length": "255"},
        "Option_Type": {"type": "Text", "length": "255"},
        "Type": {"type": "Text"},
    },
    "v4_backtest_results": {
        "Date": {"type": "Date"},
        "Entry_Time": {"type": "DateTime"},
        "Exit_Time": {"type": "DateTime"},
    },
}

# Salesforce limits: API name base ≤ 40 chars (we leave 2 for __c → max 38 here)
_MAX_API_BASE = 38
# Salesforce field label max = 80 chars, object label max = 40 chars
_MAX_LABEL = 40


# ── PostgreSQL → Salesforce type mapping ──────────────────────────────────────

def _pg_col_to_sf(col_type) -> dict:
    """
    Return a Salesforce Metadata API field descriptor dict for a given
    SQLAlchemy column type.

    Keys returned vary by field type (matching Salesforce CustomField XML):
      type, length, precision, scale, defaultValue, visibleLines
    """
    t = col_type

    # Integer / BigInteger / SmallInteger
    if isinstance(t, (sa_types.Integer, sa_types.BigInteger, sa_types.SmallInteger)):
        return {"type": "Number", "precision": 18, "scale": 0}

    # Numeric / Float / Decimal
    if isinstance(t, (sa_types.Numeric, sa_types.Float)):
        return {"type": "Number", "precision": 18, "scale": 2}

    # Boolean → Checkbox
    if isinstance(t, sa_types.Boolean):
        return {"type": "Checkbox", "defaultValue": "false"}

    # Date
    if isinstance(t, sa_types.Date):
        return {"type": "Date"}

    # DateTime / Timestamp
    if isinstance(t, (sa_types.DateTime, sa_types.TIMESTAMP)):
        return {"type": "DateTime"}

    # VARCHAR(n): ≤ 255 → Text; larger → LongTextArea
    if isinstance(t, (sa_types.String, sa_types.VARCHAR, sa_types.CHAR)):
        length = getattr(t, "length", None)
        if length and int(length) <= 255:
            return {"type": "Text", "length": str(length)}
        return {"type": "LongTextArea", "length": "32768", "visibleLines": "5"}

    # TEXT → Text(255) instead of LongTextArea
    if isinstance(t, sa_types.Text):
        return {"type": "Text", "length": "255"}

    # Fallback: treat unknown types as Text(255)
    return {"type": "Text", "length": "255"}


# ── Name sanitisation ─────────────────────────────────────────────────────────

def _to_api_name(raw: str) -> str:
    """
    Convert an arbitrary string (table name or column name) to a valid
    Salesforce API name base (no __c suffix).

    Rules applied:
      • Replace any non-alphanumeric / non-underscore char with _
      • Must start with a letter — prepend 'F_' if needed
      • Collapse consecutive underscores to one; strip leading/trailing _
      • Truncate to _MAX_API_BASE characters
    """
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw)
    if name and not name[0].isalpha():
        name = "F_" + name
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:_MAX_API_BASE]


def _to_label(raw: str) -> str:
    """Human-readable label, max _MAX_LABEL chars."""
    return raw.replace("_", " ").title()[:_MAX_LABEL]


# ── Metadata XML builders ─────────────────────────────────────────────────────

_PACKAGE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<Package xmlns="http://soap.sforce.com/2006/04/metadata">
    <types>
{members}        <name>CustomObject</name>
    </types>
    <types>
        <members>AntiGravity_Access</members>
        <name>PermissionSet</name>
    </types>
    <version>{version}</version>
</Package>
"""

_STRATEGY_OBJECT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>Strategy</label>
    <pluralLabel>Strategies</pluralLabel>
    <nameField>
        <label>Name</label>
        <type>AutoNumber</type>
        <displayFormat>STR-{0000000}</displayFormat>
    </nameField>
    <deploymentStatus>Deployed</deploymentStatus>
    <sharingModel>ReadWrite</sharingModel>
    <fields>
        <fullName>Strategy_Name__c</fullName>
        <label>Strategy Name</label>
        <type>Text</type>
        <length>255</length>
        <required>false</required>
    </fields>
    <fields>
        <fullName>Description__c</fullName>
        <label>Description</label>
        <type>LongTextArea</type>
        <length>32768</length>
        <visibleLines>5</visibleLines>
        <required>false</required>
    </fields>
</CustomObject>
"""

def _build_package_xml(object_full_api_names: list, version: str) -> str:
    members = "".join(
        f"        <members>{n}</members>\n" for n in object_full_api_names
    )
    return _PACKAGE_XML.format(members=members, version=version)


PERMSET_NAME = "AntiGravity_Access"


def _build_permissionset_xml(schema: dict) -> str:
    """Build a PermissionSet XML that grants read+edit FLS for every custom field."""
    field_perms = []
    obj_perms = []

    for tbl, meta in schema.items():
        obj_full_api = f"{meta['obj_api']}__c"
        explicit_map = TABLE_COLUMN_TO_FIELD_MAP.get(tbl, {})
        for col in meta["columns"]:
            raw = col["raw_name"]
            field_api = explicit_map.get(raw, col["api_name"] + "__c")
            if not field_api.endswith("__c"):
                continue  # skip standard fields like Name
            field_perms.append(
                f"    <fieldPermissions>\n"
                f"        <field>{obj_full_api}.{field_api}</field>\n"
                f"        <readable>true</readable>\n"
                f"        <editable>true</editable>\n"
                f"    </fieldPermissions>"
            )
        obj_perms.append(
            f"    <objectPermissions>\n"
            f"        <object>{obj_full_api}</object>\n"
            f"        <allowCreate>true</allowCreate>\n"
            f"        <allowDelete>true</allowDelete>\n"
            f"        <allowEdit>true</allowEdit>\n"
            f"        <allowRead>true</allowRead>\n"
            f"        <modifyAllRecords>true</modifyAllRecords>\n"
            f"        <viewAllRecords>true</viewAllRecords>\n"
            f"    </objectPermissions>"
        )

    # Manual metadata additions not represented by PostgreSQL tables.
    field_perms.append(
        "    <fieldPermissions>\n"
        "        <field>Strategy__c.Strategy_Name__c</field>\n"
        "        <readable>true</readable>\n"
        "        <editable>true</editable>\n"
        "    </fieldPermissions>"
    )
    field_perms.append(
        "    <fieldPermissions>\n"
        "        <field>Strategy__c.Description__c</field>\n"
        "        <readable>true</readable>\n"
        "        <editable>true</editable>\n"
        "    </fieldPermissions>"
    )
    field_perms.append(
        "    <fieldPermissions>\n"
        "        <field>historical_backtests__c.Strategy__c</field>\n"
        "        <readable>true</readable>\n"
        "        <editable>true</editable>\n"
        "    </fieldPermissions>"
    )

    obj_perms.append(
        "    <objectPermissions>\n"
        "        <object>Strategy__c</object>\n"
        "        <allowCreate>true</allowCreate>\n"
        "        <allowDelete>true</allowDelete>\n"
        "        <allowEdit>true</allowEdit>\n"
        "        <allowRead>true</allowRead>\n"
        "        <modifyAllRecords>true</modifyAllRecords>\n"
        "        <viewAllRecords>true</viewAllRecords>\n"
        "    </objectPermissions>"
    )

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<PermissionSet xmlns="http://soap.sforce.com/2006/04/metadata">',
        '    <label>AntiGravity Access</label>',
        '    <description>Auto-generated FLS grant for all AntiGravity custom fields</description>',
    ]
    parts.extend(field_perms)
    parts.extend(obj_perms)
    parts.append('</PermissionSet>')
    return "\n".join(parts)


def _build_object_xml(obj_api_base: str, label: str, plural_label: str,
                      columns: list) -> str:
    """
    Build the full Salesforce CustomObject metadata XML string for one table.

    Parameters
    ──────────
    obj_api_base : API name without __c  (e.g. 'historical_backtests')
    label        : human-readable singular label
    plural_label : human-readable plural label
    columns      : list of column dicts produced by _collect_schema()
    """
    fields_xml_parts = []

    for col in columns:
        sf_fd   = col["sf"]
        api     = col["api_name"]       # no __c yet
        lbl     = col["label"]
        sf_type = sf_fd["type"]

        parts = [
            f"    <fields>",
            f"        <fullName>{api}__c</fullName>",
            f"        <label>{lbl}</label>",
            f"        <type>{sf_type}</type>",
        ]

        if sf_type == "Text":
            parts.append(f"        <length>{sf_fd.get('length', '255')}</length>")
        elif sf_type == "Number":
            parts.append(f"        <precision>{sf_fd.get('precision', 18)}</precision>")
            parts.append(f"        <scale>{sf_fd.get('scale', 0)}</scale>")
        elif sf_type == "LongTextArea":
            parts.append(f"        <length>{sf_fd.get('length', '32768')}</length>")
            parts.append(f"        <visibleLines>{sf_fd.get('visibleLines', '5')}</visibleLines>")
        elif sf_type == "Checkbox":
            parts.append(f"        <defaultValue>{sf_fd.get('defaultValue', 'false')}</defaultValue>")
        # Date / DateTime need no extra attributes

        parts.append(f"        <required>false</required>")
        parts.append(f"    </fields>")
        fields_xml_parts.append("\n".join(parts))

    fields_block = "\n".join(fields_xml_parts)
    strategy_lookup_block = ""
    if obj_api_base == "historical_backtests":
        strategy_lookup_block = """
    <fields>
        <fullName>Strategy__c</fullName>
        <label>Strategy</label>
        <type>Lookup</type>
        <referenceTo>Strategy__c</referenceTo>
        <relationshipLabel>Historical Backtests</relationshipLabel>
        <relationshipName>Historical_Backtests</relationshipName>
        <required>false</required>
    </fields>"""

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<CustomObject xmlns="http://soap.sforce.com/2006/04/metadata">
    <label>{label}</label>
    <pluralLabel>{plural_label}</pluralLabel>
    <nameField>
        <label>Name</label>
        <type>AutoNumber</type>
        <displayFormat>REC-{{0000000}}</displayFormat>
    </nameField>
    <deploymentStatus>Deployed</deploymentStatus>
    <sharingModel>ReadWrite</sharingModel>
{fields_block}
{strategy_lookup_block}
</CustomObject>
"""


# ── Schema collection ─────────────────────────────────────────────────────────

def _collect_schema(engine) -> dict:
    """
    Reflect all tables from PostgreSQL and return a dict keyed by table name.

    Each value is:
      {
        "obj_api":  str,   # Salesforce custom object API name base (no __c)
        "label":    str,
        "plural":   str,
        "columns":  [{"raw_name", "api_name", "label", "sf": {type descriptor}}]
      }
    """
    insp = inspect(engine)
    table_names = insp.get_table_names()
    logger.info("PostgreSQL tables found: %s", table_names)

    schema: dict = {}
    for tbl in table_names:
        pk_cols = set(
            insp.get_pk_constraint(tbl).get("constrained_columns", [])
        )
        excluded_cols = TABLE_COLUMN_EXCLUDE.get(tbl, set())
        columns = []
        tbl_field_map = TABLE_COLUMN_TO_FIELD_MAP.get(tbl, {})
        tbl_type_overrides = TABLE_COLUMN_TYPE_OVERRIDES.get(tbl, {})
        for col in insp.get_columns(tbl):
            if col["name"] in pk_cols:
                logger.debug("  Skipping PK column '%s.%s'", tbl, col["name"])
                continue
            if col["name"] in excluded_cols:
                logger.debug("  Skipping excluded column '%s.%s'", tbl, col["name"])
                continue
            # Derive Salesforce api_name (without __c) from explicit mapping when
            # available; otherwise auto-generate from the column name.
            mapped_field = tbl_field_map.get(col["name"])
            if mapped_field and mapped_field.endswith("__c"):
                api_name = mapped_field[:-3]  # strip trailing __c
            else:
                api_name = _to_api_name(col["name"])
            columns.append({
                "raw_name": col["name"],
                "api_name": api_name,
                "label":    _to_label(col["name"])[:_MAX_LABEL],
                "sf":       tbl_type_overrides.get(col["name"], _pg_col_to_sf(col["type"])),
            })

        target_obj_api = TABLE_TO_OBJECT_API_BASE.get(tbl, _to_api_name(tbl))

        schema[tbl] = {
            "obj_api": target_obj_api,
            "label":   _to_label(tbl),
            "plural":  _to_label(tbl) + "s",
            "columns": columns,
        }

        logger.info(
            "  Table %-30s → SF object %-30s  (%d fields)",
            tbl,
            schema[tbl]["obj_api"] + "__c",
            len(columns),
        )

    return schema


# ── Metadata deployment ───────────────────────────────────────────────────────

def _build_deployment_zip(schema: dict, api_version: str) -> bytes:
    """Return the bytes of the Metadata deployment ZIP."""
    buf = io.BytesIO()
    full_api_names = []

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Manual object required by business model.
        zf.writestr("objects/Strategy__c.object", _STRATEGY_OBJECT_XML)
        full_api_names.append("Strategy__c")
        logger.info("  Packaged → objects/Strategy__c.object")

        for tbl, meta in schema.items():
            obj_api_base = meta["obj_api"]
            full_api     = f"{obj_api_base}__c"
            full_api_names.append(full_api)

            obj_xml = _build_object_xml(
                obj_api_base  = obj_api_base,
                label         = meta["label"],
                plural_label  = meta["plural"],
                columns       = meta["columns"],
            )
            zf.writestr(f"objects/{full_api}.object", obj_xml)
            logger.info("  Packaged → objects/%s.object", full_api)

        # Include a PermissionSet so newly deployed fields get FLS automatically.
        permset_xml = _build_permissionset_xml(schema)
        zf.writestr(f"permissionsets/{PERMSET_NAME}.permissionset", permset_xml)
        logger.info("  Packaged → permissionsets/%s.permissionset", PERMSET_NAME)

        pkg_xml = _build_package_xml(full_api_names, api_version)
        zf.writestr("package.xml", pkg_xml)

    return buf.getvalue()


def _deploy_and_wait(sf, zip_bytes: bytes) -> None:
    """Deploy the metadata ZIP to Salesforce and poll until done."""
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp_zip:
        tmp_zip.write(zip_bytes)
        zip_path = tmp_zip.name

    logger.info("Submitting Metadata deployment …")
    try:
        result = sf.mdapi.deploy(zip_path, sandbox=(SF_DOMAIN == "test"), options={
            "allowMissingFiles": False,
            "autoUpdatePackage":  False,
            "checkOnly":          False,
            "ignoreWarnings":     True,
            "rollbackOnError":    True,
            "runTests":           [],
            "singlePackage":      True,
            "testLevel":          "NoTestRun",
        })
    finally:
        try:
            os.remove(zip_path)
        except OSError:
            pass

    if isinstance(result, tuple):
        deploy_id = result[0] if result else None
    elif isinstance(result, dict):
        deploy_id = result.get("id") or result.get("asyncProcessId")
    else:
        deploy_id = getattr(result, "id", None) or getattr(result, "asyncProcessId", None)

    if not deploy_id:
        raise RuntimeError(f"Could not extract deployment id from response: {result!r}")

    logger.info("Deployment id: %s — polling for completion …", deploy_id)

    failures = []
    while True:
        status = sf.mdapi.check_deploy_status(deploy_id, include_details=True)
        if isinstance(status, tuple):
            state = status[0] if len(status) > 0 else "Unknown"
            deployment_detail = status[2] if len(status) > 2 and isinstance(status[2], dict) else {}
            failures = deployment_detail.get("errors", []) or []
        else:
            state = status.get("status") if isinstance(status, dict) else getattr(status, "status", "Unknown")
        logger.info("  status: %s", state)
        if state in ("Succeeded", "Failed", "Cancelled"):
            break
        time.sleep(5)

    if state != "Succeeded":
        try:
            for f in (failures if isinstance(failures, list) else [failures]):
                if isinstance(f, dict):
                    fn = f.get("fullName") or f.get("file") or "?"
                    prb = f.get("problem") or f.get("message") or "?"
                else:
                    fn = getattr(f, "fullName", "?")
                    prb = getattr(f, "problem", "?")
                logger.error("  Component failure: %s — %s", fn, prb)
        except Exception:
            pass
        raise RuntimeError(
            f"Metadata deployment FAILED with status '{state}'. "
            "Check Salesforce Setup > Deployment Status for details."
        )

    logger.info("Metadata deployment SUCCEEDED.")


# ── Permission Set assignment ─────────────────────────────────────────────────

def _ensure_permissionset_assigned(sf) -> None:
    """Assign the AntiGravity_Access PermissionSet to the running user if needed.

    The PermissionSet grants FLS read+edit on all custom fields deployed by
    _build_deployment_zip so that describe() and REST inserts can see them.
    """
    try:
        ps_result = sf.query(
            f"SELECT Id FROM PermissionSet WHERE Name='{PERMSET_NAME}' LIMIT 1"
        )
        if ps_result["totalSize"] == 0:
            logger.warning("PermissionSet '%s' not found — FLS may be missing.", PERMSET_NAME)
            return
        permset_id = ps_result["records"][0]["Id"]

        user_result = sf.query(
            f"SELECT Id FROM User WHERE Username='{SF_USERNAME}' LIMIT 1"
        )
        if user_result["totalSize"] == 0:
            logger.warning("Current SF user not found — cannot assign PermissionSet.")
            return
        user_id = user_result["records"][0]["Id"]

        existing = sf.query(
            f"SELECT Id FROM PermissionSetAssignment "
            f"WHERE AssigneeId='{user_id}' AND PermissionSetId='{permset_id}' LIMIT 1"
        )
        if existing["totalSize"] > 0:
            logger.info("PermissionSet '%s' already assigned to user.", PERMSET_NAME)
            return

        sf.PermissionSetAssignment.create({
            "AssigneeId": user_id,
            "PermissionSetId": permset_id,
        })
        logger.info("PermissionSet '%s' assigned — FLS granted for all new fields.", PERMSET_NAME)
    except Exception as exc:
        logger.warning("Could not assign PermissionSet: %s", exc)


# ── Data migration ────────────────────────────────────────────────────────────

def _migrate_data(sf, engine, schema: dict, chunk_size: int = 200) -> None:
    """Read every row from every PostgreSQL table and bulk-insert into Salesforce."""
    for tbl, meta in schema.items():
        obj_full_api = f"{meta['obj_api']}__c"
        logger.info("Migrating data  %s  →  %s", tbl, obj_full_api)

        try:
            sf_desc = sf.__getattr__(obj_full_api).describe()
            sf_fields = {f.get("name") for f in sf_desc.get("fields", [])}
            sf_field_types = {
                f.get("name"): f.get("type")
                for f in sf_desc.get("fields", [])
                if f.get("name")
            }
        except Exception as exc:
            logger.error("  Failed to describe Salesforce object '%s': %s", obj_full_api, exc)
            continue

        try:
            df = pd.read_sql(f'SELECT * FROM "{tbl}"', con=engine)
        except Exception as exc:
            logger.error("  Failed to read table '%s': %s", tbl, exc)
            continue

        if df.empty:
            logger.info("  No rows to migrate.")
            continue

        # Build column rename map raw_name -> Salesforce field API name.
        # TABLE_COLUMN_TO_FIELD_MAP is the authoritative source; for any column
        # not listed there fall back to the auto-generated api_name__c.
        # We no longer gate on sf_fields so freshly-deployed fields are included.
        explicit_map = TABLE_COLUMN_TO_FIELD_MAP.get(tbl, {})
        col_map = {}
        for c in meta["columns"]:
            raw_name = c["raw_name"]
            if raw_name in explicit_map:
                target = explicit_map[raw_name]
                if target.endswith("__c"):   # custom field — include
                    col_map[raw_name] = target
                # standard fields (like Name) are skipped intentionally
            else:
                col_map[raw_name] = c["api_name"] + "__c"

        if not col_map:
            logger.warning(
                "  No mapped fields for table '%s'. Skipping table.", tbl,
            )
            continue

        # Keep only columns that were mapped (drops PK columns)
        keep_cols = [c for c in df.columns if c in col_map]
        df = df[keep_cols].rename(columns=col_map)

        # Replace NaN / NaT with None for JSON serialisation
        df = df.where(pd.notnull(df), None)

        # Convert EVERY column that still contains Python date/datetime/Timestamp
        # objects or pandas Timestamps to safe string forms so the Bulk API JSON
        # serialiser never hits "Object of type date is not JSON serializable".
        import datetime as _dt
        for sf_col in list(df.columns):
            target_type = sf_field_types.get(sf_col, "")
            sample = df[sf_col].dropna().iloc[0] if not df[sf_col].dropna().empty else None
            is_dt_like = (
                isinstance(sample, (pd.Timestamp, _dt.datetime)) or
                target_type == "datetime"
            )
            is_date_like = (
                isinstance(sample, _dt.date) and not isinstance(sample, _dt.datetime) or
                target_type == "date"
            )
            if is_dt_like:
                df[sf_col] = pd.to_datetime(df[sf_col], errors="coerce") \
                               .dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
                df[sf_col] = df[sf_col].where(df[sf_col] != "NaT", None)
            elif is_date_like:
                df[sf_col] = pd.to_datetime(df[sf_col], errors="coerce") \
                               .dt.strftime("%Y-%m-%d")
                df[sf_col] = df[sf_col].where(df[sf_col] != "NaT", None)
            elif target_type == "boolean":
                def _to_bool(v):
                    if v is None:
                        return None
                    if isinstance(v, bool):
                        return v
                    if isinstance(v, (int, float)):
                        return bool(v)
                    if isinstance(v, str):
                        t = v.strip().lower()
                        if t in ("true", "t", "1", "yes", "y"):
                            return True
                        if t in ("false", "f", "0", "no", "n"):
                            return False
                    return None
                df[sf_col] = df[sf_col].apply(_to_bool)
            elif target_type in ("double", "currency", "percent", "int"):
                nums = pd.to_numeric(df[sf_col], errors="coerce")
                if target_type == "int":
                    df[sf_col] = nums.round().astype("Int64").astype(object)
                else:
                    df[sf_col] = nums.astype(object)
                df[sf_col] = df[sf_col].where(pd.notnull(df[sf_col]), None)
            # Convert any remaining non-primitive types to str as a safety net
            elif sample is not None and not isinstance(sample, (str, int, float, bool)):
                df[sf_col] = df[sf_col].astype(str).where(df[sf_col].notna(), None)

        records = df.to_dict(orient="records")
        logger.info("  %d rows to insert …", len(records))

        sf_obj = sf.__getattr__(obj_full_api)
        total_ok = 0
        total_fail = 0

        # Verify the target fields exist via REST describe before inserting.
        # This catches Bulk API / metadata propagation timing issues early.
        live_fields = {f.get("name") for f in sf_obj.describe().get("fields", [])}
        missing_fields = {v for v in col_map.values() if v not in live_fields}
        if missing_fields:
            logger.warning(
                "  Fields not yet visible via REST describe — %s. Retrying once after 15 s ...",
                missing_fields,
            )
            time.sleep(15)
            live_fields = {f.get("name") for f in sf_obj.describe().get("fields", [])}
            missing_fields = {v for v in col_map.values() if v not in live_fields}
            if missing_fields:
                logger.error(
                    "  Fields still missing after wait: %s. Skipping table '%s'.",
                    missing_fields, tbl,
                )
                continue

        # Use per-record REST create for reliability across org API quirks.
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            try:
                ok = 0
                fail = 0
                first_error = None
                for rec in chunk:
                    try:
                        res = sf_obj.create(rec)
                        if res.get("success"):
                            ok += 1
                        else:
                            fail += 1
                            if first_error is None:
                                first_error = res.get("errors", "Unknown error")
                    except Exception as row_exc:
                        fail += 1
                        if first_error is None:
                            first_error = str(row_exc)

                total_ok   += ok
                total_fail += fail
                if fail:
                    logger.warning(
                        "  Chunk %d‒%d: %d OK, %d FAILED — first error: %s",
                        i + 1, i + len(chunk), ok, fail,
                        first_error or "?",
                    )
                else:
                    logger.info(
                        "  Chunk %d‒%d: %d rows inserted OK",
                        i + 1, i + len(chunk), ok,
                    )
            except Exception as exc:
                logger.error(
                    "  REST insert failed for chunk %d‒%d: %s",
                    i + 1, i + len(chunk), exc,
                )

        logger.info(
            "  Table '%s' done — %d inserted, %d failed.", tbl, total_ok, total_fail
        )


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    # ── Validate required config ──────────────────────────────────────────
    missing = [v for v in ("POSTGRES_URI", "SF_USERNAME", "SF_PASSWORD")
               if not os.getenv(v)]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variable(s): {', '.join(missing)}\n"
            "Set them in your .env file or Render Environment Variables."
        )

    # ── Connect to PostgreSQL ─────────────────────────────────────────────
    logger.info("Connecting to PostgreSQL …")
    engine = create_engine(POSTGRES_URI)

    # ── Collect schema ────────────────────────────────────────────────────
    schema = _collect_schema(engine)
    if not schema:
        logger.warning("No tables found in PostgreSQL. Nothing to migrate.")
        return

    # ── Connect to Salesforce ─────────────────────────────────────────────
    try:
        from simple_salesforce import Salesforce
    except ImportError:
        raise ImportError(
            "simple-salesforce is not installed.\n"
            "Run:  pip install simple-salesforce"
        )

    logger.info(
        "Connecting to Salesforce  username=%s  domain=%s  api_version=%s …",
        SF_USERNAME, SF_DOMAIN, SF_API_VERSION,
    )
    sf = Salesforce(
        username      = SF_USERNAME,
        password      = SF_PASSWORD,
        security_token = SF_TOKEN,
        domain        = SF_DOMAIN,
        version       = SF_API_VERSION,
    )
    logger.info("Salesforce connected — instance: %s", sf.sf_instance)

    # ── Build & deploy Metadata ZIP ───────────────────────────────────────
    skip_metadata_deploy = os.getenv("SF_SKIP_METADATA_DEPLOY", "false").lower() in {
        "1", "true", "yes", "y", "on"
    }
    if skip_metadata_deploy:
        logger.info("Skipping metadata deployment (SF_SKIP_METADATA_DEPLOY=true).")
    else:
        logger.info(
            "Building Metadata deployment package for %d object(s) …", len(schema)
        )
        zip_bytes = _build_deployment_zip(schema, SF_API_VERSION)
        _deploy_and_wait(sf, zip_bytes)
        _ensure_permissionset_assigned(sf)

    # ── Migrate data ──────────────────────────────────────────────────────
    logger.info("Starting data migration …")
    _migrate_data(sf, engine, schema)

    logger.info("=" * 60)
    logger.info("Migration complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
