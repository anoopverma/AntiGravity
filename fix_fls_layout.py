#!/usr/bin/env python3
"""
fix_fls_layout.py

Grants FLS (Read + Edit) on every custom field of historical_backtests__c
to ALL profiles in the org, and adds every field to the object's default
page layout — so the fields become visible in the UI immediately.

Usage:
    python fix_fls_layout.py

Required env vars (same as pg_to_salesforce.py):
    SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN, SF_DOMAIN, SF_API_VERSION
"""

import io
import logging
import os
import tempfile
import time
import zipfile

from dotenv import load_dotenv
from simple_salesforce import Salesforce

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

OBJECT_API_NAME = "historical_backtests__c"
SF_USERNAME    = os.getenv("SF_USERNAME")
SF_PASSWORD    = os.getenv("SF_PASSWORD")
SF_TOKEN       = os.getenv("SF_SECURITY_TOKEN", "")
SF_DOMAIN      = os.getenv("SF_DOMAIN", "login")
SF_API_VERSION = os.getenv("SF_API_VERSION", "59.0")


# ── helpers ───────────────────────────────────────────────────────────────────

def _xml(tag: str, content: str = "", attrs: str = "") -> str:
    if attrs:
        return f"<{tag} {attrs}>{content}</{tag}>"
    return f"<{tag}>{content}</{tag}>"


def build_zip(
    object_api_name: str,
    field_names: list[str],
    profile_names: list[str],
    layout_label: str,
) -> bytes:
    """
    Build a Metadata API deployment ZIP that:
      1. Sets fieldPermissions (read + edit) for every field on every profile.
      2. Rebuilds the page layout with every field visible in a single section.
    """
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:

        # ── package.xml ──────────────────────────────────────────────────────
        profile_members = "\n".join(
            f"        <members>{p}</members>" for p in profile_names
        )
        layout_member = f"{object_api_name}-{layout_label}"
        package_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Package xmlns="http://soap.sforce.com/2006/04/metadata">
    <types>
{profile_members}
        <name>Profile</name>
    </types>
    <types>
        <members>{layout_member}</members>
        <name>Layout</name>
    </types>
    <version>{SF_API_VERSION}</version>
</Package>"""
        zf.writestr("package.xml", package_xml)

        # ── Profile files — FLS ───────────────────────────────────────────────
        for profile in profile_names:
            perms = "\n".join(
                f"""    <fieldPermissions>
        <field>{object_api_name}.{f}</field>
        <editable>true</editable>
        <readable>true</readable>
    </fieldPermissions>"""
                for f in field_names
            )
            profile_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Profile xmlns="http://soap.sforce.com/2006/04/metadata">
{perms}
</Profile>"""
            zf.writestr(f"profiles/{profile}.profile", profile_xml)

        # ── Page layout — one section with all fields ─────────────────────────
        layout_items = "\n".join(
            f"""            <layoutItems>
                <behavior>Edit</behavior>
                <field>{f}</field>
            </layoutItems>"""
            for f in field_names
        )
        layout_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<Layout xmlns="http://soap.sforce.com/2006/04/metadata">
    <layoutSections>
        <customLabel>false</customLabel>
        <detailHeading>true</detailHeading>
        <editHeading>true</editHeading>
        <label>Backtest Fields</label>
        <layoutColumns>
{layout_items}
        </layoutColumns>
        <style>OneColumn</style>
    </layoutSections>
    <showEmailCheckbox>false</showEmailCheckbox>
    <showHighlightsPanel>false</showHighlightsPanel>
    <showInteractionLogPanel>false</showInteractionLogPanel>
    <showRunAssignmentRulesCheckbox>false</showRunAssignmentRulesCheckbox>
    <showSubmitAndAttachButton>false</showSubmitAndAttachButton>
</Layout>"""
        zf.writestr(f"layouts/{layout_member}.layout", layout_xml)

    return buf.getvalue()


def deploy_zip(sf: Salesforce, zip_bytes: bytes) -> bool:
    """Deploy a metadata zip, poll until done, return True on success."""
    is_sandbox = SF_DOMAIN == "test"
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp.write(zip_bytes)
        zip_path = tmp.name

    try:
        async_id, initial_state = sf.mdapi.deploy(
            zipfile=zip_path,
            sandbox=is_sandbox,
            options={"checkOnly": False},
        )
    finally:
        if os.path.exists(zip_path):
            os.unlink(zip_path)
    logger.info("Deploy job submitted: %s  (initial state: %s) — polling...", async_id, initial_state)
    for _ in range(60):
        time.sleep(5)
        state, state_detail, deployment_detail, unit_test_detail = sf.mdapi.check_deploy_status(async_id)
        logger.info("  status: %s", state)
        if state == "Succeeded":
            logger.info("Deploy SUCCEEDED.")
            return True
        if state in ("Failed", "Canceled"):
            logger.error(
                "Deploy %s: %s | deployment=%s | tests=%s",
                state,
                state_detail,
                deployment_detail,
                unit_test_detail,
            )
            return False

    logger.error("Deploy timed out after 5 minutes.")
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    logger.info("Connecting to Salesforce (domain=%s)...", SF_DOMAIN)
    sf = Salesforce(
        username=SF_USERNAME,
        password=SF_PASSWORD,
        security_token=SF_TOKEN,
        domain=SF_DOMAIN,
        version=SF_API_VERSION,
    )
    logger.info("Connected.")

    # 1. Discover all custom fields on the object
    obj_obj = getattr(sf, OBJECT_API_NAME.replace("__c", "__c"))
    describe = obj_obj.describe()
    custom_fields = [
        f["name"] for f in describe["fields"]
        if f["name"].endswith("__c")
    ]
    logger.info("Found %d custom fields on %s:", len(custom_fields), OBJECT_API_NAME)
    for f in custom_fields:
        logger.info("  %s", f)

    if not custom_fields:
        logger.warning("No custom fields found — nothing to do.")
        return

    # 2. Discover all profiles
    profiles_result = sf.query_all("SELECT Name FROM Profile")
    profile_names = sorted({r["Name"] for r in profiles_result["records"]})
    logger.info("Found %d profiles.", len(profile_names))

    # 3. Target the default custom-object layout name from the object label.
    # SOQL against Layout is not supported in this org/API surface.
    layout_label = f"{describe.get('label', OBJECT_API_NAME)} Layout"
    logger.info("Targeting layout: %s", layout_label)

    # 4. Build and deploy
    logger.info("Building metadata ZIP...")
    zip_bytes = build_zip(OBJECT_API_NAME, custom_fields, profile_names, layout_label)
    logger.info("ZIP size: %d bytes", len(zip_bytes))

    success = deploy_zip(sf, zip_bytes)
    if success:
        logger.info(
            "All %d fields on %s are now readable/editable for all profiles "
            "and visible on the layout.",
            len(custom_fields), OBJECT_API_NAME,
        )
    else:
        logger.error("Deployment failed — check errors above.")


if __name__ == "__main__":
    main()
