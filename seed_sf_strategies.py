#!/usr/bin/env python3
"""
seed_sf_strategies.py

Upserts all 3 trading strategies as records in the Salesforce Strategy__c object.

Usage:
    python seed_sf_strategies.py

Required env vars: SF_USERNAME, SF_PASSWORD, SF_SECURITY_TOKEN, SF_DOMAIN
"""

import logging
import os

from dotenv import load_dotenv
from simple_salesforce import Salesforce

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

STRATEGIES = [
    {
        "Strategy_Name__c": "v4_gamma",
        "Description__c": (
            "V4 Gamma — Nifty options straddle with trailing stop-loss. "
            "Enters at 13:30 IST on expiry day with 35% initial SL, locks in at 20% profit, "
            "trails at 15%. BUY straddle + SL-M order; modify trail via Dhan API."
        ),
    },
    {
        "Strategy_Name__c": "gamma_blast",
        "Description__c": (
            "Gamma Blast — Aggressive NIFTY options short-sell strategy. "
            "SELL straddle on 5-minute momentum bursts with SL-M BUY protection. "
            "48-week historical sweep capturing expansion straddles and directional moves."
        ),
    },
    {
        "Strategy_Name__c": "zscore_nifty",
        "Description__c": (
            "Z-Score Nifty — Mean-reversion on NIFTY spot 5-min candles (13:30–15:00 IST). "
            "Entry on z-score ±2.0 (lookback 20), exit at ±0.5, stop at ±2.3. "
            "ADX filter (max 25), rupee stop ₹5000/trade, daily loss cap ₹7500, max 3 trades/day. "
            "1-year performance: 27 trades, 70% win rate, ₹53,210 profit, 10.64% ROI, PF 12.78."
        ),
    },
]


def connect_sf():
    sf = Salesforce(
        username=os.getenv("SF_USERNAME"),
        password=os.getenv("SF_PASSWORD"),
        security_token=os.getenv("SF_SECURITY_TOKEN", ""),
        domain=os.getenv("SF_DOMAIN", "login"),
    )
    logger.info("Connected to Salesforce as %s", os.getenv("SF_USERNAME"))
    return sf


def upsert_strategies(sf: Salesforce):
    # Fetch existing strategy names
    existing_result = sf.query("SELECT Id, Strategy_Name__c FROM Strategy__c")
    existing_map = {r["Strategy_Name__c"]: r["Id"] for r in existing_result.get("records", [])}
    logger.info("Found %d existing Strategy__c records: %s", len(existing_map), list(existing_map.keys()))

    for strat in STRATEGIES:
        name = strat["Strategy_Name__c"]
        if name in existing_map:
            # Update existing record
            sf.Strategy__c.update(existing_map[name], strat)
            logger.info("Updated Strategy__c: %s (Id=%s)", name, existing_map[name])
        else:
            # Insert new record
            result = sf.Strategy__c.create(strat)
            if result.get("success"):
                logger.info("Inserted Strategy__c: %s (Id=%s)", name, result.get("id"))
            else:
                logger.error("Failed to insert %s: %s", name, result.get("errors"))


if __name__ == "__main__":
    sf = connect_sf()
    upsert_strategies(sf)
    logger.info("Done — all 3 strategies are now in Salesforce Strategy__c.")
