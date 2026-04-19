"""
VIPER v3 — Hybrid SMC + Indicator Engine
Configuration
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
load_dotenv()

# =============================================================================
# PROP FIRM — Funding Pips $5k
# =============================================================================
ACCOUNT_SIZE = 5000
DAILY_DD_LIMIT = 0.05
MAX_DD_LIMIT = 0.10
EQUITY_FLOOR = ACCOUNT_SIZE * (1 - MAX_DD_LIMIT)
PROFIT_TARGET_PHASE1 = 0.08
PROFIT_TARGET_PHASE2 = 0.05

# =============================================================================
# INSTRUMENTS — validated profitable over 730 days
# =============================================================================
# SP500 only — 75% WR, +$390 over 560 days. Only proven winner.
INSTRUMENTS = {
    "SP500": {
        "ticker": "ES=F", "spread": 0.5, "lot_mult": 50,
        "min_sl": 10.0, "comm": 3.0,
        "session": [(13, 20)],
    },
}

# =============================================================================
# SMC PARAMETERS
# =============================================================================
SWING_STRENGTH = 3
STRUCTURE_LOOKBACK = 50
ZONE_MIN_HEIGHT_MULT = 0.3     # zone must be > min_sl * this
ZONE_MAX_HEIGHT_MULT = 20      # cap wide zones
REJECTION_WICK_RATIO = 0.35
REJECTION_BODY_RATIO = 0.15
SL_BUFFER_MULT = 0.3           # SL behind wick + this * zone_height

# =============================================================================
# INDICATOR PARAMETERS (from PHANTOM — proven on crypto, adapted for forex/indices)
# =============================================================================
# Keltner squeeze
KC_PERIOD = 20
KC_ATR_PERIOD = 10
KC_MULTIPLIER = 1.5
BB_PERIOD = 20
BB_STD = 2.0

# HMA crossover
HMA_FAST = 9
HMA_SLOW = 21

# Volume
VOLUME_MULTIPLIER = 1.3

# =============================================================================
# HYBRID STRATEGY
# =============================================================================
# SMC zones boost confidence but don't block
REQUIRE_SMC_ZONE = False

# Indicator-only signals are the primary method
ALLOW_INDICATOR_ONLY = True
INDICATOR_ONLY_CONFIDENCE = 0.3     # 30% confidence — small exploratory position
SMC_ZONE_CONFIDENCE = 0.8           # 80% confidence (bigger lots)
SMC_ZONE_REJECTION_CONFIDENCE = 1.0 # 100% confidence (full lots)

# =============================================================================
# RISK
# =============================================================================
BASE_RISK_PCT = 0.02           # 2% base
MAX_RISK_CAP = 0.03            # 3% absolute max
MIN_RR = 1.5

# =============================================================================
# TELEGRAM
# =============================================================================
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "viper.log"
