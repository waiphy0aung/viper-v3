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
# Seasonal rotation: SP500 Oct-Mar, US30 Apr-Sep
INSTRUMENTS = {
    "SP500": {
        "ticker": "ES=F", "spread": 0.5, "lot_mult": 50,
        "min_sl": 10.0, "comm": 3.0,
        "session": [(13, 20)],
        "months": [10, 11, 12, 1, 2, 3],  # Oct-Mar
    },
    "US30": {
        "ticker": "YM=F", "spread": 2.0, "lot_mult": 5,
        "min_sl": 30.0, "comm": 3.0,
        "session": [(13, 20)],
        "months": [4, 5, 6, 7, 8, 9],     # Apr-Sep
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

# =============================================================================
# TRAILING SL
# =============================================================================
TRAILING_SL_ENABLED = True
TRAILING_SL_TRIGGER_RR = 1.0  # move SL to BE when price moves 1:1 in favor

# =============================================================================
# FILTERS
# =============================================================================
# Seasonal rotation: each instrument has its own trading months
SEASONAL_FILTER = True

# Weekend: no signals Fri 20:00 UTC → Sun 22:00 UTC
WEEKEND_FILTER = True

# News: block NFP (1st Friday 12-14 UTC) + FOMC (3rd Wednesday 18-20 UTC)
NEWS_FILTER = True

# Cooldown: pause 4h after 2 consecutive losses
COOLDOWN_AFTER_LOSSES = 2
COOLDOWN_HOURS = 4

# =============================================================================
# DD ALERTS — Telegram alerts at these thresholds (one-time per level)
# =============================================================================
DD_ALERT_LEVELS = [0.03, 0.05, 0.07]  # 3%, 5%, 7%

# =============================================================================
# EXPECTED PERFORMANCE (for weekly report comparison)
# =============================================================================
EXPECTED_WR = 51.4    # monster mode funded phase
EXPECTED_PF = 2.75

# =============================================================================
# MONSTER MODE — sniper trades with 1:5 to 1:10+ R:R
# =============================================================================
# When enabled: A/A+ only, weekly TP targets, hold for days, partial at milestones
# Both modes active: normal for flow, monster for snipers
# Every signal is evaluated — A-grade with 1:4+ R:R → monster treatment
# B-grade or low R:R → normal quick trade
MONSTER_MODE = True        # enables monster TP/trailing for qualifying trades
NORMAL_MODE = False        # B-grade hurts funded PnL — keep off

# Quality gate — only these grades allowed in monster mode
MONSTER_GRADES = ["A"]  # A+ rejection requirement hurts more than helps

# TP targeting: weekly draw on liquidity (PWH/PWL), not daily
# Minimum R:R to take a trade — if weekly target doesn't give this, skip
MONSTER_MIN_RR = 4.0

# Hold time: up to 20 days (480 bars on 1H) — let the trade breathe
MONSTER_TIME_STOP = 480

# Partial TP milestones: close portions as price moves in favor
# [R:R trigger, % of position to close]
MONSTER_PARTIALS = [
    (3.0, 0.30),   # at 1:3 → close 30%, move SL to 1:1
    (5.0, 0.30),   # at 1:5 → close 30%, move SL to 1:3
    # remaining 40% runs to full TP or trailing SL
]

# Risk: bigger position on A+ (sniper precision)
MONSTER_RISK = {
    "A+": 0.03,    # 3% — full conviction
    "A":  0.02,    # 2% — high conviction
}

# Trailing after partials: move SL to last swing low/high every 24 bars
MONSTER_TRAIL_INTERVAL = 24

# Cooldown between monster trades: 3 days minimum
MONSTER_COOLDOWN_HOURS = 72

# =============================================================================
# SCALPER MODE — fast indicator signals, ATR SL, fixed 1:2 TP
# =============================================================================
SCALPER_SL_ATR_MULT = 1.5       # SL = 1.5 × 14-bar ATR
SCALPER_RR = 2.0                # TP = RR × SL distance (1:2)
SCALPER_TIME_STOP = 8           # close after 8 bars if unresolved
SCALPER_RISK_PCT = 0.01         # 1% risk per trade

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = "viper.log"
