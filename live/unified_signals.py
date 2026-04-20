"""
Unified signal generator — ORB + MR alerts to Telegram.

Read-only: no account, no auto-execution. Just pings when a signal fires.
User manually executes on Funding Pips MT5.

Loop:
- Every 5 min during market hours
- Fetch latest 1H + daily data
- Evaluate ORB (intraday) and MR (end-of-day) per instrument
- Dedupe so each signal fires max once per day
- Telegram alert with entry/SL/TP/lots/risk
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategy.mean_reversion import ibs as mr_ibs
from live.notifier import _send

# ============================================================
# CONFIG — mirrors wf_unified.py
# ============================================================
INSTRUMENTS = {
    "US30":  {"ticker": "YM=F",   "spread": 2.0, "lot_mult": 5,  "min_sl": 30.0, "comm": 3.0,
              "orb_hour": 13, "mr_check_hour": 20, "use_orb": True,  "use_mr": True},
    "RTY":   {"ticker": "RTY=F",  "spread": 0.5, "lot_mult": 50, "min_sl": 5.0,  "comm": 3.0,
              "orb_hour": 13, "mr_check_hour": 20, "use_orb": True,  "use_mr": True},
    "DAX":   {"ticker": "^GDAXI", "spread": 1.0, "lot_mult": 25, "min_sl": 20.0, "comm": 3.0,
              "orb_hour": 7,  "mr_check_hour": 15, "use_orb": True,  "use_mr": False},
    "SP500": {"ticker": "ES=F",   "spread": 0.5, "lot_mult": 50, "min_sl": 10.0, "comm": 3.0,
              "orb_hour": 13, "mr_check_hour": 20, "use_orb": False, "use_mr": True},
}

ORB_TP_MULT = 2.0
ORB_RISK_PCT = 0.01
MR_SL_ATR_MULT = 1.5
MR_RISK_PCT = 0.01
ACCOUNT_SIZE = 5000   # for lot size calc in alerts

LOOP_INTERVAL_SEC = 300   # 5 minutes
STATE_FILE = Path("live_state.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
# STATE — track daily dedup across restarts
# ============================================================
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            pass
    return {"date": "", "sent": []}


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2))


def reset_state_if_new_day(state: dict) -> dict:
    today = datetime.now(timezone.utc).date().isoformat()
    if state.get("date") != today:
        state = {"date": today, "sent": []}
        save_state(state)
    return state


def already_sent(state: dict, key: str) -> bool:
    return key in state["sent"]


def mark_sent(state: dict, key: str):
    state["sent"].append(key)
    save_state(state)


# ============================================================
# DATA FETCH — with retry
# ============================================================
def fetch_1h(ticker: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            d = yf.download(ticker, period="60d", interval="1h", progress=False)
            if d is None or d.empty:
                continue
            d.columns = [c[0].lower() for c in d.columns]
            if d.index.tz is None:
                d.index = d.index.tz_localize("UTC")
            return d
        except Exception as e:
            logger.warning(f"Fetch {ticker} attempt {attempt+1} failed: {e}")
            time.sleep(5)
    return None


def fetch_daily(ticker: str) -> pd.DataFrame | None:
    try:
        d = yf.download(ticker, period="60d", interval="1d", progress=False)
        if d is None or d.empty:
            return None
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        return d
    except Exception as e:
        logger.warning(f"Fetch daily {ticker} failed: {e}")
        return None


# ============================================================
# SIGNAL GENERATORS
# ============================================================
def orb_range(df: pd.DataFrame, now_ts: pd.Timestamp, orb_hour: int):
    start = now_ts.replace(hour=orb_hour, minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=1)
    s = df.loc[start:end]
    if len(s) < 1:
        return 0, 0, 0, False
    h, l = float(s["high"].max()), float(s["low"].min())
    size = h - l
    return h, l, size, size > 0


def check_orb(df_1h: pd.DataFrame, cfg: dict, now_ts: pd.Timestamp) -> dict | None:
    hour = now_ts.hour
    orb_h = cfg["orb_hour"]
    if hour < orb_h + 1 or hour >= orb_h + 5:
        return None
    h, l, size, valid = orb_range(df_1h, now_ts, orb_h)
    if not valid or size < cfg["min_sl"]:
        return None

    last_bar = df_1h.iloc[-1]
    price = float(last_bar["close"])
    bh = float(last_bar["high"])
    bl = float(last_bar["low"])

    direction = None
    if bh > h and price > h:
        direction = "long"
    elif bl < l and price < l:
        direction = "short"
    else:
        return None

    # Trend filter
    daily = df_1h["close"].resample("1D").last().dropna()
    if len(daily) >= 20:
        sma20 = float(daily.tail(20).mean())
        last = float(daily.iloc[-1])
        if direction == "long" and last < sma20:
            return None
        if direction == "short" and last > sma20:
            return None

    buf = size * 0.1
    if direction == "long":
        sl = l - buf
        tp = price + size * ORB_TP_MULT
        fill = price + cfg["spread"]
    else:
        sl = h + buf
        tp = price - size * ORB_TP_MULT
        fill = price - cfg["spread"]

    risk = abs(fill - sl)
    if risk < cfg["min_sl"] * 0.5:
        return None

    rr = abs(tp - fill) / risk
    return {"strategy": "ORB", "side": direction, "entry": fill, "sl": sl, "tp": tp,
            "rr": rr, "orb_high": h, "orb_low": l, "orb_size": size}


def check_mr(df_daily: pd.DataFrame, cfg: dict, today_high: float, today_low: float,
             today_close: float) -> dict | None:
    if len(df_daily) < 10:
        return None
    prior5_low = float(df_daily["low"].iloc[-6:-1].min())
    if today_close >= prior5_low:
        return None
    if mr_ibs(today_high, today_low, today_close) >= 0.25:
        return None
    daily_range = float((df_daily["high"] - df_daily["low"]).tail(14).mean())
    sl_dist = max(daily_range * MR_SL_ATR_MULT, cfg["min_sl"])
    sma5 = float(df_daily["close"].tail(5).mean())
    fill = today_close + cfg["spread"]
    sl = today_close - sl_dist
    tp = sma5
    if tp <= fill:
        return None
    risk = fill - sl
    if risk <= 0 or (tp - fill) / risk < 0.8:
        return None
    return {"strategy": "MR", "side": "long", "entry": fill, "sl": sl, "tp": tp,
            "rr": (tp - fill) / risk, "sma5": sma5, "prior5_low": prior5_low}


# ============================================================
# TELEGRAM ALERT
# ============================================================
def format_signal(sym: str, sig: dict, equity: float) -> str:
    strat = sig["strategy"]
    side = sig["side"]
    emoji = "\U0001f7e2" if side == "long" else "\U0001f534"
    direction = "BUY" if side == "long" else "SELL"

    risk_per_unit = abs(sig["entry"] - sig["sl"])
    risk_pct = ORB_RISK_PCT if strat == "ORB" else MR_RISK_PCT
    risk_dollars = equity * risk_pct
    cfg = INSTRUMENTS[sym]
    lots = max(0.01, min(0.50, round(risk_dollars / (risk_per_unit * cfg["lot_mult"]), 2)))
    est_risk = lots * risk_per_unit * cfg["lot_mult"]
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    lines = [
        f"{emoji} <b>{direction} {sym}</b> [{strat}]",
        "",
        f"\U0001f4cd Entry:  <code>{sig['entry']:.2f}</code>",
        f"\U0001f6d1 SL:     <code>{sig['sl']:.2f}</code>",
        f"\U0001f3af TP:     <code>{sig['tp']:.2f}</code>",
        "",
        f"\U0001f4e6 Lot:    <code>{lots:.2f}</code>",
        f"\U0001f4b0 Risk:   <code>${est_risk:.0f}</code> ({risk_pct*100:.1f}%)",
        f"\U0001f4ca R:R:    <code>1:{sig['rr']:.1f}</code>",
        "",
    ]

    if strat == "ORB":
        lines.append(f"\U0001f4ca ORB range: <code>{sig['orb_low']:.2f}-{sig['orb_high']:.2f}</code>")
    else:
        lines.append(f"\U0001f4ca 5d-low: <code>{sig['prior5_low']:.2f}</code> | SMA5: <code>{sig['sma5']:.2f}</code>")

    lines.append(f"\u23f0 {now}")
    return "\n".join(lines)


# ============================================================
# MAIN LOOP
# ============================================================
def scan_once(state: dict):
    now = datetime.now(timezone.utc)
    state = reset_state_if_new_day(state)

    for sym, cfg in INSTRUMENTS.items():
        try:
            # ORB check — only during NY/EU open window
            if cfg["use_orb"]:
                key = f"{sym}_ORB_{state['date']}"
                if not already_sent(state, key):
                    orb_h = cfg["orb_hour"]
                    if orb_h + 1 <= now.hour < orb_h + 5:
                        df = fetch_1h(cfg["ticker"])
                        if df is not None and len(df) >= 30:
                            sig = check_orb(df, cfg, now)
                            if sig:
                                logger.info(f"ORB signal fired for {sym}: {sig['side']}")
                                _send(format_signal(sym, sig, ACCOUNT_SIZE))
                                mark_sent(state, key)

            # MR check — at mr_check_hour (end of day)
            if cfg["use_mr"]:
                key = f"{sym}_MR_{state['date']}"
                if not already_sent(state, key):
                    if now.hour == cfg["mr_check_hour"]:
                        df_daily = fetch_daily(cfg["ticker"])
                        df_1h = fetch_1h(cfg["ticker"])
                        if df_daily is not None and df_1h is not None and len(df_daily) >= 20:
                            # Today's bar from 1H
                            today_start = pd.Timestamp(now.date()).tz_localize("UTC")
                            today_data = df_1h.loc[today_start:]
                            if len(today_data) >= 3:
                                today_close = float(today_data["close"].iloc[-1])
                                today_high = float(today_data["high"].max())
                                today_low = float(today_data["low"].min())
                                # Exclude today from daily history
                                hist = df_daily.loc[:today_start - pd.Timedelta(hours=1)]
                                sig = check_mr(hist, cfg, today_high, today_low, today_close)
                                if sig:
                                    logger.info(f"MR signal fired for {sym}")
                                    _send(format_signal(sym, sig, ACCOUNT_SIZE))
                                    mark_sent(state, key)

        except Exception:
            logger.error(f"Scan error for {sym}:\n{traceback.format_exc()}")


def main():
    logger.info("VIPER v3 Unified Signals — starting")
    _send("\U0001f40d <b>VIPER v3 Unified Signals ONLINE</b>\n\n"
          "Strategies: ORB (intraday) + MR (daily)\n"
          f"Instruments: {', '.join(INSTRUMENTS.keys())}\n"
          "Mode: alert-only (manual execution)")

    state = load_state()

    while True:
        try:
            scan_once(state)
        except Exception:
            logger.error(f"Loop error:\n{traceback.format_exc()}")
            try:
                _send(f"\u26a0\ufe0f Signal loop error — retrying in {LOOP_INTERVAL_SEC}s")
            except Exception:
                pass
        time.sleep(LOOP_INTERVAL_SEC)


if __name__ == "__main__":
    main()
