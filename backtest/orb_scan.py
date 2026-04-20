"""
Scan ORB across many instruments to find which have real edge.
Tests: US indices, European indices, commodities, crypto.
Each with appropriate ORB session timing.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from statistics import mean

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategy.hybrid import Signal, Quality

logging.basicConfig(level=logging.WARNING)

# Instruments to scan, with ORB session hour (UTC)
SCAN_UNIVERSE = {
    # US session (13:00 UTC NY open)
    "US30":    {"ticker": "YM=F",   "spread": 2.0, "lot_mult": 5,   "min_sl": 30.0, "comm": 3.0, "orb_hour": 13},
    "RTY":     {"ticker": "RTY=F",  "spread": 0.5, "lot_mult": 50,  "min_sl": 5.0,  "comm": 3.0, "orb_hour": 13},
    "NAS100":  {"ticker": "NQ=F",   "spread": 1.0, "lot_mult": 20,  "min_sl": 20.0, "comm": 3.0, "orb_hour": 13},
    "SP500":   {"ticker": "ES=F",   "spread": 0.5, "lot_mult": 50,  "min_sl": 10.0, "comm": 3.0, "orb_hour": 13},
    # US session commodities
    "CRUDE":   {"ticker": "CL=F",   "spread": 0.03, "lot_mult": 1000, "min_sl": 0.20, "comm": 3.0, "orb_hour": 13},
    "SILVER":  {"ticker": "SI=F",   "spread": 0.03, "lot_mult": 5000, "min_sl": 0.10, "comm": 3.0, "orb_hour": 13},
    # European session (07:00 UTC London open)
    "DAX":     {"ticker": "^GDAXI", "spread": 1.0, "lot_mult": 25,  "min_sl": 20.0, "comm": 3.0, "orb_hour": 7},
    "FTSE":    {"ticker": "^FTSE",  "spread": 0.5, "lot_mult": 10,  "min_sl": 5.0,  "comm": 3.0, "orb_hour": 7},
    "CAC":     {"ticker": "^FCHI",  "spread": 0.5, "lot_mult": 10,  "min_sl": 5.0,  "comm": 3.0, "orb_hour": 7},
}

ORB_TP_MULT = 2.0
ORB_TIME_STOP = 5
RISK_PCT = 0.015


def fetch(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="730d", interval="1h", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def get_orb_range(df: pd.DataFrame, now_ts, orb_hour: int) -> tuple[float, float, float, bool]:
    start = now_ts.replace(hour=orb_hour, minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=1)
    session = df.loc[start:end]
    if len(session) < 1:
        return 0, 0, 0, False
    h = float(session["high"].max())
    l = float(session["low"].min())
    size = h - l
    if size <= 0:
        return h, l, size, False
    return h, l, size, True


def gen_orb_signal(df: pd.DataFrame, price: float, bh: float, bl: float,
                   now_ts, cfg: dict) -> dict | None:
    hour = now_ts.hour
    orb_hour = cfg["orb_hour"]
    # Entry window: orb_hour+1 to orb_hour+5
    if hour < orb_hour + 1 or hour >= orb_hour + 5:
        return None
    h, l, size, valid = get_orb_range(df, now_ts, orb_hour)
    if not valid or size < cfg["min_sl"]:
        return None

    direction = None
    if bh > h and price > h:
        direction = "long"
    elif bl < l and price < l:
        direction = "short"
    else:
        return None

    # Trend filter: daily close > 20-day SMA for longs
    daily = df["close"].resample("1D").last().dropna()
    if len(daily) >= 20:
        sma20 = float(daily.tail(20).mean())
        last = float(daily.iloc[-1])
        if direction == "long" and last < sma20: return None
        if direction == "short" and last > sma20: return None

    buffer = size * 0.1
    if direction == "long":
        sl = l - buffer
        tp = price + size * ORB_TP_MULT
        fill = price + cfg["spread"]
    else:
        sl = h + buffer
        tp = price - size * ORB_TP_MULT
        fill = price - cfg["spread"]
    risk = abs(fill - sl)
    if risk < cfg["min_sl"] * 0.5:
        return None
    return {"side": direction, "entry": fill, "sl": sl, "tp": tp}


def run_window(data: pd.DataFrame, cfg: dict, tradeable) -> dict:
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_curve = [equity]
    daily_eq = {}
    daily_trade_taken = set()
    orb_hour = cfg["orb_hour"]

    for bar, ts in enumerate(tradeable):
        day = ts.date()
        if day not in daily_eq:
            daily_eq[day] = equity

        if pos is not None:
            if ts not in data.index:
                eq_curve.append(equity); continue
            price = float(data.loc[ts, "close"])
            bh = float(data.loc[ts, "high"])
            bl = float(data.loc[ts, "low"])
            bars_held = bar - pos["bar"]
            close_it, reason, ep = False, "", price

            if pos["side"] == "long" and bl <= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bh >= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            if not close_it:
                if pos["side"] == "long" and bh >= pos["tp"]:
                    close_it, reason, ep = True, "TP", pos["tp"]
                elif pos["side"] == "short" and bl <= pos["tp"]:
                    close_it, reason, ep = True, "TP", pos["tp"]
            if pos["side"] == "long" and bl <= pos["sl"] and bh >= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bh >= pos["sl"] and bl <= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            if not close_it and bars_held >= ORB_TIME_STOP:
                close_it, reason, ep = True, "Time", price
            # Session end — 6 hours after ORB
            if not close_it and ts.hour >= orb_hour + 7:
                close_it, reason, ep = True, "SessionEnd", price

            if close_it:
                raw = ((ep - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - ep)) * pos["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * pos["lots"]
                equity += raw - c
                trades.append({"pnl": raw - c, "reason": reason, "ts": ts})
                pos = None

        if pos is None and day not in daily_trade_taken:
            dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
            if dd < 0.04 and ts in data.index:
                loc = data.index.get_loc(ts)
                if loc >= 500:
                    w = data.iloc[max(0, loc - 500):loc + 1]
                    price = float(data.loc[ts, "close"])
                    bh = float(data.loc[ts, "high"])
                    bl = float(data.loc[ts, "low"])
                    sig = gen_orb_signal(w, price, bh, bl, ts, cfg)
                    if sig is not None:
                        risk = abs(sig["entry"] - sig["sl"])
                        if risk >= cfg["min_sl"] * 0.5:
                            rd = equity * RISK_PCT
                            lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                            pos = {"side": sig["side"], "entry": sig["entry"], "sl": sig["sl"],
                                   "tp": sig["tp"], "lots": lots, "bar": bar}
                            daily_trade_taken.add(day)

        eq = equity
        if pos and ts in data.index:
            px = float(data.loc[ts, "close"])
            eq += ((px - pos["entry"]) if pos["side"] == "long" else
                   (pos["entry"] - px)) * pos["lots"] * cfg["lot_mult"]
        eq_curve.append(eq)

    # Summarize
    pnl = equity - config.ACCOUNT_SIZE
    wins = sum(1 for t in trades if t["pnl"] > 0)
    wr = wins / len(trades) * 100 if trades else 0
    peak = eq_curve[0]; mdd = 0
    for eq in eq_curve:
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        mdd = max(mdd, dd)
    wp = [t["pnl"] for t in trades if t["pnl"] > 0]
    lp = [t["pnl"] for t in trades if t["pnl"] < 0]
    gw, gl = sum(wp), abs(sum(lp))
    pf = gw / gl if gl > 0 else float("inf")
    return {"trades": len(trades), "wr": wr, "pnl": pnl, "pf": pf, "mdd": mdd}


def main():
    print(f"\n{'=' * 84}")
    print(f"  ORB SCAN — trend filter ON, {RISK_PCT*100:.1f}% risk, TP={ORB_TP_MULT}×range")
    print(f"{'=' * 84}")
    print(f"  {'Symbol':8s} {'TRAIN':>28s}  {'TEST':>28s}  {'VALIDATE':>28s}")
    print(f"  {'':8s} {'T  WR    PnL    PF   DD':>28s}  {'T  WR    PnL    PF   DD':>28s}  {'T  WR    PnL    PF   DD':>28s}")
    print(f"  {'-' * 88}")

    survivors = []
    for sym, cfg in SCAN_UNIVERSE.items():
        try:
            data = fetch(cfg["ticker"])
            idx = data.index[500:]
            n = len(idx)
            if n < 500:
                print(f"  {sym:8s} insufficient data"); continue
            splits = [
                idx[:int(n * 0.40)],
                idx[int(n * 0.40):int(n * 0.70)],
                idx[int(n * 0.70):],
            ]
            rs = [run_window(data, cfg, s) for s in splits]

            def fmt(r):
                pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
                return f"{r['trades']:3d} {r['wr']:4.0f}% ${r['pnl']:>+5,.0f} {pf_s:>5s} {r['mdd']:>4.1f}%"

            print(f"  {sym:8s} {fmt(rs[0]):>28s}  {fmt(rs[1]):>28s}  {fmt(rs[2]):>28s}")
            # Survivor check: all windows profitable or VALIDATE > 0
            val = rs[2]
            if val["pnl"] > 0 and val["mdd"] < 10 and val["pf"] > 1.0:
                survivors.append((sym, val))
            time.sleep(1)
        except Exception as e:
            print(f"  {sym:8s} ERROR: {e}")

    print(f"\n{'=' * 84}")
    print(f"  SURVIVORS (VALIDATE positive, DD < 10%, PF > 1.0):")
    if not survivors:
        print(f"    (none)")
    else:
        for sym, v in survivors:
            print(f"    {sym:8s} VALIDATE: ${v['pnl']:+,.0f} | {v['wr']:.0f}% WR | PF {v['pf']:.2f} | DD {v['mdd']:.1f}%")
    print(f"{'=' * 84}\n")


if __name__ == "__main__":
    main()
