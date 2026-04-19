"""
Instrument scanner — test every available pair with the hybrid strategy.
730 days, wick SL, spread, commission. Rank by PF.
"""

from __future__ import annotations

from __future__ import annotations

import logging
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yfinance as yf

import config
from strategy.hybrid import generate_signal
from core.structure import detect_structure, Bias

logging.basicConfig(level=logging.WARNING)

ALL_PAIRS = {
    "SP500":  {"ticker": "ES=F",     "spread": 0.5,     "lot_mult": 50,     "min_sl": 10.0,   "comm": 3.0, "session": [(13, 20)]},
    "US30":   {"ticker": "YM=F",     "spread": 2.0,     "lot_mult": 5,      "min_sl": 30.0,   "comm": 3.0, "session": [(13, 20)]},
    "NAS100": {"ticker": "NQ=F",     "spread": 1.5,     "lot_mult": 20,     "min_sl": 20.0,   "comm": 3.0, "session": [(13, 20)]},
    "RUSSEL": {"ticker": "RTY=F",    "spread": 0.3,     "lot_mult": 50,     "min_sl": 3.0,    "comm": 3.0, "session": [(13, 20)]},
    "GBPUSD": {"ticker": "GBPUSD=X", "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.001,  "comm": 5.0, "session": [(7, 11), (13, 17)]},
    "EURUSD": {"ticker": "EURUSD=X", "spread": 0.00012, "lot_mult": 100000, "min_sl": 0.0008, "comm": 5.0, "session": [(7, 11), (13, 17)]},
    "USDJPY": {"ticker": "USDJPY=X", "spread": 0.015,   "lot_mult": 1000,   "min_sl": 0.10,   "comm": 5.0, "session": [(7, 11), (13, 17)]},
    "GBPJPY": {"ticker": "GBPJPY=X", "spread": 0.02,    "lot_mult": 1000,   "min_sl": 0.15,   "comm": 5.0, "session": [(7, 11), (13, 17)]},
    "AUDUSD": {"ticker": "AUDUSD=X", "spread": 0.00015, "lot_mult": 100000, "min_sl": 0.0008, "comm": 5.0, "session": [(0, 5), (13, 17)]},
    "GOLD":   {"ticker": "GC=F",     "spread": 2.5,     "lot_mult": 100,    "min_sl": 5.0,    "comm": 7.0, "session": [(7, 11), (13, 17)]},
    "OIL":    {"ticker": "CL=F",     "spread": 0.05,    "lot_mult": 1000,   "min_sl": 0.20,   "comm": 5.0, "session": [(13, 20)]},
    "SILVER": {"ticker": "SI=F",     "spread": 0.03,    "lot_mult": 5000,   "min_sl": 0.05,   "comm": 5.0, "session": [(7, 11), (13, 17)]},
}


def fetch(ticker):
    r = {}
    for tf, interval, period in [("1h", "1h", "730d"), ("daily", "1d", "5y")]:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None: d.index = d.index.tz_localize("UTC")
        r[tf] = d
    r["4h"] = r["1h"].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return r


def in_session(sessions, hour):
    return any(s <= hour < e for s, e in sessions) if sessions else True


def test_pair(sym, cfg):
    # Temporarily set config to this instrument
    original = config.INSTRUMENTS.copy()
    config.INSTRUMENTS = {sym: cfg}

    tf = fetch(cfg["ticker"])
    d1h = tf["1h"]

    warmup = 200
    if len(d1h) <= warmup:
        config.INSTRUMENTS = original
        return None

    tradeable = d1h.index[warmup:]
    equity = 5000.0
    pos = None
    trades = []

    for i, ts in enumerate(tradeable):
        loc = d1h.index.get_loc(ts)
        w1h = d1h.iloc[max(0, loc - 199):loc + 1]
        w4h = tf["4h"].loc[:ts].iloc[-100:]
        wd = tf["daily"].loc[:ts].iloc[-60:]

        if len(w1h) < 50 or len(w4h) < 10 or len(wd) < 15:
            continue

        price = float(w1h["close"].iloc[-1])
        bh = float(w1h["high"].iloc[-1])
        bl = float(w1h["low"].iloc[-1])

        if pos:
            bars = i - pos["bar"]
            close_it, reason, ep = False, "", price

            if pos["side"] == "long" and bl <= pos["sl"]: close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bh >= pos["sl"]: close_it, reason, ep = True, "SL", pos["sl"]

            if not close_it:
                if pos["side"] == "long" and bh >= pos["tp"]: close_it, reason, ep = True, "TP", pos["tp"]
                elif pos["side"] == "short" and bl <= pos["tp"]: close_it, reason, ep = True, "TP", pos["tp"]

            if pos["side"] == "long" and bl <= pos["sl"] and bh >= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            elif pos["side"] == "short" and bh >= pos["sl"] and bl <= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]

            if not close_it and bars >= 20: close_it, reason, ep = True, "Time", price

            if close_it:
                raw = ((ep - pos["entry"]) if pos["side"] == "long" else (pos["entry"] - ep)) * pos["lots"] * cfg["lot_mult"]
                net = raw - cfg["comm"] * pos["lots"]
                equity += net
                trades.append(net)
                pos = None

        if pos is None and in_session(cfg["session"], ts.hour):
            sig = generate_signal(w1h, w4h, wd, price, sym)
            if sig:
                risk = abs(sig.entry - sig.sl)
                if risk >= cfg["min_sl"] * 0.3:
                    risk_d = min(equity * 0.02 * sig.confidence, equity * 0.03)
                    lots = max(0.01, min(0.10, round(risk_d / (risk * cfg["lot_mult"]), 2)))
                    pos = {"side": sig.direction, "entry": sig.entry, "sl": sig.sl,
                           "tp": sig.tp, "lots": lots, "bar": i}

    config.INSTRUMENTS = original

    if not trades:
        return None

    wins = sum(1 for t in trades if t > 0)
    gw = sum(t for t in trades if t > 0)
    gl = abs(sum(t for t in trades if t < 0))
    pf = gw / gl if gl > 0 else float("inf")
    pnl = sum(trades)
    days = len(tradeable) / 24

    return {
        "sym": sym, "trades": len(trades), "wins": wins,
        "wr": wins / len(trades) * 100, "pf": pf,
        "pnl": pnl, "days": days,
        "monthly": pnl / (days / 30),
    }


def main():
    print(f"\n{'='*75}")
    print(f"  VIPER v3 — Full Instrument Scan (730 days)")
    print(f"{'='*75}\n")

    results = []
    for sym, cfg in ALL_PAIRS.items():
        print(f"  {sym:8s}...", end=" ", flush=True)
        try:
            r = test_pair(sym, cfg)
            if r and r["trades"] >= 3:
                pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
                print(f"{r['trades']:3d}T  {r['wr']:.0f}%WR  PF={pf_s:>5}  "
                      f"${r['pnl']:>8,.2f}  ${r['monthly']:>6,.2f}/mo")
                results.append(r)
            elif r:
                print(f"too few trades ({r['trades']})")
            else:
                print("no trades")
        except Exception as e:
            print(f"ERROR: {e}")
        time.sleep(1)

    results.sort(key=lambda r: r["pf"], reverse=True)

    print(f"\n{'='*75}")
    print(f"  RANKINGS — by Profit Factor (min 3 trades)")
    print(f"{'='*75}")
    print(f"  {'#':>2} {'Pair':8s} {'T':>4} {'WR%':>5} {'PF':>6} {'PnL':>10} {'$/mo':>8} {'Pass?':>6}")
    print(f"  {'-'*55}")

    for i, r in enumerate(results):
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        passes = "YES" if r["pnl"] > 400 else "maybe" if r["pnl"] > 200 else "no"
        print(f"  {i+1:>2} {r['sym']:8s} {r['trades']:>4} {r['wr']:>5.1f} {pf_s:>6} "
              f"${r['pnl']:>9,.2f} ${r['monthly']:>7,.2f} {passes:>6}")

    profitable = [r for r in results if r["pf"] > 1.0]
    if profitable:
        combined = sum(r["pnl"] for r in profitable)
        combined_monthly = sum(r["monthly"] for r in profitable)
        print(f"\n  Profitable instruments: {len(profitable)}")
        print(f"  Combined PnL: ${combined:,.2f}")
        print(f"  Combined monthly: ${combined_monthly:,.2f}")
        est = 400 / combined_monthly * 30 if combined_monthly > 0 else float("inf")
        print(f"  Phase 1 estimate (combined): {est:.0f} days")

    print(f"\n{'='*75}")


if __name__ == "__main__":
    main()
