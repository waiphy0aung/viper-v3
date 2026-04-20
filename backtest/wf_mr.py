"""
Walk-forward for mean reversion on daily timeframe across indices.
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
from strategy.mean_reversion import generate_mr_signal

logging.basicConfig(level=logging.WARNING)

INSTRUMENTS = {
    "US30":  {"ticker": "^DJI",   "spread": 2.0, "lot_mult": 5,  "min_sl": 30.0, "comm": 3.0},
    "SP500": {"ticker": "^GSPC",  "spread": 0.5, "lot_mult": 50, "min_sl": 10.0, "comm": 3.0},
    "NAS100":{"ticker": "^NDX",   "spread": 1.0, "lot_mult": 20, "min_sl": 20.0, "comm": 3.0},
    "RTY":   {"ticker": "^RUT",   "spread": 0.5, "lot_mult": 50, "min_sl": 5.0,  "comm": 3.0},
    "DAX":   {"ticker": "^GDAXI", "spread": 1.0, "lot_mult": 25, "min_sl": 20.0, "comm": 3.0},
}


def fetch_daily(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="5y", interval="1d", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def run_window(data: pd.DataFrame, cfg: dict, tradeable_idx) -> dict:
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_curve = [equity]

    for i, ts in enumerate(tradeable_idx):
        if ts not in data.index:
            eq_curve.append(equity); continue
        bar_h = float(data.loc[ts, "high"])
        bar_l = float(data.loc[ts, "low"])
        bar_c = float(data.loc[ts, "close"])

        # Manage open position (daily bars)
        if pos is not None:
            bars_held = i - pos["bar"]
            close_it, reason, ep = False, "", bar_c

            if bar_l <= pos["sl"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            if not close_it and bar_h >= pos["tp"]:
                close_it, reason, ep = True, "TP", pos["tp"]
            # Both: SL wins (conservative)
            if bar_l <= pos["sl"] and bar_h >= pos["tp"]:
                close_it, reason, ep = True, "SL", pos["sl"]
            if not close_it and bars_held >= config.MR_TIME_STOP:
                close_it, reason, ep = True, "Time", bar_c

            if close_it:
                raw = (ep - pos["entry"]) * pos["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * pos["lots"]
                equity += raw - c
                trades.append({"pnl": raw - c, "reason": reason, "ts": ts})
                pos = None

        # New entry at close of day (uses today's bar that just closed)
        if pos is None:
            loc = data.index.get_loc(ts)
            if loc >= 20:
                w = data.iloc[max(0, loc - 30):loc + 1]
                sig = generate_mr_signal(w, bar_c, bar_h, bar_l, "MR")
                if sig is not None:
                    risk = abs(sig.entry - sig.sl)
                    if risk >= cfg["min_sl"] * 0.3:
                        rd = equity * config.MR_RISK_PCT
                        lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                        pos = {"side": "long", "entry": sig.entry, "sl": sig.sl,
                               "tp": sig.tp, "lots": lots, "bar": i}

        eq = equity
        if pos:
            eq += (bar_c - pos["entry"]) * pos["lots"] * cfg["lot_mult"]
        eq_curve.append(eq)

    return summarize(trades, eq_curve, tradeable_idx, equity)


def summarize(trades, eq_curve, tradeable, final_eq) -> dict:
    pnl = final_eq - config.ACCOUNT_SIZE
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
    days = (tradeable[-1] - tradeable[0]).days
    return {"days": days, "trades": len(trades), "wr": wr, "pnl": pnl,
            "pnl_pct": pnl / config.ACCOUNT_SIZE * 100, "pf": pf, "mdd": mdd}


def main():
    print(f"\n{'=' * 84}")
    print(f"  MEAN REVERSION — DAILY — close<5d-low + IBS<0.25, TP=SMA5, SL=1.5×ATR")
    print(f"{'=' * 84}")
    print(f"  {'Symbol':8s} {'TRAIN':>25s}  {'TEST':>25s}  {'VALIDATE':>25s}")
    print(f"  {'-' * 88}")

    survivors = []
    for sym, cfg in INSTRUMENTS.items():
        try:
            data = fetch_daily(cfg["ticker"])
            idx = data.index[20:]
            n = len(idx)
            if n < 200:
                print(f"  {sym:8s} insufficient data"); continue
            splits = [
                idx[:int(n * 0.40)],
                idx[int(n * 0.40):int(n * 0.70)],
                idx[int(n * 0.70):],
            ]
            rs = [run_window(data, cfg, s) for s in splits]

            def fmt(r):
                pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
                return f"{r['trades']:3d} {r['wr']:3.0f}% ${r['pnl']:>+5,.0f} {pf_s:>5s} {r['mdd']:>3.1f}%"

            print(f"  {sym:8s} {fmt(rs[0]):>25s}  {fmt(rs[1]):>25s}  {fmt(rs[2]):>25s}")
            val = rs[2]
            if val["pnl"] > 0 and val["mdd"] < 10 and val["pf"] > 1.0:
                survivors.append((sym, val))
            time.sleep(1)
        except Exception as e:
            print(f"  {sym:8s} ERROR: {e}")

    print(f"\n  SURVIVORS:")
    if not survivors:
        print(f"    (none)")
    else:
        for sym, v in survivors:
            print(f"    {sym:8s} VALIDATE: ${v['pnl']:+,.0f} | {v['wr']:.0f}% WR | PF {v['pf']:.2f} | DD {v['mdd']:.1f}%")
    print(f"{'=' * 84}\n")


if __name__ == "__main__":
    main()
