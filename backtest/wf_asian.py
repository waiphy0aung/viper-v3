"""
Walk-forward validation for Asian Session Breakout on GOLD.
Adds GOLD instrument if not already in config.
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
from strategy.asian_breakout import generate_asian_breakout_signal

logging.basicConfig(level=logging.WARNING)

# GOLD config override — added locally so we don't touch INSTRUMENTS
GOLD_CFG = {
    "ticker": "GC=F", "spread": 0.3, "lot_mult": 10,
    "min_sl": 1.5, "comm": 3.0,
    "session": [(7, 15)],  # London-NY overlap
    "months": [],
}


def fetch(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="730d", interval="1h", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def run_window(data: pd.DataFrame, tradeable: pd.DatetimeIndex) -> dict:
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_curve = [equity]
    daily_eq = {}
    tot_comm = 0.0
    daily_trade_taken = set()

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

            if not close_it and bars_held >= config.ASIAN_TIME_STOP:
                close_it, reason, ep = True, "Time", price

            # Force close at end of London-NY window
            if not close_it and ts.hour >= 15:
                close_it, reason, ep = True, "SessionEnd", price

            if close_it:
                raw = ((ep - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - ep)) * pos["lots"] * GOLD_CFG["lot_mult"]
                c = GOLD_CFG["comm"] * pos["lots"]
                equity += raw - c
                tot_comm += c
                trades.append({"pnl": raw - c, "reason": reason, "ts": ts, "side": pos["side"]})
                pos = None

        if pos is None and day not in daily_trade_taken:
            dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
            if dd < config.DAILY_DD_LIMIT * 0.8 and ts in data.index:
                loc = data.index.get_loc(ts)
                if loc >= 30:
                    w = data.iloc[max(0, loc - 30):loc + 1]
                    price = float(data.loc[ts, "close"])
                    bh = float(data.loc[ts, "high"])
                    bl = float(data.loc[ts, "low"])
                    sig = generate_asian_breakout_signal(w, price, bh, bl, ts, "GOLD")
                    if sig is not None:
                        risk = abs(sig.entry - sig.sl)
                        if risk >= GOLD_CFG["min_sl"] * 0.5:
                            rd = equity * config.ASIAN_RISK_PCT
                            lots = max(0.01, min(0.10, round(rd / (risk * GOLD_CFG["lot_mult"]), 2)))
                            pos = {"side": sig.direction, "entry": sig.entry, "sl": sig.sl,
                                   "tp": sig.tp, "lots": lots, "bar": bar}
                            daily_trade_taken.add(day)

        eq = equity
        if pos and ts in data.index:
            px = float(data.loc[ts, "close"])
            eq += ((px - pos["entry"]) if pos["side"] == "long" else
                   (pos["entry"] - px)) * pos["lots"] * GOLD_CFG["lot_mult"]
        eq_curve.append(eq)

    return summarize(trades, eq_curve, tradeable, equity)


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
    aw = sum(wp) / len(wp) if wp else 0
    al = sum(lp) / len(lp) if lp else 0

    if trades:
        df = pd.DataFrame([{"ts": t["ts"], "pnl": t["pnl"]} for t in trades])
        df["ym"] = df["ts"].dt.to_period("M")
        monthly = df.groupby("ym")["pnl"].sum()
        total_months = len(monthly)
        pct_pos = (monthly > 0).sum() / total_months * 100 if total_months else 0
        avg_monthly = mean([m / config.ACCOUNT_SIZE * 100 for m in monthly.values])
    else:
        pct_pos = 0; avg_monthly = 0; total_months = 0

    days = (tradeable[-1] - tradeable[0]).days
    return {
        "days": days, "trades": len(trades), "wr": wr, "pnl": pnl,
        "pnl_pct": pnl / config.ACCOUNT_SIZE * 100, "pf": pf, "mdd": mdd,
        "avg_win": aw, "avg_loss": al,
        "months": total_months, "pct_pos_months": pct_pos, "avg_monthly_pct": avg_monthly,
    }


def main():
    print(f"  Fetching GOLD (GC=F)...", end=" ", flush=True)
    data = fetch("GC=F")
    print(f"{len(data)} bars")

    idx = data.index[30:]
    n = len(idx)
    splits = [
        ("TRAIN",    idx[:int(n * 0.40)]),
        ("TEST",     idx[int(n * 0.40):int(n * 0.70)]),
        ("VALIDATE", idx[int(n * 0.70):]),
    ]

    print(f"\n{'=' * 78}")
    print(f"  ASIAN SESSION BREAKOUT — GOLD (GC=F)")
    print(f"  Range window: 00:00-07:00 UTC. Entry: 07:00-09:00 UTC break.")
    print(f"  TP mult: {config.ASIAN_TP_MULT}× range | Time stop: {config.ASIAN_TIME_STOP} bars | Risk: {config.ASIAN_RISK_PCT*100:.1f}%")
    print(f"{'=' * 78}")

    results = []
    for label, window in splits:
        print(f"\n  {label} ({window[0].date()} → {window[-1].date()}, {len(window)} bars)...")
        r = run_window(data, window)
        r["label"] = label
        results.append(r)

    print(f"\n{'=' * 78}")
    print(f"  RESULTS")
    print(f"{'=' * 78}")
    print(f"  {'Window':10s} {'Days':>5s} {'Trades':>7s} {'WR':>6s} {'PnL':>10s} {'PF':>6s} {'MaxDD':>7s} {'AvgW':>7s} {'AvgL':>7s} {'+Mo':>5s}")
    print(f"  {'-' * 84}")
    for r in results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        print(f"  {r['label']:10s} {r['days']:>5d} {r['trades']:>7d} {r['wr']:>5.1f}% "
              f"${r['pnl']:>7,.0f} {pf_s:>6s} {r['mdd']:>6.2f}% "
              f"${r['avg_win']:>5,.0f} ${r['avg_loss']:>5,.0f} {r['pct_pos_months']:>4.0f}%")

    print(f"\n  VERDICT:")
    pfs = [r["pf"] for r in results if r["pf"] != float("inf")]
    wrs = [r["wr"] for r in results]
    pnls_pct = [r["pnl_pct"] for r in results]
    all_positive = all(p > 0 for p in pnls_pct)
    pf_stable = max(pfs) / min(pfs) < 2.5 if pfs and min(pfs) > 0 else False
    wr_stable = max(wrs) - min(wrs) < 20 if wrs else False

    if all_positive and pf_stable and wr_stable:
        print(f"    EDGE IS REAL — all windows positive, metrics stable")
    elif all_positive:
        print(f"    Edge exists but UNSTABLE")
    else:
        neg = [r["label"] for r in results if r["pnl_pct"] <= 0]
        print(f"    LIKELY CURVE-FIT — losing window(s): {', '.join(neg)}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
