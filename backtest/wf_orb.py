"""Walk-forward for ORB on SP500."""

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
from strategy.orb import generate_orb_signal

logging.basicConfig(level=logging.WARNING)


def fetch(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="730d", interval="1h", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def run_window(data: pd.DataFrame, cfg: dict, sym: str, tradeable: pd.DatetimeIndex) -> dict:
    equity = config.ACCOUNT_SIZE
    pos = None
    trades = []
    eq_curve = [equity]
    daily_eq = {}
    daily_trade_taken = set()
    tot_comm = 0.0

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
            if not close_it and bars_held >= config.ORB_TIME_STOP:
                close_it, reason, ep = True, "Time", price
            if not close_it and ts.hour >= 20:
                close_it, reason, ep = True, "SessionEnd", price

            if close_it:
                raw = ((ep - pos["entry"]) if pos["side"] == "long" else
                       (pos["entry"] - ep)) * pos["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * pos["lots"]
                equity += raw - c
                tot_comm += c
                trades.append({"pnl": raw - c, "reason": reason, "ts": ts, "side": pos["side"]})
                pos = None

        if pos is None and day not in daily_trade_taken:
            dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
            if dd < config.DAILY_DD_LIMIT * 0.8 and ts in data.index:
                loc = data.index.get_loc(ts)
                if loc >= 500:
                    w = data.iloc[max(0, loc - 500):loc + 1]
                    price = float(data.loc[ts, "close"])
                    bh = float(data.loc[ts, "high"])
                    bl = float(data.loc[ts, "low"])
                    sig = generate_orb_signal(w, price, bh, bl, ts, sym)
                    if sig is not None:
                        risk = abs(sig.entry - sig.sl)
                        if risk >= cfg["min_sl"] * 0.5:
                            rd = equity * config.ORB_RISK_PCT
                            lots = max(0.01, min(0.10, round(rd / (risk * cfg["lot_mult"]), 2)))
                            pos = {"side": sig.direction, "entry": sig.entry, "sl": sig.sl,
                                   "tp": sig.tp, "lots": lots, "bar": bar}
                            daily_trade_taken.add(day)

        eq = equity
        if pos and ts in data.index:
            px = float(data.loc[ts, "close"])
            eq += ((px - pos["entry"]) if pos["side"] == "long" else
                   (pos["entry"] - px)) * pos["lots"] * cfg["lot_mult"]
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
        "avg_win": aw, "avg_loss": al, "pct_pos_months": pct_pos, "avg_monthly_pct": avg_monthly,
    }


def main():
    for sym in ["US30"]:  # only US30 — other instruments failed walk-forward
        cfg = config.INSTRUMENTS[sym]
        print(f"\n  Fetching {sym} ({cfg['ticker']})...", end=" ", flush=True)
        data = fetch(cfg["ticker"])
        print(f"{len(data)} bars")

        idx = data.index[30:]
        n = len(idx)
        splits = [
            ("TRAIN",    idx[:int(n * 0.40)]),
            ("TEST",     idx[int(n * 0.40):int(n * 0.70)]),
            ("VALIDATE", idx[int(n * 0.70):]),
        ]

        print(f"\n{'=' * 78}")
        print(f"  ORB — {sym}  |  ORB window 13:00-14:00 UTC  |  TP={config.ORB_TP_MULT}×range  |  stop={config.ORB_TIME_STOP} bars")
        print(f"{'=' * 78}")

        results = []
        for label, window in splits:
            r = run_window(data, cfg, sym, window)
            r["label"] = label
            results.append(r)

        print(f"  {'Window':10s} {'Days':>5s} {'Trades':>7s} {'WR':>6s} {'PnL':>10s} {'PF':>6s} {'MaxDD':>7s} {'AvgW':>7s} {'AvgL':>7s} {'+Mo':>5s}")
        print(f"  {'-' * 84}")
        for r in results:
            pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
            print(f"  {r['label']:10s} {r['days']:>5d} {r['trades']:>7d} {r['wr']:>5.1f}% "
                  f"${r['pnl']:>7,.0f} {pf_s:>6s} {r['mdd']:>6.2f}% "
                  f"${r['avg_win']:>5,.0f} ${r['avg_loss']:>5,.0f} {r['pct_pos_months']:>4.0f}%")
        time.sleep(1)


if __name__ == "__main__":
    main()
