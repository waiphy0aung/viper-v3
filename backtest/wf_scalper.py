"""
Walk-forward validation for SCALPER strategy.
Same 40/30/30 split as walkforward.py but uses strategy.scalper.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from statistics import mean, stdev

import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from strategy.scalper import generate_scalper_signal

logging.basicConfig(level=logging.WARNING)


def fetch(ticker: str) -> dict[str, pd.DataFrame]:
    r = {}
    for tf, interval, period in [("1h", "1h", "730d")]:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        r[tf] = d
    r["4h"] = r["1h"].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return r


def in_session(sym: str, hour: int) -> bool:
    w = config.INSTRUMENTS.get(sym, {}).get("session", [])
    return any(s <= hour < e for s, e in w) if w else True


def run_window(all_data: dict, tradeable: pd.DatetimeIndex) -> dict:
    equity = config.ACCOUNT_SIZE
    pos: dict = {}
    trades: list = []
    eq_curve = [equity]
    daily_eq = {}
    tot_comm = 0.0

    for bar, ts in enumerate(tradeable):
        hour = ts.hour
        day = ts.date()
        if day not in daily_eq:
            daily_eq[day] = equity

        for sym in list(pos.keys()):
            cfg = config.INSTRUMENTS[sym]
            d1h = all_data[sym]["1h"]
            if ts not in d1h.index:
                continue
            p = pos[sym]
            price = float(d1h.loc[ts, "close"])
            bh = float(d1h.loc[ts, "high"])
            bl = float(d1h.loc[ts, "low"])
            bars_held = bar - p["bar"]
            close_it, reason, ep = False, "", price

            if p["side"] == "long" and bl <= p["sl"]:
                close_it, reason, ep = True, "SL", p["sl"]
            elif p["side"] == "short" and bh >= p["sl"]:
                close_it, reason, ep = True, "SL", p["sl"]
            if not close_it:
                if p["side"] == "long" and bh >= p["tp"]:
                    close_it, reason, ep = True, "TP", p["tp"]
                elif p["side"] == "short" and bl <= p["tp"]:
                    close_it, reason, ep = True, "TP", p["tp"]
            if p["side"] == "long" and bl <= p["sl"] and bh >= p["tp"]:
                close_it, reason, ep = True, "SL", p["sl"]
            elif p["side"] == "short" and bh >= p["sl"] and bl <= p["tp"]:
                close_it, reason, ep = True, "SL", p["sl"]

            if not close_it and bars_held >= config.SCALPER_TIME_STOP:
                close_it, reason, ep = True, "Time", price

            if close_it:
                raw = ((ep - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - ep)) * p["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * p["lots"]
                equity += raw - c
                tot_comm += c
                trades.append({"sym": sym, "pnl": raw - c, "reason": reason,
                               "quality": p.get("quality", ""), "ts": ts})
                del pos[sym]

        if len(pos) == 0:
            dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
            if dd < config.DAILY_DD_LIMIT * 0.8:
                for sym in config.INSTRUMENTS:
                    if not in_session(sym, hour):
                        continue
                    cfg = config.INSTRUMENTS[sym]
                    im = cfg.get("months")
                    if config.SEASONAL_FILTER and im and ts.month not in im:
                        continue
                    d = all_data[sym]
                    d1h = d["1h"]
                    if ts not in d1h.index:
                        continue
                    loc = d1h.index.get_loc(ts)
                    w1h = d1h.iloc[max(0, loc - 199):loc + 1]
                    w4h = d["4h"].loc[:ts].iloc[-100:]
                    if len(w1h) < 50 or len(w4h) < 10:
                        continue
                    price = float(w1h["close"].iloc[-1])
                    sig = generate_scalper_signal(w1h, w4h, price, sym)
                    if sig is None:
                        continue
                    risk = abs(sig.entry - sig.sl)
                    if risk < cfg["min_sl"] * 0.3:
                        continue
                    rd = equity * config.SCALPER_RISK_PCT
                    lots = max(0.01, min(0.10, round(rd / (risk * cfg["lot_mult"]), 2)))
                    pos[sym] = {"side": sig.direction, "entry": sig.entry, "sl": sig.sl,
                                "tp": sig.tp, "lots": lots, "bar": bar,
                                "quality": sig.quality.value}
                    break

        eq = equity
        for sym, p in pos.items():
            d1h = all_data[sym]["1h"]
            if ts in d1h.index:
                px = float(d1h.loc[ts, "close"])
                eq += ((px - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - px)) * p["lots"] * config.INSTRUMENTS[sym]["lot_mult"]
        eq_curve.append(eq)

    return summarize(trades, eq_curve, tradeable, equity, tot_comm)


def summarize(trades, eq_curve, tradeable, final_eq, comm) -> dict:
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

    if trades:
        df = pd.DataFrame([{"ts": t["ts"], "pnl": t["pnl"]} for t in trades])
        df["ym"] = df["ts"].dt.to_period("M")
        monthly = df.groupby("ym")["pnl"].sum()
        total_months = len(monthly)
        pct_pos = (monthly > 0).sum() / total_months * 100 if total_months else 0
        monthly_pct = [m / config.ACCOUNT_SIZE * 100 for m in monthly.values]
        avg_monthly = mean(monthly_pct) if monthly_pct else 0
    else:
        pct_pos = 0; avg_monthly = 0; total_months = 0

    days = (tradeable[-1] - tradeable[0]).days
    return {
        "days": days, "trades": len(trades), "wr": wr, "pnl": pnl,
        "pnl_pct": pnl / config.ACCOUNT_SIZE * 100, "pf": pf, "mdd": mdd,
        "months": total_months, "pct_pos_months": pct_pos,
        "avg_monthly_pct": avg_monthly,
        "start": tradeable[0].date(), "end": tradeable[-1].date(),
    }


def main():
    all_data = {}
    for sym, cfg in config.INSTRUMENTS.items():
        print(f"  Fetching {sym}...", end=" ", flush=True)
        all_data[sym] = fetch(cfg["ticker"])
        print(f"{len(all_data[sym]['1h'])} bars")
        time.sleep(1)

    starts = [d["1h"].index[0] for d in all_data.values()]
    ends = [d["1h"].index[-1] for d in all_data.values()]
    start, end = max(starts), min(ends)
    master = list(all_data.keys())[0]
    idx = all_data[master]["1h"].loc[start:end].index[200:]
    n = len(idx)
    splits = [
        ("TRAIN",    idx[:int(n * 0.40)]),
        ("TEST",     idx[int(n * 0.40):int(n * 0.70)]),
        ("VALIDATE", idx[int(n * 0.70):]),
    ]

    print(f"\n{'=' * 78}")
    print(f"  VIPER v3 — SCALPER Walk-Forward Validation")
    print(f"  Config: ATR SL × {config.SCALPER_SL_ATR_MULT}, 1:{config.SCALPER_RR} TP, "
          f"{config.SCALPER_TIME_STOP}-bar stop, {config.SCALPER_RISK_PCT*100:.1f}% risk")
    print(f"  Instruments: {', '.join(config.INSTRUMENTS.keys())}")
    print(f"{'=' * 78}")

    results = []
    for label, window in splits:
        print(f"\n  Running {label} ({window[0].date()} → {window[-1].date()}, {len(window)} bars)...")
        r = run_window(all_data, window)
        r["label"] = label
        results.append(r)

    print(f"\n{'=' * 78}")
    print(f"  RESULTS")
    print(f"{'=' * 78}")
    print(f"  {'Window':10s} {'Days':>5s} {'Trades':>7s} {'WR':>6s} {'PnL':>10s} {'PF':>6s} {'MaxDD':>7s} {'+Months':>8s} {'Avg/Mo':>8s}")
    print(f"  {'-' * 76}")
    for r in results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        print(f"  {r['label']:10s} {r['days']:>5d} {r['trades']:>7d} {r['wr']:>5.1f}% "
              f"${r['pnl']:>7,.0f} {pf_s:>6s} {r['mdd']:>6.2f}% "
              f"{r['pct_pos_months']:>6.0f}% {r['avg_monthly_pct']:>+6.2f}%")

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
        print(f"    Edge exists but UNSTABLE — metrics swing between windows")
    else:
        neg = [r["label"] for r in results if r["pnl_pct"] <= 0]
        print(f"    LIKELY CURVE-FIT — losing window(s): {', '.join(neg)}")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
