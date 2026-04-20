"""
Portfolio walk-forward — runs MULTIPLE strategies across MULTIPLE instruments
simultaneously, sharing one equity curve. The real question: do uncorrelated
weak edges combine into a deployable strong edge?

Strategies:
- Monster (hybrid SMC+indicator, big R:R) — on SP500, US30
- ORB (opening range breakout, 1:2) — on SP500, US30, NAS100, DAX
- Scalper (HMA cross + ATR, 1:2) — on SP500, US30

Risk: 0.5% per trade (split across strategies). Max 3 concurrent positions.
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
from strategy.hybrid import generate_signal as gen_monster
from strategy.scalper import generate_scalper_signal
from strategy.orb import generate_orb_signal
from core.structure import find_swings

logging.basicConfig(level=logging.WARNING)

# Portfolio instruments (extend beyond config.INSTRUMENTS)
PORTFOLIO = {
    "SP500":  {"ticker": "ES=F",    "spread": 0.5, "lot_mult": 50, "min_sl": 10.0, "comm": 3.0, "session": [(13, 20)]},
    "US30":   {"ticker": "YM=F",    "spread": 2.0, "lot_mult": 5,  "min_sl": 30.0, "comm": 3.0, "session": [(13, 20)]},
    "NAS100": {"ticker": "NQ=F",    "spread": 1.0, "lot_mult": 20, "min_sl": 20.0, "comm": 3.0, "session": [(13, 20)]},
    "DAX":    {"ticker": "^GDAXI",  "spread": 1.0, "lot_mult": 25, "min_sl": 20.0, "comm": 3.0, "session": [(7, 16)]},
}

STRATS = ["orb"]          # pure ORB portfolio across 4 instruments
RISK_PER_TRADE = 0.01    # 1% per trade
MAX_CONCURRENT = 4        # one per instrument


def fetch(ticker: str) -> dict:
    r = {}
    for tf, interval, period in [("1h", "1h", "730d"), ("daily", "1d", "5y"), ("weekly", "1wk", "10y")]:
        d = yf.download(ticker, period=period, interval=interval, progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        r[tf] = d
    r["4h"] = r["1h"].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return r


def in_session(sym: str, hour: int) -> bool:
    w = PORTFOLIO[sym]["session"]
    return any(s <= hour < e for s, e in w)


def try_strategy(strat: str, sym: str, d: dict, ts, loc: int) -> object:
    """Return a Signal or None for the given strategy."""
    d1h = d["1h"]
    w1h = d1h.iloc[max(0, loc - 199):loc + 1]
    w4h = d["4h"].loc[:ts].iloc[-100:] if "4h" in d else pd.DataFrame()
    wd = d["daily"].loc[:ts].iloc[-60:] if "daily" in d else pd.DataFrame()
    price = float(w1h["close"].iloc[-1])
    bh = float(w1h["high"].iloc[-1])
    bl = float(w1h["low"].iloc[-1])

    if strat == "monster":
        if len(w1h) < 50 or len(w4h) < 10 or len(wd) < 15:
            return None
        return gen_monster(w1h, w4h, wd, price, sym)
    if strat == "scalper":
        if len(w1h) < 50 or len(w4h) < 10:
            return None
        return generate_scalper_signal(w1h, w4h, price, sym)
    if strat == "orb":
        if len(w1h) < 30:
            return None
        return generate_orb_signal(w1h, price, bh, bl, ts, sym)
    return None


def run_window(all_data: dict, tradeable: pd.DatetimeIndex) -> dict:
    equity = config.ACCOUNT_SIZE
    pos: dict = {}          # (sym, strat) -> position
    trades: list = []
    eq_curve = [equity]
    daily_eq = {}
    tot_comm = 0.0
    daily_trades_by_strat = {}  # (day, strat) -> count

    monster_cooldown_bar = 0

    for bar, ts in enumerate(tradeable):
        hour = ts.hour
        day = ts.date()
        if day not in daily_eq:
            daily_eq[day] = equity

        # Manage open positions
        for key in list(pos.keys()):
            sym, strat = key
            cfg = PORTFOLIO[sym]
            d1h = all_data[sym]["1h"]
            if ts not in d1h.index:
                continue
            p = pos[key]
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

            # Time stops per strategy
            if strat == "monster":
                tlimit = config.MONSTER_TIME_STOP if p.get("is_monster") else 20
            elif strat == "scalper":
                tlimit = config.SCALPER_TIME_STOP
            else:  # orb
                tlimit = config.ORB_TIME_STOP

            if not close_it and bars_held >= tlimit:
                close_it, reason, ep = True, "Time", price

            if close_it:
                raw = ((ep - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - ep)) * p["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * p["lots"]
                equity += raw - c
                tot_comm += c
                trades.append({"sym": sym, "strat": strat, "pnl": raw - c,
                               "reason": reason, "ts": ts})
                if strat == "monster" and p.get("is_monster"):
                    monster_cooldown_bar = bar + int(config.MONSTER_COOLDOWN_HOURS)
                del pos[key]

        # Daily DD check
        dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
        if dd >= config.DAILY_DD_LIMIT * 0.8:
            eq_curve.append(equity); continue

        # Equity floor
        if equity <= config.EQUITY_FLOOR:
            eq_curve.append(equity); continue

        # Open new positions — try each (sym, strat) combination
        if len(pos) < MAX_CONCURRENT:
            for sym in PORTFOLIO:
                if not in_session(sym, hour):
                    continue
                if sym not in all_data:
                    continue
                d1h = all_data[sym]["1h"]
                if ts not in d1h.index:
                    continue
                loc = d1h.index.get_loc(ts)

                for strat in STRATS:
                    key = (sym, strat)
                    if key in pos:
                        continue
                    if len(pos) >= MAX_CONCURRENT:
                        break
                    # Monster cooldown
                    if strat == "monster" and bar < monster_cooldown_bar:
                        continue
                    # Strategy/instrument pairing (skip expensive strategies on wrong instruments)
                    # ORB works across all instruments; monster/scalper only on SP500/US30
                    if strat in ("monster", "scalper") and sym not in ("SP500", "US30"):
                        continue
                    # Per-day cap for ORB (1/day/instrument)
                    if strat == "orb":
                        dkey = (day, sym, "orb")
                        if daily_trades_by_strat.get(dkey, 0) >= 1:
                            continue

                    sig = try_strategy(strat, sym, all_data[sym], ts, loc)
                    if sig is None:
                        continue
                    cfg = PORTFOLIO[sym]
                    risk = abs(sig.entry - sig.sl)
                    if risk < cfg["min_sl"] * 0.3:
                        continue
                    rd = equity * RISK_PER_TRADE
                    lots = max(0.01, min(0.10, round(rd / (risk * cfg["lot_mult"]), 2)))

                    pos[key] = {
                        "side": sig.direction, "entry": sig.entry, "sl": sig.sl,
                        "tp": sig.tp, "lots": lots, "bar": bar,
                        "is_monster": getattr(sig, "is_monster", False),
                    }
                    if strat == "orb":
                        daily_trades_by_strat[(day, sym, "orb")] = \
                            daily_trades_by_strat.get((day, sym, "orb"), 0) + 1

        # Mark-to-market equity
        eq = equity
        for key, p in pos.items():
            sym = key[0]
            d1h = all_data[sym]["1h"]
            if ts in d1h.index:
                px = float(d1h.loc[ts, "close"])
                eq += ((px - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - px)) * p["lots"] * PORTFOLIO[sym]["lot_mult"]
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
    if trades:
        df = pd.DataFrame([{"ts": t["ts"], "pnl": t["pnl"], "strat": t["strat"], "sym": t["sym"]} for t in trades])
        df["ym"] = df["ts"].dt.to_period("M")
        monthly = df.groupby("ym")["pnl"].sum()
        total_months = len(monthly)
        pct_pos = (monthly > 0).sum() / total_months * 100 if total_months else 0
        avg_monthly = mean([m / config.ACCOUNT_SIZE * 100 for m in monthly.values])
        # Per-strategy breakdown
        by_strat = df.groupby("strat").agg({"pnl": ["sum", "count"]})
        by_sym = df.groupby("sym").agg({"pnl": ["sum", "count"]})
    else:
        pct_pos = 0; avg_monthly = 0; total_months = 0
        by_strat = None; by_sym = None

    days = (tradeable[-1] - tradeable[0]).days
    return {
        "days": days, "trades": len(trades), "wr": wr, "pnl": pnl,
        "pnl_pct": pnl / config.ACCOUNT_SIZE * 100, "pf": pf, "mdd": mdd,
        "pct_pos_months": pct_pos, "avg_monthly_pct": avg_monthly,
        "by_strat": by_strat, "by_sym": by_sym,
    }


def main():
    all_data = {}
    for sym, cfg in PORTFOLIO.items():
        print(f"  Fetching {sym} ({cfg['ticker']})...", end=" ", flush=True)
        try:
            all_data[sym] = fetch(cfg["ticker"])
            print(f"{len(all_data[sym]['1h'])} bars")
        except Exception as e:
            print(f"FAILED: {e}")
        time.sleep(1)

    # Use SP500 as master timeline
    idx = all_data["SP500"]["1h"].index[200:]
    n = len(idx)
    splits = [
        ("TRAIN",    idx[:int(n * 0.40)]),
        ("TEST",     idx[int(n * 0.40):int(n * 0.70)]),
        ("VALIDATE", idx[int(n * 0.70):]),
    ]

    print(f"\n{'=' * 78}")
    print(f"  PORTFOLIO WALK-FORWARD")
    print(f"  Strategies: {', '.join(STRATS)} | Risk: {RISK_PER_TRADE*100:.1f}%/trade | Max concurrent: {MAX_CONCURRENT}")
    print(f"  Instruments: {', '.join(all_data.keys())}")
    print(f"{'=' * 78}")

    results = []
    for label, window in splits:
        print(f"\n  {label} ({window[0].date()} → {window[-1].date()})...")
        r = run_window(all_data, window)
        r["label"] = label
        results.append(r)

    print(f"\n{'=' * 78}")
    print(f"  AGGREGATE RESULTS")
    print(f"{'=' * 78}")
    print(f"  {'Window':10s} {'Days':>5s} {'Trades':>7s} {'WR':>6s} {'PnL':>10s} {'PF':>6s} {'MaxDD':>7s} {'+Mo':>5s} {'AvgMo':>7s}")
    print(f"  {'-' * 76}")
    for r in results:
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        print(f"  {r['label']:10s} {r['days']:>5d} {r['trades']:>7d} {r['wr']:>5.1f}% "
              f"${r['pnl']:>7,.0f} {pf_s:>6s} {r['mdd']:>6.2f}% "
              f"{r['pct_pos_months']:>4.0f}% {r['avg_monthly_pct']:>+6.2f}%")

    print(f"\n  PER-STRATEGY (VALIDATE window):")
    val = results[-1]
    if val["by_strat"] is not None:
        print(val["by_strat"])
    print(f"\n  PER-INSTRUMENT (VALIDATE window):")
    if val["by_sym"] is not None:
        print(val["by_sym"])

    # Verdict
    all_positive = all(r["pnl_pct"] > 0 for r in results)
    val_month_pct = val["avg_monthly_pct"]
    print(f"\n  Funding Pips viability: {val_month_pct:+.2f}%/month (need 8% in 30 days)")
    if val_month_pct >= 5:
        print(f"    DEPLOYABLE — hits prop firm speed threshold")
    elif val_month_pct >= 2:
        print(f"    Viable for personal account but too slow for prop sprint")
    elif val_month_pct > 0:
        print(f"    Positive but too weak for any real goal")
    else:
        print(f"    Portfolio still unprofitable in VALIDATE")

    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
