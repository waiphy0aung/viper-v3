"""
Unified backtester — single file, all instruments, phased simulation.
Uses the hybrid strategy. Wick-based SL/TP. Spread + commission.
"""

from __future__ import annotations

import argparse
import logging
import time

import pandas as pd
import yfinance as yf

import config
from strategy.hybrid import generate_signal

logging.basicConfig(level=logging.WARNING)


def fetch(ticker: str) -> dict[str, pd.DataFrame]:
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
    windows = config.INSTRUMENTS.get(sym, {}).get("session", [])
    return any(s <= hour < e for s, e in windows) if windows else True


def run_phased():
    # Fetch all instruments
    all_data = {}
    for sym, cfg in config.INSTRUMENTS.items():
        print(f"  Fetching {sym}...", end=" ", flush=True)
        all_data[sym] = fetch(cfg["ticker"])
        print(f"{len(all_data[sym]['1h'])} bars")
        time.sleep(1)

    # Common range
    starts = [d["1h"].index[0] for d in all_data.values()]
    ends = [d["1h"].index[-1] for d in all_data.values()]
    start, end = max(starts), min(ends)
    master = list(all_data.keys())[0]
    tradeable = all_data[master]["1h"].loc[start:end].index[200:]
    total = len(tradeable)

    phases = [
        ("Phase 1", config.PROFIT_TARGET_PHASE1),
        ("Phase 2", config.PROFIT_TARGET_PHASE2),
        ("Funded", None),
    ]

    print(f"\n{'='*70}")
    print(f"  VIPER v3 — Hybrid SMC+Indicator Backtest")
    print(f"  Instruments: {', '.join(config.INSTRUMENTS.keys())}")
    print(f"  Period: {tradeable[0].date()} to {tradeable[-1].date()} ({total} bars)")
    print(f"{'='*70}")

    bar = 0
    for phase_name, target_pct in phases:
        if bar >= total:
            print(f"\n  {phase_name}: No data."); break

        equity = config.ACCOUNT_SIZE
        target = target_pct * equity if target_pct else None
        floor = config.EQUITY_FLOOR

        print(f"\n  --- {phase_name} ---")
        print(f"  Account: ${equity:,} | Target: {'${:,.0f}'.format(target) if target else 'None'}")

        pos = {}
        trades = []
        eq_curve = [equity]
        daily_start = equity
        cur_date = None
        tot_comm = 0.0
        phase_start = bar
        blown = False

        while bar < total:
            ts = tradeable[bar]
            hour = ts.hour
            day = ts.date()
            if day != cur_date:
                cur_date = day
                daily_start = equity

            # --- Manage positions ---
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

                # SL on wick
                if p["side"] == "long" and bl <= p["sl"]:
                    close_it, reason, ep = True, "SL", p["sl"]
                elif p["side"] == "short" and bh >= p["sl"]:
                    close_it, reason, ep = True, "SL", p["sl"]

                # TP on wick
                if not close_it:
                    if p["side"] == "long" and bh >= p["tp"]:
                        close_it, reason, ep = True, "TP", p["tp"]
                    elif p["side"] == "short" and bl <= p["tp"]:
                        close_it, reason, ep = True, "TP", p["tp"]

                # Both — SL wins
                if p["side"] == "long" and bl <= p["sl"] and bh >= p["tp"]:
                    close_it, reason, ep = True, "SL", p["sl"]
                elif p["side"] == "short" and bh >= p["sl"] and bl <= p["tp"]:
                    close_it, reason, ep = True, "SL", p["sl"]

                # Time: 20 bars on 1H
                if not close_it and bars_held >= 20:
                    close_it, reason, ep = True, "Time", price

                if close_it:
                    raw = ((ep - p["entry"]) if p["side"] == "long" else
                           (p["entry"] - ep)) * p["lots"] * cfg["lot_mult"]
                    c = cfg["comm"] * p["lots"]
                    equity += raw - c
                    tot_comm += c
                    trades.append({"sym": sym, "pnl": raw - c, "reason": reason,
                                   "quality": p.get("quality", ""), "bars": bars_held})
                    del pos[sym]

                    if equity <= floor:
                        blown = True; break
                    if target and (equity - config.ACCOUNT_SIZE) >= target:
                        d = (bar - phase_start) / 24
                        print(f"  >>> {phase_name} PASSED in {d:.0f} days | ${equity:,.2f} | {len(trades)} trades")
                        bar += 1; break

            if blown or (target and (equity - config.ACCOUNT_SIZE) >= target):
                break

            # --- New entries (max 1 position) ---
            if len(pos) == 0 and not blown:
                dd = (daily_start - equity) / daily_start if equity < daily_start else 0
                if dd >= config.DAILY_DD_LIMIT * 0.8:
                    eq_curve.append(equity); bar += 1; continue

                dd_util = (1.0 - equity / config.ACCOUNT_SIZE) / config.MAX_DD_LIMIT if equity < config.ACCOUNT_SIZE else 0
                throttle = max(0.25, 1.0 - dd_util * 0.9)

                for sym in config.INSTRUMENTS:
                    if not in_session(sym, hour):
                        continue
                    cfg = config.INSTRUMENTS[sym]
                    d = all_data[sym]
                    d1h = d["1h"]
                    if ts not in d1h.index:
                        continue

                    loc = d1h.index.get_loc(ts)
                    w1h = d1h.iloc[max(0, loc - 199):loc + 1]
                    w4h = d["4h"].loc[:ts].iloc[-100:]
                    wd = d["daily"].loc[:ts].iloc[-60:]

                    price = float(w1h["close"].iloc[-1])
                    sig = generate_signal(w1h, w4h, wd, price, sym)

                    if sig is None:
                        continue

                    risk = abs(sig.entry - sig.sl)
                    if risk < cfg["min_sl"] * 0.3:
                        continue

                    risk_d = min(equity * config.BASE_RISK_PCT * sig.confidence * throttle,
                                 equity * config.MAX_RISK_CAP)
                    lots = round(risk_d / (risk * cfg["lot_mult"]), 2)
                    lots = max(0.01, min(0.10, lots))

                    pos[sym] = {
                        "side": sig.direction, "entry": sig.entry,
                        "sl": sig.sl, "tp": sig.tp,
                        "lots": lots, "bar": bar, "quality": sig.quality.value,
                    }
                    break

            eq = equity
            for sym, p in pos.items():
                d1h = all_data[sym]["1h"]
                if ts in d1h.index:
                    px = float(d1h.loc[ts, "close"])
                    eq += ((px - p["entry"]) if p["side"] == "long" else
                           (p["entry"] - px)) * p["lots"] * config.INSTRUMENTS[sym]["lot_mult"]
            eq_curve.append(eq)
            bar += 1

        # Summary
        pnl = equity - config.ACCOUNT_SIZE
        days = (bar - phase_start) / 24
        wins = sum(1 for t in trades if t["pnl"] > 0)
        wr = wins / len(trades) * 100 if trades else 0

        peak = eq_curve[0]
        mdd = max((peak - eq) / peak * 100 if peak > 0 else 0
                  for eq in eq_curve) if eq_curve else 0
        for eq in eq_curve:
            peak = max(peak, eq)

        status = "BLOWN" if blown else ("PASSED" if target and pnl >= target else "RUNNING" if not target else "NOT YET")

        print(f"\n  {phase_name}: {status}")
        print(f"  PnL: ${pnl:,.2f} ({pnl/config.ACCOUNT_SIZE*100:+.2f}%)")
        print(f"  Trades: {len(trades)} ({wins}W/{len(trades)-wins}L) WR: {wr:.1f}%")
        print(f"  Max DD: {mdd:.2f}% | Comm: ${tot_comm:,.2f} | Days: {days:.0f}")

        if trades:
            wp = [t["pnl"] for t in trades if t["pnl"] > 0]
            lp = [t["pnl"] for t in trades if t["pnl"] < 0]
            aw = sum(wp) / len(wp) if wp else 0
            al = sum(lp) / len(lp) if lp else 0
            gw, gl = sum(wp), abs(sum(lp))
            pf = gw / gl if gl > 0 else float("inf")
            pf_str = f"{pf:.2f}" if pf != float("inf") else "INF"
            print(f"  Avg Win: ${aw:,.2f} | Avg Loss: ${al:,.2f} | PF: {pf_str}")

            # Quality breakdown
            for q in ["A+", "A", "B"]:
                qt = [t for t in trades if t.get("quality") == q]
                if qt:
                    qw = sum(1 for t in qt if t["pnl"] > 0)
                    qp = sum(t["pnl"] for t in qt)
                    print(f"    {q:3s}: {len(qt)}T  {qw/len(qt)*100:.0f}%WR  ${qp:,.2f}")

            # Per instrument
            for sym in sorted(set(t["sym"] for t in trades)):
                st = [t for t in trades if t["sym"] == sym]
                sw = sum(1 for t in st if t["pnl"] > 0)
                sp = sum(t["pnl"] for t in st)
                print(f"    {sym:8s} {len(st):3d}T  {sw/len(st)*100:.0f}%WR  ${sp:>8,.2f}")

        if blown:
            print(f"\n  FAILED."); break

    print(f"\n{'='*70}")


if __name__ == "__main__":
    run_phased()
