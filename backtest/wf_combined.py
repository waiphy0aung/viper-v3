"""
Combined walk-forward — ORB on US30 + RTY + DAX, shared equity, stacked edges.
Different sessions (DAX European morning, US30+RTY NY) → low correlation → smoother curve.
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

logging.basicConfig(level=logging.WARNING)

INSTRUMENTS = {
    "US30": {"ticker": "YM=F",   "spread": 2.0, "lot_mult": 5,  "min_sl": 30.0, "comm": 3.0, "orb_hour": 13},
    "RTY":  {"ticker": "RTY=F",  "spread": 0.5, "lot_mult": 50, "min_sl": 5.0,  "comm": 3.0, "orb_hour": 13},
    "DAX":  {"ticker": "^GDAXI", "spread": 1.0, "lot_mult": 25, "min_sl": 20.0, "comm": 3.0, "orb_hour": 7},
}
ORB_TP_MULT = 2.0
ORB_TIME_STOP = 5
RISK_PCT = 0.010
MAX_CONCURRENT = 2


def fetch(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="730d", interval="1h", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def orb_range(df, now_ts, orb_hour):
    start = now_ts.replace(hour=orb_hour, minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=1)
    session = df.loc[start:end]
    if len(session) < 1: return 0, 0, 0, False
    h = float(session["high"].max()); l = float(session["low"].min())
    size = h - l
    return h, l, size, size > 0


def gen_signal(df, price, bh, bl, now_ts, cfg):
    hour = now_ts.hour
    orb_h = cfg["orb_hour"]
    if hour < orb_h + 1 or hour >= orb_h + 5: return None
    h, l, size, valid = orb_range(df, now_ts, orb_h)
    if not valid or size < cfg["min_sl"]: return None
    direction = None
    if bh > h and price > h: direction = "long"
    elif bl < l and price < l: direction = "short"
    else: return None
    daily = df["close"].resample("1D").last().dropna()
    if len(daily) >= 20:
        sma20 = float(daily.tail(20).mean())
        last = float(daily.iloc[-1])
        if direction == "long" and last < sma20: return None
        if direction == "short" and last > sma20: return None
    buf = size * 0.1
    if direction == "long":
        sl = l - buf; tp = price + size * ORB_TP_MULT; fill = price + cfg["spread"]
    else:
        sl = h + buf; tp = price - size * ORB_TP_MULT; fill = price - cfg["spread"]
    risk = abs(fill - sl)
    if risk < cfg["min_sl"] * 0.5: return None
    return {"side": direction, "entry": fill, "sl": sl, "tp": tp}


def run_window(all_data: dict, tradeable) -> dict:
    equity = config.ACCOUNT_SIZE
    pos: dict = {}
    trades: list = []
    eq_curve = [equity]
    daily_eq = {}
    daily_orb_taken: dict = {}
    circuit_breaker_until = None  # date to resume after rolling DD breach

    for bar, ts in enumerate(tradeable):
        day = ts.date()
        if day not in daily_eq: daily_eq[day] = equity

        # Manage positions
        for sym in list(pos.keys()):
            cfg = INSTRUMENTS[sym]
            data = all_data[sym]
            if ts not in data.index: continue
            p = pos[sym]
            price = float(data.loc[ts, "close"])
            bh = float(data.loc[ts, "high"])
            bl = float(data.loc[ts, "low"])
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
            if not close_it and bars_held >= ORB_TIME_STOP:
                close_it, reason, ep = True, "Time", price
            if not close_it and ts.hour >= cfg["orb_hour"] + 7:
                close_it, reason, ep = True, "SessionEnd", price

            if close_it:
                raw = ((ep - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - ep)) * p["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * p["lots"]
                equity += raw - c
                trades.append({"sym": sym, "pnl": raw - c, "reason": reason, "ts": ts})
                del pos[sym]

        # Daily DD check — pause all new entries if today down 4%
        dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
        if dd >= 0.04:
            eq_curve.append(equity); continue
        if equity <= config.EQUITY_FLOOR:
            eq_curve.append(equity); continue

        # Rolling 10-day DD circuit breaker — pause 5 days if lost >5% from 10-day high
        if circuit_breaker_until and day < circuit_breaker_until:
            eq_curve.append(equity); continue
        daily_vals = list(daily_eq.values())
        if len(daily_vals) >= 10:
            recent10_max = max(daily_vals[-10:])
            if recent10_max > 0 and (recent10_max - equity) / recent10_max >= 0.05:
                circuit_breaker_until = (pd.Timestamp(day) + pd.Timedelta(days=5)).date()
                eq_curve.append(equity); continue

        # New entries
        if len(pos) < MAX_CONCURRENT:
            for sym, cfg in INSTRUMENTS.items():
                if sym in pos: continue
                if len(pos) >= MAX_CONCURRENT: break
                if sym not in all_data: continue
                data = all_data[sym]
                if ts not in data.index: continue
                dkey = (day, sym)
                if daily_orb_taken.get(dkey, 0) >= 1: continue
                loc = data.index.get_loc(ts)
                if loc < 500: continue
                w = data.iloc[max(0, loc - 500):loc + 1]
                price = float(data.loc[ts, "close"])
                bh = float(data.loc[ts, "high"])
                bl = float(data.loc[ts, "low"])
                sig = gen_signal(w, price, bh, bl, ts, cfg)
                if sig is None: continue
                risk = abs(sig["entry"] - sig["sl"])
                if risk < cfg["min_sl"] * 0.5: continue
                rd = equity * RISK_PCT
                lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                pos[sym] = {"side": sig["side"], "entry": sig["entry"], "sl": sig["sl"],
                            "tp": sig["tp"], "lots": lots, "bar": bar}
                daily_orb_taken[dkey] = 1

        # MTM
        eq = equity
        for sym, p in pos.items():
            cfg = INSTRUMENTS[sym]
            data = all_data[sym]
            if ts in data.index:
                px = float(data.loc[ts, "close"])
                eq += ((px - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - px)) * p["lots"] * cfg["lot_mult"]
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
    per_sym = {}
    if trades:
        df = pd.DataFrame(trades)
        df["ym"] = df["ts"].dt.to_period("M")
        monthly = df.groupby("ym")["pnl"].sum()
        pct_pos = (monthly > 0).sum() / len(monthly) * 100 if len(monthly) else 0
        avg_monthly = mean([m / config.ACCOUNT_SIZE * 100 for m in monthly.values])
        for sym, g in df.groupby("sym"):
            per_sym[sym] = {"trades": len(g), "pnl": g["pnl"].sum()}
    else:
        pct_pos = 0; avg_monthly = 0

    days = (tradeable[-1] - tradeable[0]).days
    return {"days": days, "trades": len(trades), "wr": wr, "pnl": pnl,
            "pnl_pct": pnl / config.ACCOUNT_SIZE * 100, "pf": pf, "mdd": mdd,
            "pct_pos_months": pct_pos, "avg_monthly_pct": avg_monthly, "per_sym": per_sym}


def main():
    data = {}
    for sym, cfg in INSTRUMENTS.items():
        print(f"  Fetching {sym} ({cfg['ticker']})...", end=" ", flush=True)
        data[sym] = fetch(cfg["ticker"])
        print(f"{len(data[sym])} bars")
        time.sleep(1)

    # Use US30 as master timeline (most coverage)
    idx = data["US30"].index[500:]
    n = len(idx)
    splits = [
        ("TRAIN",    idx[:int(n * 0.40)]),
        ("TEST",     idx[int(n * 0.40):int(n * 0.70)]),
        ("VALIDATE", idx[int(n * 0.70):]),
    ]

    print(f"\n{'=' * 84}")
    print(f"  COMBINED ORB PORTFOLIO — US30 + RTY + DAX")
    print(f"  Risk: {RISK_PCT*100:.1f}% per trade | Max concurrent: {MAX_CONCURRENT}")
    print(f"  Daily DD breaker: 4% | Trend filter ON")
    print(f"{'=' * 84}")

    for label, window in splits:
        print(f"\n  {label} ({window[0].date()} → {window[-1].date()})...")
        r = run_window(data, window)
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        print(f"    Trades: {r['trades']} | WR: {r['wr']:.1f}% | PnL: ${r['pnl']:+,.0f} "
              f"({r['pnl_pct']:+.2f}%) | PF: {pf_s} | MaxDD: {r['mdd']:.2f}%")
        print(f"    Monthly: {r['avg_monthly_pct']:+.2f}% | +Months: {r['pct_pos_months']:.0f}%")
        for sym, stats in r["per_sym"].items():
            print(f"      {sym:6s} {stats['trades']:3d}T  ${stats['pnl']:>+7,.2f}")

    print(f"\n{'=' * 84}\n")


if __name__ == "__main__":
    main()
