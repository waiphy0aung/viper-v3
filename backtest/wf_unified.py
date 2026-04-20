"""
Unified walk-forward — ORB (1H, intraday) + MR (daily, end-of-day) on ONE equity.

Edges:
- ORB on US30, RTY, DAX (NY/EU open breakouts, intraday 1:2 TP)
- MR on US30, SP500, RTY (daily close<5d-low + IBS<0.25, hold to SMA5)

Negatively correlated in TEST regime → stack for smoother equity curve.
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
from strategy.mean_reversion import ibs as mr_ibs

logging.basicConfig(level=logging.WARNING)

# Instrument config — reused across both strategies
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

# Strategy params
ORB_TP_MULT = 2.0
ORB_TIME_STOP = 5                       # 1H bars
ORB_RISK_PCT = 0.01
MR_SL_ATR_MULT = 1.5
MR_TIME_STOP = 5 * 24                   # 5 days × 24 hours
MR_RISK_PCT = 0.01
MAX_CONCURRENT = 3


def fetch(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="730d", interval="1h", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.resample("1D").agg({"open": "first", "high": "max",
                                    "low": "min", "close": "last",
                                    "volume": "sum"}).dropna()


# ============================================================
# ORB signal
# ============================================================
def orb_range(df, now_ts, orb_hour):
    start = now_ts.replace(hour=orb_hour, minute=0, second=0, microsecond=0)
    end = start + pd.Timedelta(hours=1)
    s = df.loc[start:end]
    if len(s) < 1: return 0, 0, 0, False
    h, l = float(s["high"].max()), float(s["low"].min())
    size = h - l
    return h, l, size, size > 0


def gen_orb(df, price, bh, bl, now_ts, cfg):
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
        sma20 = float(daily.tail(20).mean()); last = float(daily.iloc[-1])
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


# ============================================================
# MR signal (daily)
# ============================================================
def gen_mr(daily_df, price, bh, bl, cfg):
    if len(daily_df) < 10: return None
    prior5_low = float(daily_df["low"].iloc[-6:-1].min())
    if price >= prior5_low: return None
    if mr_ibs(bh, bl, price) >= 0.25: return None
    daily_range = float((daily_df["high"] - daily_df["low"]).tail(14).mean())
    sl_dist = max(daily_range * MR_SL_ATR_MULT, cfg["min_sl"])
    sma5 = float(daily_df["close"].tail(5).mean())
    fill = price + cfg["spread"]
    sl = price - sl_dist
    tp = sma5
    if tp <= fill: return None
    risk = fill - sl
    if risk <= 0 or (tp - fill) / risk < 0.8: return None
    return {"side": "long", "entry": fill, "sl": sl, "tp": tp}


# ============================================================
# Backtest loop
# ============================================================
def run_window(all_1h: dict, all_daily: dict, tradeable) -> dict:
    equity = config.ACCOUNT_SIZE
    pos: dict = {}            # (sym, strat) -> position
    trades: list = []
    eq_curve = [equity]
    daily_eq = {}
    daily_orb_taken: dict = {}
    daily_mr_taken: dict = {}
    circuit_breaker_until = None

    for bar, ts in enumerate(tradeable):
        day = ts.date()
        if day not in daily_eq: daily_eq[day] = equity

        # Manage open positions
        for key in list(pos.keys()):
            sym, strat = key
            cfg = INSTRUMENTS[sym]
            data = all_1h[sym]
            if ts not in data.index: continue
            p = pos[key]
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

            tlimit = ORB_TIME_STOP if strat == "orb" else MR_TIME_STOP
            if not close_it and bars_held >= tlimit:
                close_it, reason, ep = True, "Time", price

            # ORB session end
            if strat == "orb" and not close_it and ts.hour >= cfg["orb_hour"] + 7:
                close_it, reason, ep = True, "SessionEnd", price

            if close_it:
                raw = ((ep - p["entry"]) if p["side"] == "long" else
                       (p["entry"] - ep)) * p["lots"] * cfg["lot_mult"]
                c = cfg["comm"] * p["lots"]
                equity += raw - c
                trades.append({"sym": sym, "strat": strat, "pnl": raw - c, "reason": reason, "ts": ts})
                del pos[key]

        # Daily DD brake
        dd = (daily_eq[day] - equity) / daily_eq[day] if equity < daily_eq[day] else 0
        if dd >= 0.04:
            eq_curve.append(equity); continue
        if equity <= config.EQUITY_FLOOR:
            eq_curve.append(equity); continue
        if circuit_breaker_until and day < circuit_breaker_until:
            eq_curve.append(equity); continue
        dvals = list(daily_eq.values())
        if len(dvals) >= 10:
            recent = max(dvals[-10:])
            if recent > 0 and (recent - equity) / recent >= 0.05:
                circuit_breaker_until = (pd.Timestamp(day) + pd.Timedelta(days=5)).date()
                eq_curve.append(equity); continue

        # --- ORB entries ---
        if len(pos) < MAX_CONCURRENT:
            for sym, cfg in INSTRUMENTS.items():
                if not cfg["use_orb"]: continue
                key = (sym, "orb")
                if key in pos: continue
                if len(pos) >= MAX_CONCURRENT: break
                data = all_1h[sym]
                if ts not in data.index: continue
                dkey = (day, sym)
                if daily_orb_taken.get(dkey, 0) >= 1: continue
                loc = data.index.get_loc(ts)
                if loc < 500: continue
                w = data.iloc[max(0, loc - 500):loc + 1]
                price = float(data.loc[ts, "close"])
                bh = float(data.loc[ts, "high"])
                bl = float(data.loc[ts, "low"])
                sig = gen_orb(w, price, bh, bl, ts, cfg)
                if sig is None: continue
                risk = abs(sig["entry"] - sig["sl"])
                if risk < cfg["min_sl"] * 0.5: continue
                rd = equity * ORB_RISK_PCT
                lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                pos[key] = {"side": sig["side"], "entry": sig["entry"], "sl": sig["sl"],
                            "tp": sig["tp"], "lots": lots, "bar": bar}
                daily_orb_taken[dkey] = 1

        # --- MR entries — once per day at mr_check_hour ---
        if len(pos) < MAX_CONCURRENT:
            for sym, cfg in INSTRUMENTS.items():
                if not cfg["use_mr"]: continue
                if ts.hour != cfg["mr_check_hour"]: continue
                key = (sym, "mr")
                if key in pos: continue
                if len(pos) >= MAX_CONCURRENT: break
                dkey = (day, sym)
                if daily_mr_taken.get(dkey, 0) >= 1: continue
                # Build today's daily bar context from 1H
                day_data = all_1h[sym].loc[pd.Timestamp(day).tz_localize("UTC"):ts]
                if len(day_data) < 3: continue
                bh_today = float(day_data["high"].max())
                bl_today = float(day_data["low"].min())
                price_today = float(day_data["close"].iloc[-1])
                # Daily history (exclude today)
                daily_hist = all_daily[sym].loc[:pd.Timestamp(day).tz_localize("UTC") - pd.Timedelta(hours=1)]
                if len(daily_hist) < 20: continue
                sig = gen_mr(daily_hist, price_today, bh_today, bl_today, cfg)
                if sig is None: continue
                risk = abs(sig["entry"] - sig["sl"])
                if risk < cfg["min_sl"] * 0.3: continue
                rd = equity * MR_RISK_PCT
                lots = max(0.01, min(0.50, round(rd / (risk * cfg["lot_mult"]), 2)))
                pos[key] = {"side": "long", "entry": sig["entry"], "sl": sig["sl"],
                            "tp": sig["tp"], "lots": lots, "bar": bar}
                daily_mr_taken[dkey] = 1

        # MTM
        eq = equity
        for key, p in pos.items():
            sym = key[0]; cfg = INSTRUMENTS[sym]
            data = all_1h[sym]
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
    per_cell = {}
    if trades:
        df = pd.DataFrame(trades)
        df["ym"] = df["ts"].dt.to_period("M")
        monthly = df.groupby("ym")["pnl"].sum()
        pct_pos = (monthly > 0).sum() / len(monthly) * 100 if len(monthly) else 0
        avg_monthly = mean([m / config.ACCOUNT_SIZE * 100 for m in monthly.values])
        for (sym, strat), g in df.groupby(["sym", "strat"]):
            per_cell[(sym, strat)] = {"trades": len(g), "pnl": g["pnl"].sum()}
    else:
        pct_pos = 0; avg_monthly = 0

    days = (tradeable[-1] - tradeable[0]).days
    return {"days": days, "trades": len(trades), "wr": wr, "pnl": pnl,
            "pnl_pct": pnl / config.ACCOUNT_SIZE * 100, "pf": pf, "mdd": mdd,
            "pct_pos_months": pct_pos, "avg_monthly_pct": avg_monthly, "per_cell": per_cell}


def main():
    all_1h = {}; all_daily = {}
    for sym, cfg in INSTRUMENTS.items():
        print(f"  Fetching {sym} ({cfg['ticker']})...", end=" ", flush=True)
        d = fetch(cfg["ticker"])
        all_1h[sym] = d
        all_daily[sym] = to_daily(d)
        print(f"{len(d)} bars 1H / {len(all_daily[sym])} days")
        time.sleep(1)

    idx = all_1h["US30"].index[500:]
    n = len(idx)
    splits = [
        ("TRAIN",    idx[:int(n * 0.40)]),
        ("TEST",     idx[int(n * 0.40):int(n * 0.70)]),
        ("VALIDATE", idx[int(n * 0.70):]),
    ]

    print(f"\n{'=' * 84}")
    print(f"  UNIFIED PORTFOLIO — ORB + MR, shared equity")
    print(f"  Risk: ORB {ORB_RISK_PCT*100:.1f}% | MR {MR_RISK_PCT*100:.1f}% | Max concurrent: {MAX_CONCURRENT}")
    print(f"  Daily DD: 4% | Rolling 10d DD breaker: 5% → pause 5d")
    print(f"{'=' * 84}")

    for label, window in splits:
        print(f"\n  {label} ({window[0].date()} → {window[-1].date()})...")
        r = run_window(all_1h, all_daily, window)
        pf_s = f"{r['pf']:.2f}" if r['pf'] != float('inf') else "INF"
        print(f"    Trades: {r['trades']} | WR: {r['wr']:.1f}% | PnL: ${r['pnl']:+,.0f} "
              f"({r['pnl_pct']:+.2f}%) | PF: {pf_s} | DD: {r['mdd']:.2f}%")
        print(f"    Monthly: {r['avg_monthly_pct']:+.2f}% | +Months: {r['pct_pos_months']:.0f}%")
        for (sym, strat), stats in sorted(r["per_cell"].items()):
            print(f"      {sym:6s} {strat:3s}  {stats['trades']:3d}T  ${stats['pnl']:>+7,.2f}")

    print(f"\n{'=' * 84}\n")


if __name__ == "__main__":
    main()
