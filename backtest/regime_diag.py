"""
Regime diagnosis — what changed between the profitable windows and the dead one.
Measures volatility, trendiness, and whipsaw frequency per walk-forward window.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config


def fetch_1h(ticker: str) -> pd.DataFrame:
    d = yf.download(ticker, period="730d", interval="1h", progress=False)
    d.columns = [c[0].lower() for c in d.columns]
    if d.index.tz is None:
        d.index = d.index.tz_localize("UTC")
    return d


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return (tr.rolling(period).mean() / c) * 100


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    up = h.diff(); dn = -l.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.rolling(period).mean()


def hma(s: pd.Series, p: int) -> pd.Series:
    def wma(x, n):
        w = np.arange(1, n + 1)
        return x.rolling(n).apply(lambda v: np.dot(v, w) / w.sum(), raw=True)
    half = int(p / 2); sqp = int(np.sqrt(p))
    return wma(2 * wma(s, half) - wma(s, p), sqp)


def hma_crosses(df: pd.DataFrame) -> int:
    c = df["close"]
    hf = hma(c, config.HMA_FAST); hs = hma(c, config.HMA_SLOW)
    diff = hf - hs
    sign_change = (np.sign(diff).diff() != 0).sum()
    return int(sign_change)


def trend_stats(df: pd.DataFrame) -> dict:
    net_drift = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100
    peak = df["close"].expanding().max()
    trough = df["close"].expanding().min()
    max_up = (peak.iloc[-1] / df["close"].iloc[0] - 1) * 100
    max_dn = (trough.iloc[-1] / df["close"].iloc[0] - 1) * 100
    return {"drift": net_drift, "max_up": max_up, "max_dn": max_dn}


def diagnose(df: pd.DataFrame, label: str, start, end) -> dict:
    w = df.loc[start:end]
    if len(w) < 100:
        return {}
    a = atr_pct(w).dropna().mean()
    ax = adx(w).dropna().mean()
    xs = hma_crosses(w)
    ts = trend_stats(w)
    # Range efficiency — low values = chop
    returns = w["close"].pct_change().abs().sum()
    net = abs(w["close"].iloc[-1] - w["close"].iloc[0]) / w["close"].iloc[0]
    efficiency = (net / returns * 100) if returns > 0 else 0
    return {
        "label": label, "bars": len(w),
        "atr_pct": a, "adx": ax, "hma_crosses": xs,
        "net_drift": ts["drift"], "max_drawup": ts["max_up"], "max_drawdown": ts["max_dn"],
        "efficiency": efficiency,
    }


def main():
    print("\n  Fetching SP500 and US30...")
    data = {}
    for sym in ["SP500", "US30"]:
        data[sym] = fetch_1h(config.INSTRUMENTS[sym]["ticker"])
        time.sleep(1)

    # Use SP500's index (both cover similar range)
    idx = data["SP500"].index[200:]
    n = len(idx)
    splits = [
        ("TRAIN",    idx[0],             idx[int(n * 0.40) - 1]),
        ("TEST",     idx[int(n * 0.40)], idx[int(n * 0.70) - 1]),
        ("VALIDATE", idx[int(n * 0.70)], idx[-1]),
    ]

    for sym in ["SP500", "US30"]:
        print(f"\n{'=' * 78}")
        print(f"  {sym} — Regime Diagnosis")
        print(f"{'=' * 78}")
        hdr = f"  {'Window':10s} {'ATR%':>7s} {'ADX':>6s} {'X-overs':>8s} {'Drift%':>8s} {'Efficiency':>11s}"
        print(hdr); print(f"  {'-' * 76}")
        for label, s, e in splits:
            r = diagnose(data[sym], label, s, e)
            if not r: continue
            print(f"  {r['label']:10s} {r['atr_pct']:>6.3f}% {r['adx']:>5.1f} {r['hma_crosses']:>7d} "
                  f"{r['net_drift']:>+7.2f}% {r['efficiency']:>10.2f}%")

    print(f"\n{'=' * 78}")
    print(f"  HOW TO READ THIS:")
    print(f"    ATR%      — volatility (higher = bigger moves = more opportunity)")
    print(f"    ADX       — trend strength (>25 = trending, <20 = ranging/chop)")
    print(f"    X-overs   — HMA 9/21 crossovers (high = whipsaws = bad for strategy)")
    print(f"    Drift%    — net directional move over window")
    print(f"    Efficiency — net_move / total_motion (high = clean trend, low = chop)")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    main()
