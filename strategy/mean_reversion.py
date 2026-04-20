"""
Mean reversion — documented edge on S&P since 1993.

Logic (classic Connors-style):
1. Close < 5-day low (new 5-day low)
2. IBS (Internal Bar Strength) < 0.25 — closed in bottom quartile of day
3. Enter long at close, exit on next close ABOVE 5-day SMA or fixed time stop

IBS = (close - low) / (high - low)

Why this exists:
- Retail panic selling creates oversold closes
- Institutional rebalancing into close buys the dip
- Mean reversion on oversold daily closes is structural, not pattern-based
- Short-only on indices doesn't work (secular uptrend bias)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from strategy.hybrid import Signal, Quality


def ibs(bar_high: float, bar_low: float, bar_close: float) -> float:
    """Internal Bar Strength — 0 at low, 1 at high."""
    if bar_high - bar_low <= 0:
        return 0.5
    return (bar_close - bar_low) / (bar_high - bar_low)


def generate_mr_signal(
    df_daily: pd.DataFrame,
    current_price: float,
    bar_high: float,
    bar_low: float,
    symbol: str,
) -> Signal | None:
    """
    Daily mean reversion signal. Checks at end of day bar.
    """
    cfg = config.INSTRUMENTS.get(symbol, {})
    spread = cfg.get("spread", 1.0)
    min_sl = cfg.get("min_sl", 3.0)

    if len(df_daily) < 10:
        return None

    # Rule 1: close < 5-day low (exclusive of current bar)
    prior_5_low = float(df_daily["low"].iloc[-6:-1].min())
    if current_price >= prior_5_low:
        return None

    # Rule 2: IBS < 0.25 (closed weak)
    if ibs(bar_high, bar_low, current_price) >= 0.25:
        return None

    # Long-only (mean reversion on indices biased up)
    # SL: N × daily ATR (simple; tight enough for quick resolve)
    daily_range = float((df_daily["high"] - df_daily["low"]).tail(14).mean())
    sl_dist = daily_range * config.MR_SL_ATR_MULT
    if sl_dist < min_sl:
        sl_dist = min_sl

    # TP: return to 5-day SMA
    sma5 = float(df_daily["close"].tail(5).mean())
    tp = sma5

    fill = current_price + spread
    sl = current_price - sl_dist

    if tp <= fill:  # already at or above SMA, no edge
        return None

    risk = fill - sl
    rr = (tp - fill) / risk if risk > 0 else 0
    if rr < 0.8:
        return None

    return Signal(
        direction="long", entry=fill, sl=sl, tp=tp,
        quality=Quality.A, confidence=1.0, rr=rr,
        reason=f"MR | close<5dlow IBS={ibs(bar_high,bar_low,current_price):.2f}",
        symbol=symbol,
        bias_info=f"SMA5={sma5:.2f} 5dlow={prior_5_low:.2f}",
        is_monster=False,
    )
