"""
Opening Range Breakout (ORB) — structural edge at NY open.

Logic:
1. Mark the high/low of the first N bars after NY open (13:30 UTC = market open)
2. Wait for breakout above high or below low
3. SL = opposite side of opening range
4. TP = opening range size × multiplier
5. One trade per day, no re-entries
6. Close at end of NY session if not resolved

Structural reason for edge:
- Overnight accumulation → NY liquidity influx causes directional commitment
- Retail + institutional orders cluster at market open
- Breakout direction has positive correlation with day's trend
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from strategy.hybrid import Signal, Quality


@dataclass
class ORBRange:
    high: float
    low: float
    size: float
    valid: bool


def get_orb_range(df_1h: pd.DataFrame, now_ts: pd.Timestamp, orb_bars: int = 1) -> ORBRange:
    """
    Build opening range from the first N 1H bars after NY open (13:00 UTC).
    With 1H data, ORB=1 means just the 13:00-14:00 bar (the opening hour).
    """
    day_start = now_ts.replace(hour=13, minute=0, second=0, microsecond=0)
    day_end = day_start + pd.Timedelta(hours=orb_bars)
    session = df_1h.loc[day_start:day_end]

    if len(session) < orb_bars:
        return ORBRange(0, 0, 0, False)

    h = float(session["high"].max())
    l = float(session["low"].min())
    size = h - l
    if size <= 0:
        return ORBRange(h, l, size, False)
    return ORBRange(h, l, size, True)


def generate_orb_signal(
    df_1h: pd.DataFrame,
    current_price: float,
    bar_high: float,
    bar_low: float,
    now_ts: pd.Timestamp,
    symbol: str,
) -> Signal | None:
    """
    ORB signal — fires during the NY session after the ORB window closes.
    Entry window: 14:00 - 18:00 UTC (4 hours after ORB)
    Filtered: only in direction of daily trend (close vs 20-bar SMA of daily closes).
    """
    cfg = config.INSTRUMENTS.get(symbol, {})
    spread = cfg.get("spread", 1.0)
    min_sl = cfg.get("min_sl", 3.0)

    # Entry window: after ORB (14:00) until mid-afternoon (18:00)
    if now_ts.hour < 14 or now_ts.hour >= 18:
        return None

    rng = get_orb_range(df_1h, now_ts, orb_bars=1)
    if not rng.valid or rng.size < min_sl:
        return None

    direction = None
    if bar_high > rng.high and current_price > rng.high:
        direction = "long"
    elif bar_low < rng.low and current_price < rng.low:
        direction = "short"
    else:
        return None

    # Trend filter: daily close vs 20-day SMA (resample 1H → daily)
    if config.ORB_TREND_FILTER:
        daily = df_1h["close"].resample("1D").last().dropna()
        if len(daily) >= 20:
            sma20 = float(daily.tail(20).mean())
            last_close = float(daily.iloc[-1])
            if direction == "long" and last_close < sma20:
                return None
            if direction == "short" and last_close > sma20:
                return None

    buffer = rng.size * 0.1
    if direction == "long":
        sl = rng.low - buffer
        tp = current_price + rng.size * config.ORB_TP_MULT
        fill = current_price + spread
    else:
        sl = rng.high + buffer
        tp = current_price - rng.size * config.ORB_TP_MULT
        fill = current_price - spread

    risk = abs(fill - sl)
    if risk < min_sl * 0.5:
        return None

    rr = abs(tp - fill) / risk if risk > 0 else 0
    if rr < 1.0:
        return None

    return Signal(
        direction=direction, entry=fill, sl=sl, tp=tp,
        quality=Quality.A, confidence=1.0, rr=rr,
        reason=f"ORB {direction.upper()} | range {rng.size:.2f}",
        symbol=symbol,
        bias_info=f"ORB [{rng.low:.2f}-{rng.high:.2f}]",
        is_monster=False,
    )
