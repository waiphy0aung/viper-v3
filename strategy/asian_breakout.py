"""
Asian Session Breakout (Gold) — structural edge, not pattern recognition.

Logic:
1. Mark the high/low of the Asian session (00:00-07:00 UTC)
2. Wait for London open (07:00-09:00 UTC) to break the Asian range
3. SL = just outside the opposite side of Asian range
4. TP = 1.5x Asian range size
5. One trade per day max
6. No signal at all if Asian range is abnormally wide (news day)

Why this exists:
- Asian session is low-volatility accumulation
- London/NY liquidity provides the directional push
- Exploits session transition, not indicator alignment
- Hard to arb because it's session-structural, not price-pattern
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from strategy.hybrid import Signal, Quality


@dataclass
class AsianRange:
    high: float
    low: float
    size: float
    valid: bool  # False if session data incomplete or too wide


def get_asian_range(df_1h: pd.DataFrame, now_ts: pd.Timestamp) -> AsianRange:
    """
    Build today's Asian session range from 00:00 to 07:00 UTC.
    Returns valid=False if data insufficient or range abnormal.
    """
    asian_start = now_ts.replace(hour=0, minute=0, second=0, microsecond=0)
    asian_end = asian_start.replace(hour=7)
    session = df_1h.loc[asian_start:asian_end]

    if len(session) < 5:  # need at least 5 hours of Asian data
        return AsianRange(0, 0, 0, False)

    h = float(session["high"].max())
    l = float(session["low"].min())
    size = h - l

    # Skip abnormal ranges (news day, gap)
    avg_bar = float((session["high"] - session["low"]).mean())
    if size > avg_bar * 12:  # range more than 12x avg bar = anomaly
        return AsianRange(h, l, size, False)

    return AsianRange(h, l, size, True)


def generate_asian_breakout_signal(
    df_1h: pd.DataFrame,
    current_price: float,
    bar_high: float,
    bar_low: float,
    now_ts: pd.Timestamp,
    symbol: str,
) -> Signal | None:
    """
    Asian breakout signal.
    Only fires during London open window (07:00-09:00 UTC) on range break.
    """
    cfg = config.INSTRUMENTS.get(symbol, {})
    spread = cfg.get("spread", 1.0)
    min_sl = cfg.get("min_sl", 3.0)

    # Only fire during London open (07:00-09:00 UTC)
    if now_ts.hour < 7 or now_ts.hour >= 9:
        return None

    rng = get_asian_range(df_1h, now_ts)
    if not rng.valid:
        return None
    if rng.size < min_sl:  # range too tight, skip
        return None

    # Breakout detection: current bar's wick exceeded one side
    direction = None
    if bar_high > rng.high and current_price > rng.high:
        direction = "long"
    elif bar_low < rng.low and current_price < rng.low:
        direction = "short"
    else:
        return None

    # SL = opposite side of Asian range + small buffer
    buffer = rng.size * 0.1
    if direction == "long":
        sl = rng.low - buffer
        tp = current_price + rng.size * config.ASIAN_TP_MULT
        fill = current_price + spread
    else:
        sl = rng.high + buffer
        tp = current_price - rng.size * config.ASIAN_TP_MULT
        fill = current_price - spread

    risk = abs(fill - sl)
    if risk < min_sl * 0.5:
        return None

    rr = abs(tp - fill) / risk if risk > 0 else 0
    if rr < 1.0:  # skip weak R:R setups
        return None

    return Signal(
        direction=direction, entry=fill, sl=sl, tp=tp,
        quality=Quality.A, confidence=1.0, rr=rr,
        reason=f"Asian {direction.upper()} brk | range {rng.size:.2f}",
        symbol=symbol,
        bias_info=f"Asian [{rng.low:.2f}-{rng.high:.2f}]",
        is_monster=False,
    )
