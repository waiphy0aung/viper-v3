"""
Scalper strategy — fast indicator signals, ATR-based SL, fixed 1:2 TP.

Design philosophy:
- Frequency over size. 1:2 RR, many trades, no chasing fat tails.
- No SMC requirement (quality filter too tight → dry spells).
- No partials, no BE trailing — clean hit-or-miss mechanics.
- Session-limited to NY open window (highest liquidity).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

import config
from core.indicators import hma, ttm_squeeze, atr
from core.structure import Bias, detect_structure
from strategy.hybrid import Signal, Quality, _check_indicator_signal


def generate_scalper_signal(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    current_price: float,
    symbol: str,
) -> Signal | None:
    """
    Scalper signal:
    1. Indicator trigger (HMA cross + squeeze + volume)
    2. 4H bias agreement (trend filter)
    3. ATR-based SL (no SMC zones)
    4. Fixed 1:2 TP
    """
    cfg = config.INSTRUMENTS.get(symbol, {})
    min_sl = cfg.get("min_sl", 10)
    spread = cfg.get("spread", 1.0)

    if len(df_1h) < 50 or len(df_4h) < 10:
        return None

    ind = _check_indicator_signal(
        df_1h["high"], df_1h["low"], df_1h["open"], df_1h["close"], df_1h["volume"])
    if ind is None:
        return None

    h4 = detect_structure(df_4h["high"], df_4h["low"], df_4h["close"], 3, 50)
    if ind == "long" and h4.bias != Bias.BULLISH:
        return None
    if ind == "short" and h4.bias != Bias.BEARISH:
        return None

    atr_val = float(atr(df_1h["high"], df_1h["low"], df_1h["close"], 14).iloc[-1])
    sl_dist = atr_val * config.SCALPER_SL_ATR_MULT

    if sl_dist < min_sl * 0.5:
        sl_dist = min_sl * 0.5

    tp_dist = sl_dist * config.SCALPER_RR

    if ind == "long":
        sl = current_price - sl_dist
        tp = current_price + tp_dist
        fill = current_price + spread
    else:
        sl = current_price + sl_dist
        tp = current_price - tp_dist
        fill = current_price - spread

    rr = tp_dist / sl_dist
    bias_info = f"4H={h4.bias.value}"
    reason = f"HMA {ind.upper()} | ATR SL {sl_dist:.2f} | 1:{rr:.1f}"

    return Signal(
        direction=ind, entry=fill, sl=sl, tp=tp,
        quality=Quality.B, confidence=1.0, rr=rr,
        reason=reason, symbol=symbol,
        bias_info=bias_info, is_monster=False,
    )
