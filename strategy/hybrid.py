"""
Hybrid Strategy — indicator signals filtered by SMC zones.

Indicators give FREQUENCY (squeeze, HMA cross → 40+ signals/month).
SMC zones give QUALITY (only trade at institutional levels).
Combined: frequent signals at the right places.

Signal quality tiers:
  A+ : Indicator signal AT an SMC zone WITH rejection → full size
  A  : Indicator signal AT an SMC zone → 80% size
  B  : Indicator signal without SMC zone → 40% size (if allowed)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

import config
from core.indicators import hma, ttm_squeeze, atr
from core.structure import Bias, detect_structure
from core.zones import Zone, ZoneStatus, find_order_blocks, find_fvgs, update_zones
from core.liquidity import get_session_levels


class Quality(Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"


@dataclass
class Signal:
    direction: str       # "long" or "short"
    entry: float
    sl: float
    tp: float
    quality: Quality
    confidence: float    # 0-1 → lot sizing
    rr: float
    reason: str
    symbol: str
    bias_info: str = ""
    is_monster: bool = False


def _check_indicator_signal(h: pd.Series, l: pd.Series, o: pd.Series,
                            c: pd.Series, v: pd.Series) -> str | None:
    """
    Check for indicator signal: squeeze fire + HMA alignment.
    Returns "long", "short", or None.
    """
    if len(c) < 25:
        return None

    sq_on, mom = ttm_squeeze(h, l, c,
                             config.BB_PERIOD, config.BB_STD,
                             config.KC_PERIOD, config.KC_ATR_PERIOD, config.KC_MULTIPLIER)

    hma_f = hma(c, config.HMA_FAST)
    hma_s = hma(c, config.HMA_SLOW)

    vol_avg = v.rolling(20).mean()
    vol_ratio = v / vol_avg

    # Check volume
    if float(vol_ratio.iloc[-1]) < config.VOLUME_MULTIPLIER:
        return None

    hf = float(hma_f.iloc[-1])
    hs = float(hma_s.iloc[-1])
    hf_p = float(hma_f.iloc[-2])
    hs_p = float(hma_s.iloc[-2])

    m = float(mom.iloc[-1])

    # Primary: HMA crossover (gives frequency)
    # Momentum must agree (rising for long, falling for short)
    bullish_cross = hf_p <= hs_p and hf > hs
    bearish_cross = hf_p >= hs_p and hf < hs

    if bullish_cross and m > 0:
        return "long"
    if bearish_cross and m < 0:
        return "short"

    return None


def _check_rejection(h: pd.Series, l: pd.Series, o: pd.Series,
                     c: pd.Series, direction: str) -> bool:
    """Check for rejection candle on completed bar (-2)."""
    for i in range(-2, -6, -1):
        if i < -len(c):
            break
        bo, bc = float(o.iloc[i]), float(c.iloc[i])
        bh, bl = float(h.iloc[i]), float(l.iloc[i])
        tot = bh - bl
        if tot == 0:
            continue
        body = abs(bc - bo)

        if direction == "long":
            lw = min(bo, bc) - bl
            if lw / tot > config.REJECTION_WICK_RATIO and body / tot > config.REJECTION_BODY_RATIO and bc > bo:
                if float(c.iloc[-1]) > float(o.iloc[-1]):
                    return True
                break
        else:
            uw = bh - max(bo, bc)
            if uw / tot > config.REJECTION_WICK_RATIO and body / tot > config.REJECTION_BODY_RATIO and bc < bo:
                if float(c.iloc[-1]) < float(o.iloc[-1]):
                    return True
                break
    return False


def _find_zone_at_price(zones: list[Zone], price: float, direction: str) -> Zone | None:
    """Find a valid SMC zone at the current price."""
    for z in zones:
        if z.status == ZoneStatus.BROKEN:
            continue
        if not z.contains(price):
            continue
        if direction == "long" and not z.is_bullish:
            continue
        if direction == "short" and z.is_bullish:
            continue
        return z
    return None


def generate_signal(
    df_1h: pd.DataFrame,
    df_4h: pd.DataFrame,
    df_daily: pd.DataFrame,
    current_price: float,
    symbol: str,
) -> Signal | None:
    """
    Generate a hybrid signal.

    1. Check indicator signal on 1H
    2. Get HTF bias (4H + Daily)
    3. Check premium/discount
    4. Find SMC zone at price
    5. Check rejection
    6. Grade quality → determine confidence
    """
    cfg = config.INSTRUMENTS.get(symbol, {})
    min_sl = cfg.get("min_sl", 10)
    spread = cfg.get("spread", 1.0)

    if len(df_1h) < 50 or len(df_4h) < 10 or len(df_daily) < 15:
        return None

    # Step 1: Indicator signal
    ind_signal = _check_indicator_signal(
        df_1h["high"], df_1h["low"], df_1h["open"], df_1h["close"], df_1h["volume"])

    if ind_signal is None:
        return None

    # Step 2: HTF bias — 4H mandatory, daily boosts confidence
    h4 = detect_structure(df_4h["high"], df_4h["low"], df_4h["close"], 3, 50)
    d = detect_structure(df_daily["high"], df_daily["low"], df_daily["close"], 3, 30)

    # 4H must agree with signal direction
    if ind_signal == "long" and h4.bias != Bias.BULLISH:
        return None
    if ind_signal == "short" and h4.bias != Bias.BEARISH:
        return None

    # Daily agreement is a confidence booster, not a blocker
    daily_agrees = (ind_signal == "long" and d.bias == Bias.BULLISH) or \
                   (ind_signal == "short" and d.bias == Bias.BEARISH)
    daily_opposes = (ind_signal == "long" and d.bias == Bias.BEARISH) or \
                    (ind_signal == "short" and d.bias == Bias.BULLISH)

    # Block if daily OPPOSES — that's fighting the trend
    if daily_opposes:
        return None

    # Step 3: Premium/Discount — reduces confidence, doesn't block
    pd_penalty = 1.0
    if h4.dealing_range:
        if ind_signal == "long" and h4.dealing_range.is_premium(current_price):
            pd_penalty = 0.6  # in premium zone for long — lower confidence
        if ind_signal == "short" and h4.dealing_range.is_discount(current_price):
            pd_penalty = 0.6

    # Step 4: Find SMC zone at current price
    h4_obs = find_order_blocks(df_4h["high"], df_4h["low"], df_4h["open"], df_4h["close"], h4.breaks, 15)
    h4_fvgs = find_fvgs(df_4h["high"], df_4h["low"], df_4h["close"])
    all_zones = update_zones(h4_obs + h4_fvgs, df_4h["high"], df_4h["low"], df_4h["close"])

    zone = _find_zone_at_price(all_zones, current_price, ind_signal)

    # Step 5: Rejection
    has_rejection = _check_rejection(
        df_1h["high"], df_1h["low"], df_1h["open"], df_1h["close"], ind_signal)

    # Step 6: Quality grading
    if zone and has_rejection:
        quality = Quality.A_PLUS
        confidence = config.SMC_ZONE_REJECTION_CONFIDENCE
    elif zone:
        quality = Quality.A
        confidence = config.SMC_ZONE_CONFIDENCE
    elif config.ALLOW_INDICATOR_ONLY:
        quality = Quality.B
        confidence = config.INDICATOR_ONLY_CONFIDENCE
    else:
        return None

    # Dual mode: monster rejects B unless normal mode is also on
    if config.MONSTER_MODE and not config.NORMAL_MODE:
        if quality.value not in config.MONSTER_GRADES:
            return None

    # Modifiers
    if daily_agrees:
        confidence = min(1.0, confidence * 1.2)
    confidence *= pd_penalty

    # SL/TP
    atr_val = float(atr(df_1h["high"], df_1h["low"], df_1h["close"], 14).iloc[-1])

    if zone:
        # SL behind zone
        recent_lows = [float(df_1h["low"].iloc[k]) for k in range(-5, 0)]
        recent_highs = [float(df_1h["high"].iloc[k]) for k in range(-5, 0)]
        if ind_signal == "long":
            wick = min(recent_lows)
            sl = wick - zone.height * config.SL_BUFFER_MULT
            sl = min(sl, wick - min_sl * 0.5)
        else:
            wick = max(recent_highs)
            sl = wick + zone.height * config.SL_BUFFER_MULT
            sl = max(sl, wick + min_sl * 0.5)
    else:
        # No zone — use ATR-based SL
        if ind_signal == "long":
            sl = current_price - atr_val * 2.5
        else:
            sl = current_price + atr_val * 2.5

    risk = abs(current_price - sl)
    if risk < min_sl * 0.3:
        return None

    # TP targeting
    sessions = get_session_levels(df_1h)

    # Determine if THIS trade qualifies as a monster
    is_monster = False
    monster_tp = None

    if config.MONSTER_MODE and quality.value in config.MONSTER_GRADES:
        # Try to find a weekly target for monster R:R
        if ind_signal == "long":
            candidates = [t for t in [sessions.pwh, sessions.pdh]
                         if t > current_price and abs(t - current_price) >= risk * config.MONSTER_MIN_RR]
            if candidates:
                monster_tp = max(candidates)
        else:
            candidates = [t for t in [sessions.pwl, sessions.pdl]
                         if t < current_price and abs(current_price - t) >= risk * config.MONSTER_MIN_RR]
            if candidates:
                monster_tp = min(candidates)

        if monster_tp:
            is_monster = True
        elif not config.NORMAL_MODE:
            # No weekly target at 1:4+, but no normal mode fallback
            # Use minimum monster R:R as TP
            if ind_signal == "long":
                monster_tp = current_price + risk * config.MONSTER_MIN_RR
            else:
                monster_tp = current_price - risk * config.MONSTER_MIN_RR
            is_monster = True

    if is_monster:
        tp = monster_tp
        rr = abs(tp - current_price) / risk if risk > 0 else 0
        if rr < config.MONSTER_MIN_RR:
            return None
        # Monster confidence: bigger size
        confidence = config.MONSTER_RISK.get(quality.value, 0.02) / config.BASE_RISK_PCT
        confidence = min(1.5, confidence)
    elif config.NORMAL_MODE:
        # Normal trade: PDH/PDL or 3x risk
        if ind_signal == "long":
            tp = sessions.pdh if sessions.pdh > current_price and abs(sessions.pdh - current_price) >= risk * 1.5 else current_price + risk * 3
        else:
            tp = sessions.pdl if sessions.pdl < current_price and abs(current_price - sessions.pdl) >= risk * 1.5 else current_price - risk * 3
        rr = abs(tp - current_price) / risk if risk > 0 else 0
        if rr < config.MIN_RR:
            return None
    else:
        return None

    fill = current_price + spread if ind_signal == "long" else current_price - spread

    # Reason
    parts = [f"{'Squeeze' if 'sq' in str(ind_signal) else 'HMA'} {ind_signal.upper()}"]
    if zone:
        parts.append(f"{zone.kind.value} @ {zone.bottom:.2f}-{zone.top:.2f}")
    if has_rejection:
        parts.append("rejection")
    parts.append(f"[{quality.value}]")

    bias_info = f"D={d.bias.value} | 4H={h4.bias.value}"

    if is_monster:
        parts.append("MONSTER")

    return Signal(
        direction=ind_signal, entry=fill, sl=sl, tp=tp,
        quality=quality, confidence=confidence, rr=rr,
        reason=" | ".join(parts), symbol=symbol,
        bias_info=bias_info, is_monster=is_monster,
    )
