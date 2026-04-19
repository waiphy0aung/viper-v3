"""
Market structure — swing points, BOS, CHoCH, dealing range.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd
import numpy as np


class Bias(Enum):
    BULLISH = "BULL"
    BEARISH = "BEAR"
    RANGING = "RANGE"


class BreakType(Enum):
    BOS = "BOS"
    CHOCH = "CHOCH"


@dataclass
class Swing:
    idx: int
    price: float
    is_high: bool


@dataclass
class Break:
    kind: BreakType
    direction: str  # "bullish" or "bearish"
    price: float
    idx: int


@dataclass
class DealingRange:
    high: float
    low: float

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2

    def is_premium(self, price: float) -> bool:
        return price > self.mid

    def is_discount(self, price: float) -> bool:
        return price < self.mid


@dataclass
class Structure:
    bias: Bias
    swings_high: list[Swing] = field(default_factory=list)
    swings_low: list[Swing] = field(default_factory=list)
    breaks: list[Break] = field(default_factory=list)
    dealing_range: DealingRange | None = None


def find_swings(h: pd.Series, l: pd.Series, strength: int = 3):
    highs, lows = [], []
    for i in range(strength, len(h) - strength):
        if all(h.iloc[i] >= h.iloc[i - j] and h.iloc[i] >= h.iloc[i + j]
               for j in range(1, strength + 1)):
            highs.append(Swing(i, float(h.iloc[i]), True))
        if all(l.iloc[i] <= l.iloc[i - j] and l.iloc[i] <= l.iloc[i + j]
               for j in range(1, strength + 1)):
            lows.append(Swing(i, float(l.iloc[i]), False))
    return highs, lows


def detect_structure(h: pd.Series, l: pd.Series, c: pd.Series,
                     strength: int = 3, lookback: int = 50) -> Structure:
    h = h.iloc[-lookback:] if len(h) > lookback else h
    l = l.iloc[-lookback:] if len(l) > lookback else l
    c = c.iloc[-lookback:] if len(c) > lookback else c

    highs, lows = find_swings(h, l, strength)
    if len(highs) < 2 or len(lows) < 2:
        return Structure(Bias.RANGING, highs, lows)

    bias = Bias.RANGING
    breaks = []
    prev_h, prev_l = None, None

    for sw in sorted(highs + lows, key=lambda s: s.idx):
        if sw.is_high:
            if prev_h and sw.price > prev_h.price:
                kind = BreakType.CHOCH if bias == Bias.BEARISH else BreakType.BOS
                breaks.append(Break(kind, "bullish", prev_h.price, sw.idx))
                bias = Bias.BULLISH
            prev_h = sw
        else:
            if prev_l and sw.price < prev_l.price:
                kind = BreakType.CHOCH if bias == Bias.BULLISH else BreakType.BOS
                breaks.append(Break(kind, "bearish", prev_l.price, sw.idx))
                bias = Bias.BEARISH
            prev_l = sw

    dr = None
    if highs and lows:
        dr = DealingRange(
            high=max(s.price for s in highs[-3:]),
            low=min(s.price for s in lows[-3:]),
        )

    return Structure(bias, highs, lows, breaks, dr)
