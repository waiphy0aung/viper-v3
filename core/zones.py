"""
Order Blocks and Fair Value Gaps — institutional entry zones.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from core.structure import Break


class ZoneKind(Enum):
    BULL_OB = "BULL_OB"
    BEAR_OB = "BEAR_OB"
    BULL_FVG = "BULL_FVG"
    BEAR_FVG = "BEAR_FVG"


class ZoneStatus(Enum):
    FRESH = "FRESH"
    TESTED = "TESTED"
    BROKEN = "BROKEN"


@dataclass
class Zone:
    kind: ZoneKind
    top: float
    bottom: float
    idx: int
    status: ZoneStatus = ZoneStatus.FRESH

    @property
    def mid(self) -> float:
        return (self.top + self.bottom) / 2

    @property
    def height(self) -> float:
        return self.top - self.bottom

    def contains(self, price: float, buffer: float = 0.002) -> bool:
        return self.bottom * (1 - buffer) <= price <= self.top * (1 + buffer)

    @property
    def is_bullish(self) -> bool:
        return self.kind in (ZoneKind.BULL_OB, ZoneKind.BULL_FVG)


def find_order_blocks(h: pd.Series, l: pd.Series, o: pd.Series,
                      c: pd.Series, breaks: list[Break], lookback: int = 10) -> list[Zone]:
    obs = []
    for brk in breaks:
        if brk.idx < lookback:
            continue
        if brk.direction == "bullish":
            for j in range(brk.idx - 1, max(brk.idx - lookback, 0), -1):
                if c.iloc[j] < o.iloc[j]:
                    obs.append(Zone(ZoneKind.BULL_OB, float(h.iloc[j]), float(l.iloc[j]), j))
                    break
        elif brk.direction == "bearish":
            for j in range(brk.idx - 1, max(brk.idx - lookback, 0), -1):
                if c.iloc[j] > o.iloc[j]:
                    obs.append(Zone(ZoneKind.BEAR_OB, float(h.iloc[j]), float(l.iloc[j]), j))
                    break
    return obs


def find_fvgs(h: pd.Series, l: pd.Series, c: pd.Series,
              tolerance: float = 0.001) -> list[Zone]:
    fvgs = []
    for i in range(1, len(h) - 1):
        # Bullish FVG (or near-gap)
        gap = float(l.iloc[i + 1]) - float(h.iloc[i - 1])
        tol = float(h.iloc[i - 1]) * tolerance
        if gap > -tol:
            top = max(float(l.iloc[i + 1]), float(h.iloc[i - 1]) + tol)
            bot = float(h.iloc[i - 1])
            if top > bot:
                fvgs.append(Zone(ZoneKind.BULL_FVG, top, bot, i))

        # Bearish FVG
        gap = float(l.iloc[i - 1]) - float(h.iloc[i + 1])
        tol = float(l.iloc[i - 1]) * tolerance
        if gap > -tol:
            top = float(l.iloc[i - 1])
            bot = min(float(h.iloc[i + 1]), float(l.iloc[i - 1]) - tol)
            if top > bot:
                fvgs.append(Zone(ZoneKind.BEAR_FVG, top, bot, i))
    return fvgs


def update_zones(zones: list[Zone], h: pd.Series, l: pd.Series,
                 c: pd.Series, lookback: int = 5) -> list[Zone]:
    for z in zones:
        if z.status == ZoneStatus.BROKEN:
            continue
        for i in range(-lookback, 0):
            if i < -len(c):
                break
            hi, lo, cl = float(h.iloc[i]), float(l.iloc[i]), float(c.iloc[i])
            if z.is_bullish:
                if lo <= z.top:
                    z.status = ZoneStatus.BROKEN if cl < z.bottom else ZoneStatus.TESTED
            else:
                if hi >= z.bottom:
                    z.status = ZoneStatus.BROKEN if cl > z.top else ZoneStatus.TESTED
    return zones
