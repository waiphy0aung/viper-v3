"""
Liquidity detection — session levels, sweeps, draw on liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from core.structure import Swing


@dataclass
class SessionLevels:
    pdh: float
    pdl: float
    pwh: float
    pwl: float


def get_session_levels(df_1h: pd.DataFrame) -> SessionLevels:
    if df_1h.index.tz is None:
        df_1h = df_1h.copy()
        df_1h.index = df_1h.index.tz_localize("UTC")

    now = df_1h.index[-1]
    today = now.normalize()
    yesterday = today - pd.Timedelta(days=1)

    prev_day = df_1h[(df_1h.index.normalize() >= yesterday - pd.Timedelta(days=3)) &
                     (df_1h.index.normalize() < today)]
    this_monday = today - pd.Timedelta(days=today.weekday())
    last_monday = this_monday - pd.Timedelta(days=7)
    prev_week = df_1h[(df_1h.index >= last_monday) & (df_1h.index < this_monday)]

    return SessionLevels(
        pdh=float(prev_day["high"].max()) if not prev_day.empty else float(df_1h["high"].iloc[-2]),
        pdl=float(prev_day["low"].min()) if not prev_day.empty else float(df_1h["low"].iloc[-2]),
        pwh=float(prev_week["high"].max()) if not prev_week.empty else float(df_1h["high"].max()),
        pwl=float(prev_week["low"].min()) if not prev_week.empty else float(df_1h["low"].min()),
    )


def detect_sweep(swings: list[Swing], h: pd.Series, l: pd.Series,
                 c: pd.Series, lookback: int = 5) -> bool:
    for sw in swings:
        for i in range(-lookback, 0):
            if i < -len(h):
                break
            if sw.is_high:
                if float(h.iloc[i]) > sw.price and float(c.iloc[i]) < sw.price:
                    return True
            else:
                if float(l.iloc[i]) < sw.price and float(c.iloc[i]) > sw.price:
                    return True
    return False
