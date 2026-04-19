"""
Technical indicators — pure functions, no side effects.
Used by both indicator strategy and SMC zone detection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, p: int) -> pd.Series:
    return s.ewm(span=p, adjust=False).mean()


def sma(s: pd.Series, p: int) -> pd.Series:
    return s.rolling(window=p).mean()


def wma(s: pd.Series, p: int) -> pd.Series:
    w = np.arange(1, p + 1, dtype=float)
    return s.rolling(window=p).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, p: int) -> pd.Series:
    hp = max(1, p // 2)
    sp = max(1, int(np.sqrt(p)))
    return wma(2 * wma(s, hp) - wma(s, p), sp)


def rsi(s: pd.Series, p: int = 14) -> pd.Series:
    d = s.diff()
    g = d.where(d > 0, 0.0)
    l = (-d).where(d < 0, 0.0)
    ag = g.ewm(com=p - 1, min_periods=p).mean()
    al = l.ewm(com=p - 1, min_periods=p).mean()
    return 100 - (100 / (1 + ag / al))


def atr(h: pd.Series, l: pd.Series, c: pd.Series, p: int = 14) -> pd.Series:
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(span=p, adjust=False).mean()


def bollinger_bands(c: pd.Series, p: int = 20, m: float = 2.0):
    mid = sma(c, p)
    std = c.rolling(window=p).std()
    return mid + m * std, mid, mid - m * std


def keltner_channels(h: pd.Series, l: pd.Series, c: pd.Series,
                     ep: int = 20, ap: int = 10, m: float = 1.5):
    mid = ema(c, ep)
    a = atr(h, l, c, ap)
    return mid + m * a, mid, mid - m * a


def ttm_squeeze(h: pd.Series, l: pd.Series, c: pd.Series,
                bp: int = 20, bs: float = 2.0,
                kp: int = 20, kap: int = 10, km: float = 1.5):
    """Returns (squeeze_on, momentum) as Series."""
    bb_u, _, bb_l = bollinger_bands(c, bp, bs)
    kc_u, _, kc_l = keltner_channels(h, l, c, kp, kap, km)
    squeeze_on = (bb_l > kc_l) & (bb_u < kc_u)
    momentum = c.rolling(window=bp).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    return squeeze_on, momentum
