"""
Real-time data fetcher — yfinance for SP500.
"""

from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

import config

logger = logging.getLogger(__name__)


class DataFeed:
    def __init__(self):
        logger.info("Data feed initialized")

    def fetch_1h(self, symbol: str, limit: int = 200) -> pd.DataFrame:
        ticker = config.INSTRUMENTS[symbol]["ticker"]
        d = yf.download(ticker, period="60d", interval="1h", progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        return d.tail(limit)

    def fetch_4h(self, symbol: str, limit: int = 100) -> pd.DataFrame:
        d1h = self.fetch_1h(symbol, 500)
        d4h = d1h.resample("4h").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum",
        }).dropna()
        return d4h.tail(limit)

    def fetch_daily(self, symbol: str, limit: int = 60) -> pd.DataFrame:
        ticker = config.INSTRUMENTS[symbol]["ticker"]
        d = yf.download(ticker, period="2y", interval="1d", progress=False)
        d.columns = [c[0].lower() for c in d.columns]
        if d.index.tz is None:
            d.index = d.index.tz_localize("UTC")
        return d.tail(limit)

    def get_price(self, symbol: str) -> float:
        d = self.fetch_1h(symbol, 5)
        return float(d["close"].iloc[-1])
