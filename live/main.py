"""
VIPER v3 — Live Signal Bot.
Scans every 5 minutes, sends Telegram signals.
"""

from __future__ import annotations

import logging
import sys
import os
import time
import traceback
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from live.data import DataFeed
from live.notifier import send_signal, send_startup, send_error
from live.tracker import ForwardTracker
from strategy.hybrid import generate_signal


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-8s | %(name)-12s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL),
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(config.LOG_FILE),
        ],
    )


logger = logging.getLogger("main")


def in_session(symbol: str) -> bool:
    hour = datetime.now(timezone.utc).hour
    windows = config.INSTRUMENTS[symbol].get("session", [])
    return any(s <= hour < e for s, e in windows) if windows else True


def run_cycle(data: DataFeed, last_signal: dict, tracker: ForwardTracker):
    # Step 1: Check open signals against current prices
    prices = {}
    for symbol in config.INSTRUMENTS:
        try:
            df = data.fetch_1h(symbol, 5)
            if len(df) > 0:
                prices[symbol] = (
                    float(df["close"].iloc[-1]),
                    float(df["high"].iloc[-1]),
                    float(df["low"].iloc[-1]),
                )
        except Exception:
            pass

    tracker.check_signals(prices)

    # Step 2: Scan for new signals
    for symbol in config.INSTRUMENTS:
        try:
            if not in_session(symbol):
                continue

            # Don't open new if we already have an open signal on this symbol
            if any(s.symbol == symbol for s in tracker.open_signals):
                continue

            df_1h = data.fetch_1h(symbol)
            df_4h = data.fetch_4h(symbol)
            df_daily = data.fetch_daily(symbol)
            price = float(df_1h["close"].iloc[-1])

            sig = generate_signal(df_1h, df_4h, df_daily, price, symbol)

            if sig is None:
                continue

            # Cooldown: 2 hours between signals per symbol
            now = time.time()
            last = last_signal.get(symbol, 0)
            if now - last < 7200:
                continue

            # Lot sizing
            cfg = config.INSTRUMENTS[symbol]
            risk = abs(sig.entry - sig.sl)
            risk_d = min(config.ACCOUNT_SIZE * config.BASE_RISK_PCT * sig.confidence,
                         config.ACCOUNT_SIZE * config.MAX_RISK_CAP)
            lots = round(risk_d / (risk * cfg["lot_mult"]), 2)
            lots = max(0.01, min(0.10, lots))

            if lots < 0.01:
                continue

            # Send signal + track it
            send_signal(sig, lots, risk_d)
            tracker.add_signal(
                symbol=symbol, direction=sig.direction, quality=sig.quality.value,
                entry=sig.entry, sl=sig.sl, tp=sig.tp,
                lots=lots, risk_dollars=risk_d, rr=sig.rr, reason=sig.reason,
            )

            last_signal[symbol] = now
            logger.info(f"SIGNAL: {sig.direction.upper()} {symbol} [{sig.quality.value}] "
                        f"@ {sig.entry:.2f} SL={sig.sl:.2f} TP={sig.tp:.2f}")

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")


def main():
    setup_logging()

    logger.info("=" * 50)
    logger.info("VIPER v3 — Hybrid SMC+Indicator Signal Bot")
    logger.info("=" * 50)
    logger.info(f"Instruments: {list(config.INSTRUMENTS.keys())}")
    logger.info(f"Risk: {config.BASE_RISK_PCT*100:.1f}% | Min R:R: 1:{config.MIN_RR}")
    logger.info("=" * 50)

    data = DataFeed()
    tracker = ForwardTracker()
    last_signal: dict[str, float] = {}
    last_summary_date = None

    send_startup()

    errors = 0

    while True:
        try:
            now = datetime.now(timezone.utc)

            run_cycle(data, last_signal, tracker)

            # Daily summary at 21:00 UTC
            if now.hour == 21 and last_summary_date != now.date():
                from live.notifier import _send
                summary = tracker.summary_text()
                _send(f"\U0001f4ca <b>Daily Forward Test</b>\n\n<pre>{summary}</pre>")
                last_summary_date = now.date()
                logger.info(f"Daily summary:\n{summary}")

            errors = 0
            time.sleep(300)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break

        except Exception as e:
            errors += 1
            logger.error(f"Cycle error #{errors}: {e}\n{traceback.format_exc()}")

            if errors >= 5:
                send_error(f"Bot stopped after 5 errors: {e}")
                break

            time.sleep(60 * errors)

    logger.info("VIPER v3 stopped.")


if __name__ == "__main__":
    main()
