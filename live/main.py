"""
VIPER v3 — Live Signal Bot with all enhancements.
"""

from __future__ import annotations

import logging
import sys
import os
import time
import traceback
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from live.data import DataFeed
from live.notifier import (
    send_signal, send_startup, send_error, send_cooldown, send_weekly_report,
)
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


def is_weekend() -> bool:
    """Friday 20:00 UTC → Sunday 22:00 UTC"""
    if not config.WEEKEND_FILTER:
        return False
    now = datetime.now(timezone.utc)
    # Friday = 4, Saturday = 5, Sunday = 6
    if now.weekday() == 4 and now.hour >= 20:
        return True
    if now.weekday() == 5:
        return True
    if now.weekday() == 6 and now.hour < 22:
        return True
    return False


def is_news_blocked() -> bool:
    """Block NFP (1st Friday 12-14 UTC) and FOMC (3rd Wednesday 18-20 UTC)."""
    if not config.NEWS_FILTER:
        return False
    now = datetime.now(timezone.utc)

    # NFP: 1st Friday of the month, 12:00-14:00 UTC
    if now.weekday() == 4 and now.day <= 7 and 12 <= now.hour < 14:
        return True

    # FOMC: ~3rd Wednesday of month, 18:00-20:00 UTC
    if now.weekday() == 2 and 15 <= now.day <= 21 and 18 <= now.hour < 20:
        return True

    return False


def run_cycle(data: DataFeed, last_signal: dict, tracker: ForwardTracker,
              cooldown_until: float):
    """Returns updated cooldown_until."""

    # Check open signals
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
    tracker.check_dd_alerts()

    # Cooldown check
    now = time.time()
    if now < cooldown_until:
        remaining = int((cooldown_until - now) / 60)
        logger.debug(f"Cooldown: {remaining}m remaining")
        return cooldown_until

    # Check consecutive losses
    if tracker.consecutive_losses >= config.COOLDOWN_AFTER_LOSSES:
        cooldown_until = now + config.COOLDOWN_HOURS * 3600
        send_cooldown(tracker.consecutive_losses, config.COOLDOWN_HOURS)
        logger.warning(f"Cooldown activated: {tracker.consecutive_losses} losses → {config.COOLDOWN_HOURS}h")
        return cooldown_until

    # Seasonal filter checked per instrument in the scan loop below

    # Weekend filter
    if is_weekend():
        logger.debug("Weekend — markets closed")
        return cooldown_until

    # News filter
    if is_news_blocked():
        logger.info("News window — skipping signals")
        return cooldown_until

    # Scan for new signals
    for symbol in config.INSTRUMENTS:
        try:
            if not in_session(symbol):
                continue

            # Seasonal: check if this instrument trades this month
            if config.SEASONAL_FILTER:
                inst_months = config.INSTRUMENTS[symbol].get("months")
                if inst_months and datetime.now(timezone.utc).month not in inst_months:
                    continue

            if any(s.symbol == symbol for s in tracker.open_signals):
                continue

            last = last_signal.get(symbol, 0)
            if now - last < 7200:
                continue

            df_1h = data.fetch_1h(symbol)
            df_4h = data.fetch_4h(symbol)
            df_daily = data.fetch_daily(symbol)
            price = float(df_1h["close"].iloc[-1])

            sig = generate_signal(df_1h, df_4h, df_daily, price, symbol)
            if sig is None:
                continue

            cfg = config.INSTRUMENTS[symbol]
            risk = abs(sig.entry - sig.sl)
            risk_d = min(config.ACCOUNT_SIZE * config.BASE_RISK_PCT * sig.confidence,
                         config.ACCOUNT_SIZE * config.MAX_RISK_CAP)
            lots = round(risk_d / (risk * cfg["lot_mult"]), 2)
            lots = max(0.01, min(0.10, lots))

            if lots < 0.01:
                continue

            send_signal(sig, lots, risk_d)
            tracker.add_signal(
                symbol=symbol, direction=sig.direction, quality=sig.quality.value,
                entry=sig.entry, sl=sig.sl, tp=sig.tp,
                lots=lots, risk_dollars=risk_d, rr=sig.rr, reason=sig.reason,
            )
            last_signal[symbol] = now

        except Exception as e:
            logger.error(f"Error scanning {symbol}: {e}")

    return cooldown_until


def main():
    setup_logging()

    logger.info("=" * 50)
    logger.info("VIPER v3 — Live Signal Bot")
    logger.info("=" * 50)

    data = DataFeed()
    tracker = ForwardTracker()
    last_signal: dict[str, float] = {}
    cooldown_until = 0.0
    last_summary_date = None
    last_weekly_date = None

    send_startup()

    errors = 0

    while True:
        try:
            now = datetime.now(timezone.utc)

            cooldown_until = run_cycle(data, last_signal, tracker, cooldown_until)

            # Daily summary at 21:00 UTC
            if now.hour == 21 and last_summary_date != now.date():
                from live.notifier import _send
                _send(f"\U0001f4ca <b>Daily Summary</b>\n\n<pre>{tracker.summary_text()}</pre>\n\nEquity: ${tracker.equity:,.2f}")
                last_summary_date = now.date()

            # Weekly report — Sunday 18:00 UTC
            if now.weekday() == 6 and now.hour == 18 and last_weekly_date != now.date():
                send_weekly_report(
                    tracker.summary_text(),
                    config.EXPECTED_WR, config.EXPECTED_PF,
                    tracker.win_rate, tracker.profit_factor,
                )
                last_weekly_date = now.date()

            errors = 0
            time.sleep(300)

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            break

        except Exception as e:
            errors += 1
            logger.error(f"Cycle error #{errors}: {e}\n{traceback.format_exc()}")
            if errors >= 5:
                send_error(f"Bot stopped: {e}")
                break
            time.sleep(60 * errors)

    logger.info("VIPER v3 stopped.")


if __name__ == "__main__":
    main()
