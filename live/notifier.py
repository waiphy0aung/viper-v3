"""
Telegram signal sender — clean, actionable signals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import requests

import config
from strategy.hybrid import Signal, Quality

logger = logging.getLogger(__name__)


def _send(text: str):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": config.TELEGRAM_CHAT_ID, "text": text,
                  "parse_mode": "HTML", "disable_web_page_preview": True},
            timeout=10,
        )
    except Exception as e:
        logger.warning(f"Telegram failed: {e}")


def send_signal(sig: Signal, lot_size: float, risk_dollars: float):
    emoji = "\U0001f7e2" if sig.direction == "long" else "\U0001f534"
    direction = "BUY" if sig.direction == "long" else "SELL"
    now = datetime.now(timezone.utc).strftime("%H:%M UTC")

    grade_emoji = {Quality.A_PLUS: "\U0001f31f", Quality.A: "\u2b50", Quality.B: "\u26aa"}
    grade = grade_emoji.get(sig.quality, "")

    text = (
        f"{emoji} <b>{direction} {sig.symbol}</b> {grade} {sig.quality.value}\n"
        f"\n"
        f"\U0001f4cd Entry:  <code>{sig.entry:.2f}</code>\n"
        f"\U0001f6d1 SL:     <code>{sig.sl:.2f}</code>\n"
        f"\U0001f3af TP:     <code>{sig.tp:.2f}</code>\n"
        f"\n"
        f"\U0001f4e6 Lot:    <code>{lot_size:.2f}</code>\n"
        f"\U0001f4b0 Risk:   <code>${risk_dollars:.0f}</code>\n"
        f"\U0001f4ca R:R:    <code>1:{sig.rr:.1f}</code>\n"
        f"\n"
        f"\U0001f9e0 {sig.reason}\n"
        f"\u23f0 {now}"
    )
    _send(text)
    logger.info(f"Signal: {direction} {sig.symbol} [{sig.quality.value}] @ {sig.entry:.2f}")


def send_startup():
    syms = ", ".join(config.INSTRUMENTS.keys())
    text = (
        f"\U0001f40d <b>VIPER v3 Started</b>\n"
        f"\n"
        f"Strategy: Hybrid SMC + Indicator\n"
        f"Instruments: {syms}\n"
        f"Risk: {config.BASE_RISK_PCT*100:.1f}%\n"
        f"Min R:R: 1:{config.MIN_RR}\n"
        f"\n"
        f"Scanning for signals..."
    )
    _send(text)


def send_error(msg: str):
    _send(f"\u26a0\ufe0f <b>ERROR</b>\n\n<code>{msg[:500]}</code>")
