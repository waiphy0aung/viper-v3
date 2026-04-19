"""
Telegram notifier — signals, closes, alerts, reports.
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
    grade = {Quality.A_PLUS: "\U0001f31f", Quality.A: "\u2b50", Quality.B: "\u26aa"}.get(sig.quality, "")

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
        f"\U0001f4ca {sig.bias_info}\n"
        f"\U0001f9e0 {sig.reason}\n"
        f"\u23f0 {now}"
    )
    _send(text)


def send_close(symbol: str, direction: str, entry: float, exit_price: float,
               pnl: float, reason: str, equity: float, progress_pct: float):
    emoji = "\U0001f4b0" if pnl > 0 else "\U0001f4a5"
    pnl_sign = "+" if pnl >= 0 else ""
    side = "BUY" if direction == "long" else "SELL"

    bar_len = 20
    filled = max(0, min(bar_len, int(progress_pct / 100 * bar_len)))
    bar = "\u2588" * filled + "\u2591" * (bar_len - filled)

    text = (
        f"{emoji} <b>CLOSED {side} {symbol}</b>\n"
        f"\n"
        f"Entry: <code>{entry:.2f}</code> \u2192 Exit: <code>{exit_price:.2f}</code>\n"
        f"PnL: <code>{pnl_sign}${pnl:.2f}</code> ({reason})\n"
        f"\n"
        f"Balance: <code>${equity:,.2f}</code>\n"
        f"Target: [{bar}] {progress_pct:.1f}%"
    )
    _send(text)


def send_trailing_sl(symbol: str, direction: str, new_sl: float):
    side = "BUY" if direction == "long" else "SELL"
    _send(f"\U0001f6e1 <b>SL → Breakeven</b>\n{side} {symbol}\nNew SL: <code>{new_sl:.2f}</code>")


def send_dd_alert(dd_pct: float, equity: float):
    remaining = equity - config.EQUITY_FLOOR
    text = (
        f"\U0001f6a8 <b>DRAWDOWN ALERT — {dd_pct:.1f}%</b>\n"
        f"\n"
        f"Equity: <code>${equity:,.2f}</code>\n"
        f"Floor: <code>${config.EQUITY_FLOOR:,.2f}</code>\n"
        f"Remaining: <code>${remaining:,.2f}</code>\n"
        f"\n"
        f"\u26a0\ufe0f Reduce size or pause trading"
    )
    _send(text)


def send_cooldown(losses: int, hours: int):
    _send(f"\u23f8 <b>COOLDOWN</b>\n{losses} consecutive losses \u2192 pausing {hours}h")


def send_weekly_report(summary: str, expected_wr: float, expected_pf: float,
                       actual_wr: float, actual_pf: float):
    wr_ok = "\u2705" if actual_wr >= expected_wr * 0.8 else "\u274c"
    pf_ok = "\u2705" if actual_pf >= expected_pf * 0.5 else "\u274c"
    pf_s = f"{actual_pf:.2f}" if actual_pf != float("inf") else "INF"

    text = (
        f"\U0001f4ca <b>Weekly Report</b>\n"
        f"\n"
        f"<pre>{summary}</pre>\n"
        f"\n"
        f"vs Backtest:\n"
        f"  WR: {actual_wr:.1f}% (exp {expected_wr:.0f}%) {wr_ok}\n"
        f"  PF: {pf_s} (exp {expected_pf:.1f}) {pf_ok}"
    )
    _send(text)


def send_startup():
    syms = ", ".join(config.INSTRUMENTS.keys())
    _send(
        f"\U0001f40d <b>VIPER v3 Started</b>\n\n"
        f"Instruments: {syms}\n"
        f"Risk: {config.BASE_RISK_PCT*100:.1f}% | R:R: 1:{config.MIN_RR}\n"
        f"Trailing SL: {'ON' if config.TRAILING_SL_ENABLED else 'OFF'}\n"
        f"Weekend filter: {'ON' if config.WEEKEND_FILTER else 'OFF'}\n"
        f"News filter: {'ON' if config.NEWS_FILTER else 'OFF'}\n\n"
        f"Scanning..."
    )


def send_error(msg: str):
    _send(f"\u26a0\ufe0f <b>ERROR</b>\n\n<code>{msg[:500]}</code>")
