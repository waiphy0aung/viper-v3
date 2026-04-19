"""
Forward test tracker — monitors open signals and records results.

Logs every signal, checks SL/TP on each cycle, tracks live performance.
Sends daily summary. Saves everything to CSV.
"""

from __future__ import annotations

import csv
import json
import os
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

import config

logger = logging.getLogger(__name__)

SIGNALS_FILE = "forward_signals.csv"
RESULTS_FILE = "forward_results.csv"

SIGNAL_FIELDS = [
    "id", "timestamp", "symbol", "direction", "quality",
    "entry", "sl", "tp", "lots", "risk_dollars", "rr", "reason", "status",
]

RESULT_FIELDS = [
    "id", "symbol", "direction", "quality", "entry", "exit_price",
    "sl", "tp", "lots", "pnl", "rr_actual", "exit_reason",
    "entry_time", "exit_time", "bars_held",
]


@dataclass
class TrackedSignal:
    id: str
    timestamp: str
    symbol: str
    direction: str
    quality: str
    entry: float
    sl: float
    tp: float
    lots: float
    risk_dollars: float
    rr: float
    reason: str
    status: str = "OPEN"
    bars_checked: int = 0
    highest: float = 0.0
    lowest: float = float("inf")
    sl_moved_to_be: bool = False


class ForwardTracker:
    def __init__(self):
        self.open_signals: list[TrackedSignal] = []
        self.results: list[dict] = []
        self.equity: float = config.ACCOUNT_SIZE
        self.dd_alerts_sent: set[float] = set()
        self._load_open()
        self._load_results()
        # Recalculate equity from results
        self.equity = config.ACCOUNT_SIZE + sum(float(r["pnl"]) for r in self.results)
        logger.info(f"Tracker: {len(self.open_signals)} open, {len(self.results)} completed")

    def _load_open(self):
        if not os.path.exists(SIGNALS_FILE):
            return
        with open(SIGNALS_FILE, "r") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "OPEN":
                    self.open_signals.append(TrackedSignal(
                        id=row["id"], timestamp=row["timestamp"],
                        symbol=row["symbol"], direction=row["direction"],
                        quality=row["quality"],
                        entry=float(row["entry"]), sl=float(row["sl"]),
                        tp=float(row["tp"]), lots=float(row["lots"]),
                        risk_dollars=float(row["risk_dollars"]),
                        rr=float(row["rr"]), reason=row["reason"],
                    ))

    def _load_results(self):
        if not os.path.exists(RESULTS_FILE):
            return
        with open(RESULTS_FILE, "r") as f:
            self.results = list(csv.DictReader(f))

    def add_signal(self, symbol: str, direction: str, quality: str,
                   entry: float, sl: float, tp: float, lots: float,
                   risk_dollars: float, rr: float, reason: str):
        now = datetime.now(timezone.utc)
        sig = TrackedSignal(
            id=f"{symbol}-{now.strftime('%Y%m%d-%H%M%S')}",
            timestamp=now.isoformat(),
            symbol=symbol, direction=direction, quality=quality,
            entry=entry, sl=sl, tp=tp, lots=lots,
            risk_dollars=risk_dollars, rr=rr, reason=reason,
            highest=entry, lowest=entry,
        )
        self.open_signals.append(sig)
        self._append_signal_csv(sig)
        logger.info(f"Tracking: {sig.id} {direction} {symbol} @ {entry:.2f}")

    def check_signals(self, prices: dict[str, tuple[float, float, float]]):
        """
        Check all open signals against current prices.
        prices: {symbol: (close, high, low)}
        """
        for sig in list(self.open_signals):
            if sig.symbol not in prices:
                continue

            close, high, low = prices[sig.symbol]
            sig.bars_checked += 1
            sig.highest = max(sig.highest, high)
            sig.lowest = min(sig.lowest, low)

            # Trailing SL: move to breakeven at 1:1 R:R
            if config.TRAILING_SL_ENABLED and not sig.sl_moved_to_be:
                risk = abs(sig.entry - sig.sl)
                if sig.direction == "long" and sig.highest >= sig.entry + risk * config.TRAILING_SL_TRIGGER_RR:
                    sig.sl = sig.entry
                    sig.sl_moved_to_be = True
                    from live.notifier import send_trailing_sl
                    send_trailing_sl(sig.symbol, sig.direction, sig.sl)
                    logger.info(f"Trailing SL → BE: {sig.id}")
                elif sig.direction == "short" and sig.lowest <= sig.entry - risk * config.TRAILING_SL_TRIGGER_RR:
                    sig.sl = sig.entry
                    sig.sl_moved_to_be = True
                    from live.notifier import send_trailing_sl
                    send_trailing_sl(sig.symbol, sig.direction, sig.sl)
                    logger.info(f"Trailing SL → BE: {sig.id}")

            hit = False
            reason = ""
            exit_price = close

            # SL on wick
            if sig.direction == "long" and low <= sig.sl:
                hit, reason, exit_price = True, "SL", sig.sl
            elif sig.direction == "short" and high >= sig.sl:
                hit, reason, exit_price = True, "SL", sig.sl

            # TP on wick
            if not hit:
                if sig.direction == "long" and high >= sig.tp:
                    hit, reason, exit_price = True, "TP", sig.tp
                elif sig.direction == "short" and low <= sig.tp:
                    hit, reason, exit_price = True, "TP", sig.tp

            # Both hit — SL wins
            if sig.direction == "long" and low <= sig.sl and high >= sig.tp:
                hit, reason, exit_price = True, "SL", sig.sl
            elif sig.direction == "short" and high >= sig.sl and low <= sig.tp:
                hit, reason, exit_price = True, "SL", sig.sl

            # Time stop: 20 bars (checked every 5 min, so 20 * 12 = 240 checks ≈ 20 hours)
            if not hit and sig.bars_checked >= 240:
                hit, reason, exit_price = True, "TIME", close

            if hit:
                self._close_signal(sig, exit_price, reason)

    def _close_signal(self, sig: TrackedSignal, exit_price: float, reason: str):
        cfg = config.INSTRUMENTS.get(sig.symbol, {})
        lot_mult = cfg.get("lot_mult", 50)
        comm = cfg.get("comm", 3.0)

        if sig.direction == "long":
            raw_pnl = (exit_price - sig.entry) * sig.lots * lot_mult
        else:
            raw_pnl = (sig.entry - exit_price) * sig.lots * lot_mult

        pnl = raw_pnl - comm * sig.lots
        rr_actual = abs(exit_price - sig.entry) / abs(sig.entry - sig.sl) if abs(sig.entry - sig.sl) > 0 else 0
        if pnl < 0:
            rr_actual = -rr_actual

        result = {
            "id": sig.id,
            "symbol": sig.symbol,
            "direction": sig.direction,
            "quality": sig.quality,
            "entry": f"{sig.entry:.5f}",
            "exit_price": f"{exit_price:.5f}",
            "sl": f"{sig.sl:.5f}",
            "tp": f"{sig.tp:.5f}",
            "lots": f"{sig.lots:.2f}",
            "pnl": f"{pnl:.2f}",
            "rr_actual": f"{rr_actual:.2f}",
            "exit_reason": reason,
            "entry_time": sig.timestamp,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "bars_held": str(sig.bars_checked),
        }

        self.results.append(result)
        self.open_signals.remove(sig)
        self._append_result_csv(result)
        self._update_signal_status(sig.id, reason)

        # Update equity
        self.equity += pnl

        # Progress toward Phase 1
        target = config.ACCOUNT_SIZE * config.PROFIT_TARGET_PHASE1
        progress = max(0, (self.equity - config.ACCOUNT_SIZE) / target * 100) if target > 0 else 0

        # Send close notification
        from live.notifier import send_close
        send_close(sig.symbol, sig.direction, sig.entry, exit_price,
                   pnl, reason, self.equity, progress)

        logger.info(f"{'WIN' if pnl > 0 else 'LOSS'}: {sig.id} → {reason} "
                    f"@ {exit_price:.2f} PnL=${pnl:.2f} Equity=${self.equity:.2f}")

        return pnl

    def _append_signal_csv(self, sig: TrackedSignal):
        exists = os.path.exists(SIGNALS_FILE)
        with open(SIGNALS_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow({
                "id": sig.id, "timestamp": sig.timestamp,
                "symbol": sig.symbol, "direction": sig.direction,
                "quality": sig.quality, "entry": sig.entry,
                "sl": sig.sl, "tp": sig.tp, "lots": sig.lots,
                "risk_dollars": sig.risk_dollars, "rr": sig.rr,
                "reason": sig.reason, "status": "OPEN",
            })

    def _update_signal_status(self, sig_id: str, status: str):
        if not os.path.exists(SIGNALS_FILE):
            return
        rows = []
        with open(SIGNALS_FILE, "r") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            if row["id"] == sig_id:
                row["status"] = status
        with open(SIGNALS_FILE, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=SIGNAL_FIELDS)
            w.writeheader()
            w.writerows(rows)

    def _append_result_csv(self, result: dict):
        exists = os.path.exists(RESULTS_FILE)
        with open(RESULTS_FILE, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if not exists:
                w.writeheader()
            w.writerow(result)

    # --- Stats ---

    @property
    def total_trades(self) -> int:
        return len(self.results)

    @property
    def wins(self) -> int:
        return sum(1 for r in self.results if float(r["pnl"]) > 0)

    @property
    def total_pnl(self) -> float:
        return sum(float(r["pnl"]) for r in self.results)

    @property
    def win_rate(self) -> float:
        return (self.wins / self.total_trades * 100) if self.total_trades else 0

    @property
    def profit_factor(self) -> float:
        gw = sum(float(r["pnl"]) for r in self.results if float(r["pnl"]) > 0)
        gl = abs(sum(float(r["pnl"]) for r in self.results if float(r["pnl"]) < 0))
        return gw / gl if gl > 0 else float("inf") if gw > 0 else 0

    @property
    def consecutive_losses(self) -> int:
        count = 0
        for r in reversed(self.results):
            if float(r["pnl"]) < 0:
                count += 1
            else:
                break
        return count

    @property
    def drawdown_pct(self) -> float:
        if self.equity >= config.ACCOUNT_SIZE:
            return 0
        return (1 - self.equity / config.ACCOUNT_SIZE) * 100

    def check_dd_alerts(self):
        from live.notifier import send_dd_alert
        for level in config.DD_ALERT_LEVELS:
            level_pct = level * 100
            if self.drawdown_pct >= level_pct and level not in self.dd_alerts_sent:
                send_dd_alert(self.drawdown_pct, self.equity)
                self.dd_alerts_sent.add(level)
                logger.warning(f"DD alert sent: {self.drawdown_pct:.1f}%")

    def summary_text(self) -> str:
        if not self.results:
            return "No completed trades yet."

        pf = self.profit_factor
        pf_s = f"{pf:.2f}" if pf != float("inf") else "INF"

        lines = [
            f"Trades: {self.total_trades} ({self.wins}W/{self.total_trades - self.wins}L)",
            f"WR: {self.win_rate:.1f}%",
            f"PnL: ${self.total_pnl:,.2f}",
            f"PF: {pf_s}",
            f"Open: {len(self.open_signals)}",
        ]

        # Per quality
        for q in ["A+", "A", "B"]:
            qt = [r for r in self.results if r["quality"] == q]
            if qt:
                qw = sum(1 for r in qt if float(r["pnl"]) > 0)
                qp = sum(float(r["pnl"]) for r in qt)
                lines.append(f"  {q}: {len(qt)}T {qw}W ${qp:,.2f}")

        return "\n".join(lines)
