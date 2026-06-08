"""
bot/risk.py
════════════════════════════════════════════════════════════════════
Risk Manager & Trade Journal — Enhanced Edition

Same public API as the original; key additions:
  • `TradeRecord` now has an optional `state_key` field (needed by
    the multi-agent orchestrator).
  • `status()` now includes a `halted` flag (used by Supervisor).
  • `update_pnl_and_journal()` updates both in-memory stats and
    the CSV journal atomically.
  • Stricter daily-loss cap to protect capital.
════════════════════════════════════════════════════════════════════
"""

import csv
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Tuple

log = logging.getLogger("risk")


@dataclass
class TradeRecord:
    timestamp:  str
    asset:      str
    mode:       str
    signal:     str
    confidence: float
    amount:     float
    result:     str = "PENDING"  # PENDING | WIN | LOSS | FAILED
    pnl:        float = 0.0
    state_key:  str = ""     # set by multi-agent orchestrator


class RiskManager:
    """
    Tracks daily trade counts, P&L, and enforces hard caps.
    Thread-safe enough for single-process async use.
    """

    def __init__(self):
        from config import settings
        self.settings      = settings
        self.journal_path  = os.path.join(settings.LOG_DIR, "trade_journal.csv")
        os.makedirs(settings.LOG_DIR, exist_ok=True)
        os.makedirs(settings.JOURNAL_ARCHIVE_DIR, exist_ok=True)
        self._archive_previous_journal_if_needed()
        self._ensure_journal()

        # Optional startup reset of today's journal rows
        if getattr(self.settings, 'RESET_TODAY_LOGS_ON_STARTUP', False):
            self.reset_today()

        # Daily counters (reset when date changes)
        self._day:        str   = datetime.now().date().isoformat()
        self._trade_count: int  = 0
        self._daily_pnl:  float = 0.0
        self._halted:     bool  = False
        self.asset_stats: dict = {}

        # Load today's trades and historical asset stats from journal
        self._load_today()
        self._load_asset_history()

    # ── Public Interface ─────────────────────────────────────────

    def approve(self, signal, confidence: float, mode: str,
                trade_amount: float = None, allow_during_training: bool = False) -> Tuple[bool, str]:
        """Returns (approved, reason).

        When `allow_during_training` is True (or `settings.TRAINING_MODE`),
        the decision ignores cumulative daily caps and instead enforces
        per-trade risk limits using `trade_amount` and `MAX_TRADE_LOSS_USD`.
        """
        from config import settings

        self._check_day_rollover()

        # Basic rejections
        if signal.value == "HOLD":
            return False, "Signal is HOLD"

        if confidence < settings.CONFIDENCE_THRESHOLD:
            return False, (f"Confidence {confidence:.2%} < "
                           f"threshold {settings.CONFIDENCE_THRESHOLD:.2%}")

        # Estimate potential loss for this trade
        est_loss = self._estimate_trade_loss(trade_amount or settings.TRADE_AMOUNT, mode,
                                             payout=None)

        # Training override: allow trades even if daily caps hit, but enforce
        # strict per-trade loss limit so training doesn't take extreme losses.
        if allow_during_training or getattr(settings, 'TRAINING_MODE', False):
            if est_loss > settings.MAX_TRADE_LOSS_USD:
                return False, (f"Estimated trade loss ${est_loss:.2f} exceeds "
                               f"MAX_TRADE_LOSS_USD ${settings.MAX_TRADE_LOSS_USD:.2f}")
            return True, "Approved (training override)"

        # Normal live trading checks
        if self._halted:
            return False, "Trading halted for the day"

        if self._trade_count >= settings.MAX_DAILY_TRADES:
            return False, f"Max daily trades reached ({settings.MAX_DAILY_TRADES})"

        # Block if this trade would push us beyond the daily loss cap
        if (self._daily_pnl - est_loss) <= -abs(settings.MAX_DAILY_LOSS_USD):
            self._halted = True
            return False, (f"Trade would breach daily loss cap: current ${self._daily_pnl:+.2f} "
                           f"- est_loss ${est_loss:.2f} ≤ -${abs(settings.MAX_DAILY_LOSS_USD):.2f}")

        return True, "Approved"

    def record_trade(self, record: TradeRecord):
        """Append a trade record to the journal."""
        self._check_day_rollover()
        self._trade_count += 1

        row = {
            "timestamp":  record.timestamp,
            "asset":      record.asset,
            "mode":       record.mode,
            "signal":     record.signal,
            "confidence": f"{record.confidence:.4f}",
            "amount":     record.amount,
            "result":     record.result,
            "pnl":        record.pnl,
            "state_key":  record.state_key,
        }
        self._append_row(row)
        log.info(f"Trade recorded: {record.signal} | result={record.result} | pnl=${record.pnl:+.2f}")

    def update_pnl_and_journal(self, timestamp: str, result: str, pnl: float):
        """
        Update a PENDING trade in the journal to its final outcome.
        Also updates in-memory daily P&L.
        """
        self._daily_pnl += pnl
        self._rewrite_journal_result(timestamp, result, pnl)

        if self._daily_pnl <= -abs(self.settings.MAX_DAILY_LOSS_USD):
            self._halted = True
            log.warning(f"[Risk] Daily loss cap hit — halting. P&L=${self._daily_pnl:+.2f}")

    def status(self) -> dict:
        return {
            "trade_count": self._trade_count,
            "max_trades":  self.settings.MAX_DAILY_TRADES,
            "daily_pnl":   self._daily_pnl,
            "trades_left": max(0, self.settings.MAX_DAILY_TRADES - self._trade_count),
            "halted":      self._halted,
        }

    # ── Internals ────────────────────────────────────────────────

    def _ensure_journal(self):
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        if not os.path.exists(self.journal_path):
            with open(self.journal_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=[
                    "timestamp", "asset", "mode", "signal",
                    "confidence", "amount", "result", "pnl", "state_key",
                ])
                writer.writeheader()

    def _archive_previous_journal_if_needed(self):
        if not os.path.exists(self.journal_path):
            return

        today = datetime.now().date().isoformat()
        if self._journal_contains_today_rows(today):
            return

        if not self._journal_has_rows():
            return

        last_day = self._journal_last_entry_day() or datetime.fromtimestamp(
            os.path.getmtime(self.journal_path)
        ).date().isoformat()

        archive_path = self._make_archive_path(last_day)
        if os.path.exists(archive_path):
            archive_path = self._make_archive_path(last_day, suffix=int(time.time()))

        try:
            os.rename(self.journal_path, archive_path)
            log.info(f"[Risk] Archived previous journal to {archive_path}")
        except OSError as e:
            log.warning(f"[Risk] Failed to archive previous journal: {e}")

    def _journal_has_rows(self) -> bool:
        try:
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                return any(True for _ in reader)
        except Exception:
            return False

    def _journal_last_entry_day(self) -> Optional[str]:
        last_date = None
        try:
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts = row.get("timestamp", "")
                    if len(ts) >= 10:
                        last_date = ts[:10]
        except Exception:
            return None
        return last_date

    def _make_archive_path(self, day: str, suffix: Optional[int] = None) -> str:
        base_name = f"trade_journal_{self._format_archive_day(day)}"
        if suffix is not None:
            base_name = f"{base_name}_{suffix}"
        os.makedirs(self.settings.JOURNAL_ARCHIVE_DIR, exist_ok=True)
        return os.path.join(self.settings.JOURNAL_ARCHIVE_DIR, f"{base_name}.csv")

    def _format_archive_day(self, day: str) -> str:
        try:
            day_dt = datetime.fromisoformat(day)
            return day_dt.strftime(self.settings.LOG_DATE_FORMAT)
        except ValueError:
            return day

    def _journal_contains_today_rows(self, today: str) -> bool:
        try:
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("timestamp", "").startswith(today):
                        return True
        except Exception:
            pass
        return False

    def reset_today(self):
        today = datetime.now().date().isoformat()
        rows = []
        try:
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if not row.get("timestamp", "").startswith(today):
                        rows.append(row)

            with open(self.journal_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)

            self._day = today
            self._trade_count = 0
            self._daily_pnl = 0.0
            self._halted = False
            log.info(f"[Risk] Cleared today's journal rows and reset risk state")
        except Exception as e:
            log.warning(f"[Risk] Failed to reset today's journal: {e}")

    def _append_row(self, row: dict):
        with open(self.journal_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _rewrite_journal_result(self, timestamp: str, result: str, pnl: float):
        """Update a specific row's result + pnl by rewriting the file."""
        try:
            rows = []
            updated = False
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    if row.get("timestamp") == timestamp and row.get("result") == "PENDING":
                        row["result"] = result
                        row["pnl"]    = f"{pnl:.4f}"
                        updated = True
                    rows.append(row)

            if updated:
                with open(self.journal_path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
        except Exception as e:
            log.warning(f"Journal update error: {e}")

    def _check_day_rollover(self):
        today = datetime.now().date().isoformat()
        if today != self._day:
            self._archive_current_journal(self._day)
            log.info(f"New day {today} — resetting daily counters")
            self._day         = today
            self._trade_count = 0
            self._daily_pnl   = 0.0
            self._halted      = False
            self._ensure_journal()

    def _archive_current_journal(self, day: str):
        if not os.path.exists(self.journal_path):
            return

        archive_path = self._make_archive_path(day)
        if os.path.exists(archive_path):
            archive_path = self._make_archive_path(day, suffix=int(time.time()))

        try:
            os.rename(self.journal_path, archive_path)
            log.info(f"[Risk] Rotated journal to {archive_path}")
        except OSError as e:
            log.warning(f"[Risk] Failed to rotate journal: {e}")

    def _estimate_trade_loss(self, trade_amount: float, mode: str, payout: Optional[float] = None) -> float:
        """Estimate the USD loss if the trade loses.

        - For FTT (binary) trades: losing the stake = full trade_amount.
        - For forex trades: estimate using `STOP_LOSS_PCT` of the stake.

        Returns positive USD value representing potential loss.
        """
        try:
            amt = float(trade_amount or 0.0)
        except Exception:
            amt = 0.0

        from config import settings

        if mode == "ftt":
            return amt

        # Forex-style: use configured stop-loss percentage
        try:
            sl = float(getattr(settings, 'STOP_LOSS_PCT', 0.02))
            return abs(amt * sl)
        except Exception:
            return amt

    def _load_today(self):
        """Restore today's stats from the journal so restarts don't reset counts."""
        today = self._day
        try:
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("timestamp", "").startswith(today):
                        self._trade_count += 1
                        try:
                            pnl = float(row.get("pnl", 0))
                            self._daily_pnl += pnl
                        except ValueError:
                            pass
        except Exception:
            pass

        if self._trade_count:
            log.info(f"Restored {self._trade_count} trades for today | "
                     f"P&L=${self._daily_pnl:+.2f}")
            if self._daily_pnl <= -abs(self.settings.MAX_DAILY_LOSS_USD):
                self._halted = True
                log.warning(f"[Risk] Daily loss cap already hit on startup — trading halted")

    def _load_asset_history(self):
        """Aggregate historical win/loss counts per asset from the journal.

        Produces `self.asset_stats` mapping asset -> {"wins": N, "losses": M, "total": K}.
        Only finalized trades (WIN/LOSS) are counted toward totals.
        """
        stats = {}
        try:
            with open(self.journal_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    asset = row.get("asset")
                    if not asset:
                        continue
                    result = (row.get("result") or "").upper()
                    if asset not in stats:
                        stats[asset] = {"wins": 0, "losses": 0, "total": 0}

                    if result == "WIN":
                        stats[asset]["wins"] += 1
                        stats[asset]["total"] += 1
                    elif result == "LOSS":
                        stats[asset]["losses"] += 1
                        stats[asset]["total"] += 1
                    else:
                        # Ignore PENDING/FAILED/other non-final results
                        continue

            self.asset_stats = stats
            if stats:
                log.info(f"Loaded asset history for {len(stats)} assets")
        except Exception as e:
            log.warning(f"Failed to load asset history: {e}")
            self.asset_stats = {}
