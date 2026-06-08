"""
agents/agent_pipeline_supervisor.py
════════════════════════════════════════════════════════════════════
Agent 3 — Pipeline Supervisor & Automation Guard

Responsibilities:
  • Pre-flight health checks on browser, data feed and model.
  • Wraps every trade execution with retry logic and timeout guard.
  • Detects asset closure / low-payout and switches to the best
    available 24/7 asset automatically.
  • Enforces circuit-breaker: stops trading if win-rate drops
    below the target threshold.
  • Recovers browser session on crash without restarting the loop.
════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
import os
import time
from typing import Optional, Tuple

try:
    from notifications.telegram import send_telegram
except Exception:
    def send_telegram(msg):
        return False

log = logging.getLogger("agent3_supervisor")

# Assets tried in order when the current one is unavailable
from agents.agent_multi_timeframe import PREFERRED_ASSETS_ORDERED
from agents.agent_asset_selector import pretty_asset


class CircuitBreaker:
    """
    Stops trading when the recent win-rate falls below the target.
    Auto-resets after a cooldown period.
    """

    def __init__(self, target_win_rate: float = 0.90,
                 min_trades: int = 5,
                 cooldown_s: float = 300.0):
        self.target          = target_win_rate
        self.min_trades      = min_trades
        self.cooldown_s      = cooldown_s
        self._tripped_at: Optional[float] = None
        self._wins  = 0
        self._total = 0

    def record(self, win: bool):
        self._total += 1
        if win:
            self._wins += 1
        # Keep only last 20 trades in memory
        if self._total > 20:
            self._total = 20
            if win:
                self._wins = min(self._wins, 20)

    @property
    def win_rate(self) -> float:
        return self._wins / self._total if self._total else 1.0

    def should_stop(self) -> Tuple[bool, str]:
        """Returns (True, reason) if trading should pause."""
        if self._tripped_at is not None:
            elapsed = time.time() - self._tripped_at
            if elapsed < self.cooldown_s:
                remaining = int(self.cooldown_s - elapsed)
                return True, f"Circuit breaker cooling down — {remaining}s left"
            else:
                log.info("[CB] Cooldown complete — resetting circuit breaker")
                self._tripped_at = None
                self._wins = self._total = 0

        if self._total >= self.min_trades and self.win_rate < self.target:
            self._tripped_at = time.time()
            return True, (f"Win-rate {self.win_rate:.0%} below target {self.target:.0%} "
                          f"— pausing {self.cooldown_s:.0f}s")
        return False, ""


class PipelineSupervisorAgent:
    """
    Pipeline Supervisor — Agent 3.

    Wraps trade execution and health monitoring.
    """

    MAX_TRADE_RETRIES        = 3
    TRADE_TIMEOUT_S          = 8.0
    MAX_CONSECUTIVE_FAILURES = 5

    def __init__(self, bot, risk_manager, agent1, agent2,
                 asset_selector=None, force_trades: bool = False,
                 disable_circuit_breaker: bool = False):
        self.bot   = bot
        self.risk  = risk_manager
        self.a1    = agent1
        self.a2    = agent2
        self.asset_selector = asset_selector
        self.force_trades = bool(force_trades)
        self.disable_circuit_breaker = bool(disable_circuit_breaker)
        self.cb    = CircuitBreaker(target_win_rate=0.90,
                                    min_trades=5,
                                    cooldown_s=300.0)
        self.consecutive_failures = 0
        self.max_consecutive_failures = self.MAX_CONSECUTIVE_FAILURES
        self._current_asset: Optional[str] = None

    # ── Health Checks ────────────────────────────────────────────

    async def verify_pipeline_health(self) -> Tuple[bool, str, bool]:
        """Quick health pulse — browser alive, data feed flowing.

        Returns (healthy, reason, recoverable).
        recoverable=False means the failure is due to risk or circuit breaker
        conditions and should not trigger browser recovery attempts.
        """
        # 1. Browser alive?
        try:
            ok = await asyncio.wait_for(self.bot.is_alive(), timeout=3.0)
            if not ok:
                reason = "Browser not alive"
                log.warning(f"[Supervisor] {reason}")
                return False, reason, True
        except Exception as e:
            reason = f"Browser health check error: {e}"
            log.warning(f"[Supervisor] {reason}")
            return False, reason, True
        from config import settings

        # 2. Risk manager not halted?
        status = self.risk.status()
        if status.get("halted"):
            if self.force_trades or getattr(settings, 'TRAINING_MODE', False):
                log.warning("[Supervisor] Risk manager halted but force/training override active")
            else:
                reason = "Risk manager has halted trading"
                log.warning(f"[Supervisor] {reason}")
                return False, reason, False

        # 3. Circuit breaker?
        stop, reason = self.cb.should_stop()
        if stop:
            if self.force_trades or self.disable_circuit_breaker:
                log.warning(f"[Supervisor] Circuit breaker active but override enabled: {reason}")
            else:
                log.warning(f"[Supervisor] Circuit breaker active: {reason}")
                try:
                    send_telegram(f"[Supervisor] Circuit breaker active: {reason}")
                except Exception:
                    pass
                return False, reason, False

        return True, "", True

    def handle_tick_outcome(self, success: bool):
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

    async def recover_pipeline(self) -> bool:
        """Attempt to recover a broken browser session."""
        log.warning("[Supervisor] Attempting pipeline recovery...")
        try:
            await self.bot.stop()
        except Exception:
            pass
        await asyncio.sleep(3)
        try:
            await self.bot.start()
            ok = await self.bot.login()
            if ok:
                await self.bot.select_asset(
                    self._current_asset or self.a1.asset
                )
                self.consecutive_failures = 0
                log.info("[Supervisor] Pipeline recovered successfully.")
                return True
        except Exception as e:
            log.error(f"[Supervisor] Recovery failed: {e}")
        return False

    # ── Asset Management ─────────────────────────────────────────

    async def ensure_best_asset(self, current_asset: Optional[str] = None) -> str:
        """
        Verify the current asset is open, then scan FTT candidates.
        Asset only changes when bot/selector proposes a switch (no forced timer rotation).
        Returns the asset that is now active.
        """
        import time
        from config import settings

        current_asset = current_asset or self._current_asset or settings.ASSET
        current_time = time.time()

        current_score = await self._score_asset(current_asset)
        payout = await self._get_payout(current_asset)
        best_asset = current_asset
        best_score = current_score if current_score is not None else 0.0

        if payout and payout >= 70:
            log.info(f"[Supervisor] Current asset {pretty_asset(current_asset)} payout={payout}%")
            best_score = max(best_score, 0.0)
        else:
            log.info(f"[Supervisor] Current asset {pretty_asset(current_asset)} payout={payout}% — scanning alternatives")

        if self.asset_selector is not None:
            pos_info = self.asset_selector.analyze_position(current_asset)
            if pos_info.get("has_open_position") and not pos_info.get("should_add"):
                log.info(f"[Supervisor] Current asset {pretty_asset(current_asset)} has open position and is not suitable for adding")
                best_score = -1.0

        # Use asset selector agent to score candidate assets if available
        if self.asset_selector is not None:
            selection_asset, selection_score, selection_reason = self.asset_selector.select_best_asset(
                available_assets=PREFERRED_ASSETS_ORDERED,
                current_asset=current_asset,
                exclude_current=False,
            )
            if selection_asset and selection_score is not None:
                log.info(f"[Supervisor] Selector proposed {pretty_asset(selection_asset)} score={selection_score:.2%} | {selection_reason}")
                selected_payout = await self._get_payout(selection_asset)
                log.info(f"[Supervisor] Selector payload info: current_payout={payout}% selected_payout={selected_payout}% current_score={best_score:.2%}")

                # Prefer the selector's high-scoring alternative asset when it is stronger
                # than the current asset, even if the payout data is not available.
                if selected_payout and selected_payout >= 70 and selection_score > best_score:
                    log.info(f"[Supervisor] Selector prefers {pretty_asset(selection_asset)} over {pretty_asset(best_asset)}")
                    best_asset = selection_asset
                    best_score = selection_score
                elif selected_payout is None and payout is None and selection_score > best_score:
                    log.info(f"[Supervisor] No payout data; switching to stronger selector asset {pretty_asset(selection_asset)}")
                    best_asset = selection_asset
                    best_score = selection_score
                elif selected_payout is None and payout is not None and selection_score > best_score + 0.05:
                    log.info(f"[Supervisor] Selector asset {pretty_asset(selection_asset)} is stronger despite missing payout data")
                    best_asset = selection_asset
                    best_score = selection_score
        for asset in PREFERRED_ASSETS_ORDERED:
            if asset == current_asset:
                continue
            score = await self._score_asset(asset)
            if score is None or abs(score) < 0.25:
                continue
            asset_payout = await self._get_payout(asset)
            if not asset_payout or asset_payout < 70:
                continue
            if abs(score) > abs(best_score) + 0.12:
                log.info(f"[Supervisor] Found stronger asset {pretty_asset(asset)} score={score:.3f} payout={asset_payout}%")
                best_asset = asset
                best_score = score

        if best_asset != current_asset:
            log.info(f"[Supervisor] Switching to best asset {pretty_asset(best_asset)} | score={best_score:.3f}")
            if await self.bot.select_asset(best_asset):
                self._current_asset = best_asset
                return best_asset
            log.warning(f"[Supervisor] Failed to select {pretty_asset(best_asset)}; keeping {pretty_asset(current_asset)}")

        self._current_asset = current_asset
        return current_asset


    async def _score_asset(self, asset: str) -> Optional[float]:
        """Estimate the asset's current FTT momentum from 1m/5m chart patterns."""
        try:
            from data.fetcher import fetch_multi_timeframe
            dfs = await asyncio.to_thread(fetch_multi_timeframe, asset, ["M1", "M5"])
        except Exception:
            return None

        if not dfs:
            return None

        total = 0.0
        weight = 0.0
        for tf, df in dfs.items():
            if df.empty or len(df) < 8:
                continue
            last = df.iloc[-1]
            start = df.iloc[-5] if len(df) >= 5 else df.iloc[0]
            momentum = (last.close - start.close) / max(abs(start.close), 1e-9)
            trend = 0.0
            if last.macd > last.macd_signal:
                trend += 0.6
            elif last.macd < last.macd_signal:
                trend -= 0.6
            if last.rsi < 40:
                trend += 0.35
            elif last.rsi > 60:
                trend -= 0.35
            bb_mid = float(last.bb_mid)
            if last.close > bb_mid:
                trend += 0.15
            else:
                trend -= 0.15

            tf_weight = 1.0 if tf == "M1" else 1.3
            total += tf_weight * (momentum * 40 + trend)
            weight += tf_weight

        if weight == 0:
            return None

        score = total / weight
        return float(score)

    async def _get_payout(self, asset: str) -> Optional[int]:
        """Try to read the payout percentage for an asset from the UI."""
        try:
            payout = await asyncio.wait_for(
                self.bot.get_asset_payout(asset), timeout=5.0
            )
            return payout
        except Exception:
            return None

    # ── Trade Execution ──────────────────────────────────────────

    async def safe_execute_trade(self, signal, confidence: float, mode: str) -> bool:
        """
        Execute a trade with:
          • Risk-manager approval gate
          • Up to 3 retries with back-off
          • Per-attempt timeout
          • Circuit-breaker recording
        """
        from config import settings

        # Gate 1: Risk approval (fetch payout, estimate trade risk, pass training override)
        from config import settings
        trade_amt = getattr(settings, 'TRADE_AMOUNT', None)
        payout = None
        try:
            # try to fetch payout for the current asset (best-effort)
            payout = await asyncio.wait_for(self._get_payout(self._current_asset or self.a1.asset), timeout=3.0)
        except Exception:
            payout = None

        approved, reason = self.risk.approve(
            signal, confidence, mode,
            trade_amount=trade_amt,
            allow_during_training=getattr(settings, 'TRAINING_MODE', False),
        )
        if not approved:
            if self.force_trades or getattr(settings, 'TRAINING_MODE', False):
                log.warning(f"[Supervisor] Risk blocked but override active: {reason}")
            else:
                log.info(f"[Supervisor] Risk blocked: {reason}")
                return False

        # Gate 2: Confidence minimum (hard floor)
        if confidence < settings.CONFIDENCE_THRESHOLD:
            if self.force_trades:
                log.warning(f"[Supervisor] Confidence {confidence:.2%} < threshold — FORCED")
            else:
                log.info(f"[Supervisor] Confidence {confidence:.2%} < threshold — skip")
                return False

        # Gate 3: Circuit breaker
        stop, cb_reason = self.cb.should_stop()
        if stop:
            if self.force_trades or self.disable_circuit_breaker:
                log.warning(f"[Supervisor] CB active but override enabled: {cb_reason}")
            else:
                log.info(f"[Supervisor] CB skip: {cb_reason}")
                return False

        # Execute with retries
        for attempt in range(1, self.MAX_TRADE_RETRIES + 1):
            try:
                log.info(f"[Supervisor] Execute attempt {attempt}/{self.MAX_TRADE_RETRIES} "
                         f"— {signal.value} {mode.upper()}")

                if mode == "ftt":
                    result = await asyncio.wait_for(
                        self.bot.place_ftt_trade(signal, settings.FTT_DURATION, asset=self._current_asset),
                        timeout=self.TRADE_TIMEOUT_S,
                    )
                else:
                    result = await asyncio.wait_for(
                        self.bot.place_forex_trade(signal, asset=self._current_asset),
                        timeout=self.TRADE_TIMEOUT_S,
                    )

                if result:
                    log.info(f"[Supervisor] Trade placed successfully (attempt {attempt})")
                    return True

                log.warning(f"[Supervisor] Attempt {attempt} returned False")
            except asyncio.TimeoutError:
                log.warning(f"[Supervisor] Attempt {attempt} timed out after {self.TRADE_TIMEOUT_S}s")
            except Exception as e:
                log.error(f"[Supervisor] Attempt {attempt} exception: {e}")

            if attempt < self.MAX_TRADE_RETRIES:
                await asyncio.sleep(0.5 * attempt)   # 0.5s, 1.0s back-off

        log.error("[Supervisor] All trade attempts failed")
        return False

    def record_outcome(self, win: bool):
        """Update circuit breaker after a trade resolves."""
        self.cb.record(win)
        wr = self.cb.win_rate
        log.info(f"[Supervisor] CB win-rate={wr:.0%} | total={self.cb._total}")
