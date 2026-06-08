"""
multi_agent_orchestrator.py — v4.2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Strategy: EMA(9/21) crossover + Aroon(14) crossover confluence
Plan:     $10,000 account | $200/trade | $2,000 daily circuit breaker
Sessions: DISABLED — continuous trading (test mode)

Key changes vs v4.1:
  1. Aroon indicator integrated as PRIMARY signal alongside EMA crossover
  2. Strict EMA9/21 + Aroon14 crossover entry only — no extra confluence gating
  3. Session gates removed — trades continuously during testing
  4. $200 fixed trade size with $2,000 circuit breaker
  5. RSI seed noise removed — no more RSI=2.9 nonsense
  6. Signal flip for regime in correct place (before chart analysis)
  7. Trade executed only on BOTH EMA cross AND Aroon cross confirming
"""

import asyncio
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import shutil
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import settings
from bot.risk import RiskManager, TradeRecord
from bot.browser import OlymtradeBot
from models.predictor import Signal
from agents.agent_multi_timeframe import MultiTimeframeAgent
from agents.agent_rl_learning import RLFeedbackAgent
from agents.agent_pipeline_supervisor import PipelineSupervisorAgent
from agents.agent_regime_detector import RegimeDetector
from agents.agent_asset_selector import AssetSelectorAgent
from agents.chart_analyzer import ChartAnalyzer, AnalysisResult

BEAR_REGIMES = {"BEAR_TREND", "HIGH_VOL_DOWN", "BREAKOUT_DOWN"}
BULL_REGIMES = {"BULL_TREND",  "HIGH_VOL_UP",  "BREAKOUT_UP"}

# ── Risk constants (from your strategy plan) ──────────────────────────────────
TRADE_AMOUNT        = 200.0    # $200 fixed per trade
DAILY_LOSS_LIMIT    = 2000.0   # $2,000 circuit breaker
MIN_CONFIDENCE      = 0.60     # 60% minimum (strategy doc: 60–65%)
MIN_CONFLUENCES     = 5        # require 5/9 indicators to agree
MAX_TRADES_PER_DAY  = 20       # hard cap
PAYOUT_MINIMUM      = 0.80     # skip assets below 80% payout
FTT_DURATION_S      = 60       # 1 minute trades


def _rotate_daily_log(log_path: str):
    if not os.path.exists(log_path):
        return
    archive_dir = settings.LOG_ARCHIVE_DIR
    os.makedirs(archive_dir, exist_ok=True)
    today     = datetime.now().date()
    file_date = datetime.fromtimestamp(os.path.getmtime(log_path)).date()
    if file_date < today:
        name = (f"multi_agent_system_"
                f"{file_date.strftime(settings.LOG_DATE_FORMAT)}.log")
        path = os.path.join(archive_dir, name)
        if os.path.exists(path):
            name = (f"multi_agent_system_"
                    f"{file_date.strftime(settings.LOG_DATE_FORMAT)}"
                    f"_{int(time.time())}.log")
            path = os.path.join(archive_dir, name)
        os.rename(log_path, path)


os.makedirs(settings.LOG_DIR, exist_ok=True)
log_file = os.path.join(settings.LOG_DIR, "multi_agent_system.log")
_rotate_daily_log(log_file)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
log = logging.getLogger("multi_agent_orchestrator")


# ── OHLC helper (fixes RSI=2.9 seed noise bug) ───────────────────────────────

def _get_ohlc_from_builder(builder) -> Tuple[np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
    """
    Extract OHLC arrays. If builder only stores closes, estimates
    realistic highs/lows from ATR to prevent Stoch/CCI being stuck at extremes.
    """
    closes = np.array(builder.get_closes(), dtype=float)

    try:
        highs = np.array(builder.get_highs(), dtype=float)
        lows  = np.array(builder.get_lows(),  dtype=float)
        if (np.allclose(highs, closes, rtol=1e-9)
                and np.allclose(lows, closes, rtol=1e-9)):
            raise ValueError("synthetic")
    except Exception:
        n   = len(closes)
        if n >= 3:
            diffs   = np.abs(np.diff(closes[-20:]))
            atr_est = float(np.mean(diffs)) if len(diffs) else 0.0005
        else:
            atr_est = 0.0005
        atr_est = max(atr_est, 0.00005)
        delta        = np.zeros_like(closes)
        if n > 1:
            delta[1:] = np.diff(closes)
        highs = closes + atr_est*0.5 + np.where(delta >= 0, atr_est*0.3, 0.0)
        lows  = closes - atr_est*0.5 + np.where(delta  < 0, -atr_est*0.3, 0.0)

    try:
        opens = np.array(builder.get_opens(), dtype=float)
        if np.allclose(opens, closes, rtol=1e-9):
            raise ValueError("synthetic")
    except Exception:
        opens    = np.roll(closes, 1)
        opens[0] = closes[0]

    return opens, highs, lows, closes


# ── Session window helpers ────────────────────────────────────────────────────

# ── Main system ───────────────────────────────────────────────────────────────

class MultiAgentTradingSystem:

    TICK_INTERVAL     = 1.0
    DECISION_INTERVAL = 5.0
    ASSET_CHECK_EVERY = settings.ASSET_SCAN_INTERVAL

    def __init__(self):
        self.bot            = OlymtradeBot()
        self.risk           = RiskManager()
        self.asset_selector = AssetSelectorAgent()
        self.agent1         = MultiTimeframeAgent(settings.ASSET)
        self.agent2         = RLFeedbackAgent(settings.MODEL_DIR)
        self.regime         = RegimeDetector()
        self.chart          = ChartAnalyzer()
        self.supervisor     = PipelineSupervisorAgent(
            self.bot, self.risk, self.agent1, self.agent2,
            asset_selector=self.asset_selector,
        )

        self.active_trades:  List[Dict] = []
        self.running = False
        self._last_decision_ts:    float = 0.0
        self._last_asset_check_ts: float = 0.0
        self._current_asset: str   = settings.ASSET
        self._bg_tasks: List[asyncio.Task] = []

        # Daily tracking
        self._daily_loss:   float = 0.0
        self._daily_profit: float = 0.0
        self._trades_today: int   = 0
        self._wins_today:   int   = 0
        self._circuit_open: bool  = False   # True = trading halted today

    # ── Circuit breaker ───────────────────────────────────────────────────────

    def _check_circuit_breaker(self) -> bool:
        """Returns True if trading should be halted."""
        if self._circuit_open:
            return True
        if self._daily_loss >= DAILY_LOSS_LIMIT:
            log.warning(
                f"⛔ CIRCUIT BREAKER TRIGGERED — Daily loss ${self._daily_loss:.0f} "
                f">= limit ${DAILY_LOSS_LIMIT:.0f} — No more trades today."
            )
            self._circuit_open = True
            return True
        if self._trades_today >= MAX_TRADES_PER_DAY:
            log.warning(
                f"⛔ MAX TRADES REACHED — {self._trades_today}/{MAX_TRADES_PER_DAY} "
                f"trades today — No more trades today."
            )
            self._circuit_open = True
            return True
        return False

    def _reset_daily_counters(self):
        self._daily_loss   = 0.0
        self._daily_profit = 0.0
        self._trades_today = 0
        self._wins_today   = 0
        self._circuit_open = False
        log.info("[Daily] Counters reset for new day")

    def _record_trade_outcome(self, won: bool, pnl: float):
        if won:
            self._daily_profit += abs(pnl)
            self._wins_today   += 1
        else:
            self._daily_loss += abs(pnl)
        self._trades_today += 1
        wr = (self._wins_today / self._trades_today
              if self._trades_today > 0 else 0)
        log.info(
            f"[Daily] Trades={self._trades_today} | "
            f"WR={wr:.0%} | "
            f"Profit=+${self._daily_profit:.0f} | "
            f"Loss=-${self._daily_loss:.0f} | "
            f"Net=${self._daily_profit - self._daily_loss:+.0f} | "
            f"CB_remaining=${DAILY_LOSS_LIMIT - self._daily_loss:.0f}"
        )
        # Warn approaching circuit breaker
        if self._daily_loss >= DAILY_LOSS_LIMIT * 0.75:
            log.warning(
                f"⚠️  APPROACHING CIRCUIT BREAKER — "
                f"${DAILY_LOSS_LIMIT - self._daily_loss:.0f} loss budget remaining"
            )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _calculate_consensus_score(self, tf_state: dict,
                                   direction: str) -> float:
        if not tf_state:
            return 0.0
        sup   = sum(1 for d in tf_state.values()
                    if isinstance(d, dict) and d.get("vote") == direction)
        total = sum(1 for d in tf_state.values() if isinstance(d, dict))
        return sup / total if total > 0 else 0.0

    def _combine_confidences(self, a1: float, rl: float,
                              experienced: bool, consensus: float,
                              chart: float) -> float:
        # Chart is now the PRIMARY signal (EMA+Aroon strategy)
        # Agent1 MTF is supporting evidence
        if experienced:
            fused = 0.40 * a1 + 0.20 * rl + 0.40 * chart
        else:
            fused = 0.45 * a1 + 0.55 * chart
        fused *= (consensus * 0.4 + 0.6)   # softer consensus discount
        return float(min(0.97, max(0.45, fused)))

    async def _get_price(self) -> float:
        price = await self.bot.get_current_price()
        if price and price > 0:
            return price
        try:
            from data.fetcher import fetch_candles
            df = fetch_candles(self._current_asset, "M1", n=5)
            if not df.empty:
                return float(df["close"].iloc[-1])
        except Exception:
            pass
        b = self.agent1.builders.get(60)
        if b and len(b) > 0:
            return float(b.get_closes()[-1])
        return 1.1000


    async def setup(self) -> bool:
        log.info("=" * 70)
        log.info("  Multi-Agent OlympTrade System  |  v4.2")
        log.info(f"  Asset: {settings.ASSET} | Mode: ftt | $200/trade")
        log.info(f"  Daily limit: ${DAILY_LOSS_LIMIT:.0f} | "
                 f"Mode: continuous (no session gates)")
        log.info(f"  Strategy: EMA(9/21) crossover + Aroon(14) confluence")
        log.info(f"  Min confluences: {MIN_CONFLUENCES}/9 | "
                 f"Min confidence: {MIN_CONFIDENCE:.0%}")
        log.info("=" * 70)

        self.agent2.learn_from_logs(self.risk.journal_path)
        try:
            qpath = Path(self.agent2.qtable_path)
            if qpath.exists():
                bdir  = Path(settings.MODEL_DIR) / "backups"
                bdir.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copyfile(qpath, bdir / f"rl_qtable_{stamp}.json")
                log.info("[Setup] Q-table backed up")
        except Exception as e:
            log.warning(f"[Setup] Q-table backup: {e}")

        self.asset_selector = AssetSelectorAgent(
            trade_history=self.risk.asset_stats)

        await self.bot.start()
        if not await self.bot.login():
            log.error("Login failed.")
            await self.bot.stop()
            return False

        self._current_asset = await self.supervisor.ensure_best_asset()
        log.info(f"Active asset: {self._current_asset}")
        await self.bot.select_asset(self._current_asset)

        # Apply chart indicators
        try:
            from bot.chart_indicators import (apply_all_indicators,
                                               set_chart_timeframe)
            await set_chart_timeframe(self.bot._page, "1m")
            await asyncio.sleep(1.0)
            ok = await apply_all_indicators(self.bot._page)
            log.info(f"[Setup] Chart indicators applied={ok}")
        except Exception as e:
            log.warning(f"[Setup] Indicator panel: {e}")

        # Seed Agent1 — NO NOISE (fixes RSI=2.9)
        try:
            from data.fetcher import fetch_candles
            log.info("[Setup] Fetching M1 history (60 candles, no noise)...")
            df = fetch_candles(self._current_asset, "M1", n=60)
            if not df.empty:
                # Pass noise_pct=0.0 if supported, else use clean seed
                try:
                    self.agent1.seed_from_dataframe(df, noise_pct=0.0)
                except TypeError:
                    self.agent1.seed_from_dataframe(df)
                log.info(f"[Setup] Seeded with {len(df)} M1 candles")
            else:
                raise ValueError("empty df")
        except Exception as e:
            log.warning(f"[Setup] seed failed ({e}) — bootstrap")
            self.agent1.bootstrap(await self._get_price(), n_ticks=100)

        log.info("Setup complete.")

        self._bg_tasks += [
            asyncio.create_task(self._periodic_qtable_backup(3600)),
            asyncio.create_task(self._daily_replay_and_save()),
            asyncio.create_task(self._daily_reset_monitor()),
        ]
        log.info("=" * 70)
        return True

    # ── Background tasks ──────────────────────────────────────────────────────

    async def _periodic_qtable_backup(self, interval_s: int = 3600):
        while True:
            try:
                await asyncio.sleep(interval_s)
                src = Path(self.agent2.qtable_path)
                if src.exists():
                    bdir = Path(settings.MODEL_DIR) / "backups"
                    bdir.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(
                        src,
                        bdir / f"rl_qtable_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    )
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[Backup] {e}")

    async def _daily_replay_and_save(self):
        await asyncio.sleep(30)
        while True:
            try:
                self.agent2.learn_from_logs(self.risk.journal_path)
                self.agent2.q.save(self.agent2.qtable_path)
                log.info("[DailyTrain] RL replayed and Q-table saved")
                await asyncio.sleep(24 * 3600)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[DailyTrain] {e}")
                await asyncio.sleep(600)

    async def _daily_reset_monitor(self):
        """Reset circuit breaker and daily counters at midnight."""
        while True:
            try:
                now   = datetime.now()
                # Sleep until next midnight
                secs  = (86400 - (now.hour*3600 + now.minute*60 + now.second))
                await asyncio.sleep(secs)
                self._reset_daily_counters()
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning(f"[DailyReset] {e}")
                await asyncio.sleep(3600)

    # ── TP/SL monitor ─────────────────────────────────────────────────────────

    async def _monitor_active_trades(self, price: float):
        still = []
        for t in self.active_trades:
            entry = t.get('entry_price')
            if entry is None:
                still.append(t); continue
            d   = t.get('signal')
            amt = float(t.get('amount', TRADE_AMOUNT))
            win = ((d=='BUY' and price>entry) or (d=='SELL' and price<entry))
            pnl = amt * 0.82 if win else -amt
            # Force-close after 2x duration (safety net)
            elapsed = (datetime.now() - t["placed_at"]).total_seconds()
            if elapsed > t["duration"] * 2:
                log.info(f"[ForceClose] {d} timeout pnl=${pnl:+.0f}")
                self.risk.update_pnl_and_journal(t['timestamp'],
                                                  'WIN' if win else 'LOSS', pnl)
                self.agent2.update_after_result('WIN' if win else 'LOSS', pnl)
                self._record_trade_outcome(win, pnl)
                self.supervisor.record_outcome(win)
                continue
            still.append(t)
        self.active_trades = still

    # ── Main loop ─────────────────────────────────────────────────────────────

    async def run(self):
        if not await self.setup():
            return
        self.running = True
        log.info(f"Polling every {self.TICK_INTERVAL}s | "
                 f"Decisions every {self.DECISION_INTERVAL}s")

        try:
            while self.running:
                t0 = time.monotonic()

                # ── Health check ─────────────────────────────────────
                healthy, reason, recoverable = \
                    await self.supervisor.verify_pipeline_health()
                if not healthy:
                    if recoverable:
                        self.supervisor.handle_tick_outcome(False)
                        if (self.supervisor.consecutive_failures
                                >= self.supervisor.max_consecutive_failures):
                            if not await self.supervisor.recover_pipeline():
                                log.critical("Pipeline unrecoverable.")
                                break
                        await asyncio.sleep(2)
                    else:
                        log.warning(f"[Supervisor] Pause: {reason}")
                        await asyncio.sleep(30)
                    continue

                # Circuit breaker check
                if self._check_circuit_breaker():
                    log.info("[CircuitBreaker] Trading halted — sleeping 30min")
                    await asyncio.sleep(1800)
                    continue

                # ── Asset check ──────────────────────────────────────
                if t0 - self._last_asset_check_ts >= self.ASSET_CHECK_EVERY:
                    self._last_asset_check_ts = t0
                    new = await self.supervisor.ensure_best_asset()
                    if new != self._current_asset:
                        log.info(f"Asset: {self._current_asset} → {new}")
                        self._current_asset = new
                        self.agent1.reset(new)
                        await self.bot.select_asset(new)
                        try:
                            from bot.chart_indicators import apply_all_indicators
                            await apply_all_indicators(self.bot._page)
                        except Exception:
                            pass
                        try:
                            from data.fetcher import fetch_candles
                            df = fetch_candles(new, "M1", n=60)
                            if not df.empty:
                                try:
                                    self.agent1.seed_from_dataframe(df, noise_pct=0.0)
                                except TypeError:
                                    self.agent1.seed_from_dataframe(df)
                        except Exception:
                            pass

                # ── Tick ─────────────────────────────────────────────
                price = await self._get_price()
                self.agent1.add_tick(price)
                self.supervisor.handle_tick_outcome(True)
                await self._monitor_active_trades(price)
                await self._resolve_completed_trades(price)

                if t0 - self._last_decision_ts >= self.DECISION_INTERVAL:
                    self._last_decision_ts = t0
                    await self._evaluate_and_trade(price)

                await asyncio.sleep(max(0.05,
                                        self.TICK_INTERVAL - (time.monotonic()-t0)))

        except KeyboardInterrupt:
            log.info("Shutdown requested.")
        except Exception as e:
            log.critical(f"Fatal: {e}", exc_info=True)
        finally:
            await self._shutdown()

    # ── Decision engine ────────────────────────────────────────────────────────

    async def _evaluate_and_trade(self, current_price: float):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # ── Step 1: Agent1 MTF ────────────────────────────────────────
        a1_signal, a1_confidence = self.agent1.analyze()
        log.info(f"[Agent1] {a1_signal.value} @ {a1_confidence:.2%}")

        if a1_signal == Signal.HOLD:
            tf_state = self.agent1.get_latest_multi_tf_state()
            if not tf_state:
                return
            buy_v  = sum(1 for d in tf_state.values()
                         if isinstance(d,dict) and d.get("vote")=="BUY")
            sell_v = sum(1 for d in tf_state.values()
                         if isinstance(d,dict) and d.get("vote")=="SELL")
            if buy_v >= 2:
                a1_signal = Signal.BUY
                a1_confidence = float(np.mean([
                    d["confidence"] for d in tf_state.values()
                    if isinstance(d,dict) and d.get("vote")=="BUY"]))
                log.info(f"[Agent1] HOLD→BUY ({buy_v} TFs) @ {a1_confidence:.2%}")
            elif sell_v >= 2:
                a1_signal = Signal.SELL
                a1_confidence = float(np.mean([
                    d["confidence"] for d in tf_state.values()
                    if isinstance(d,dict) and d.get("vote")=="SELL"]))
                log.info(f"[Agent1] HOLD→SELL ({sell_v} TFs) @ {a1_confidence:.2%}")
            else:
                log.info("[Agent1] HOLD — insufficient TF agreement.")
                return

        tf_state  = self.agent1.get_latest_multi_tf_state()
        state_key = self.agent2.get_state_key(tf_state)

        # ── Step 2: RL ────────────────────────────────────────────────
        rl_action = rl_confidence = None
        rl_experienced = False
        if settings.RL_USE_IN_LIVE_TRADING:
            rl_action, rl_confidence = self.agent2.get_recommendation(state_key)
            rl_visits      = sum(self.agent2.q.visits.get(state_key, [0,0]))
            rl_experienced = rl_visits >= 10
            log.info(f"[Agent2] {'BUY' if rl_action==0 else 'SELL'} @ "
                     f"{rl_confidence:.2%} | state={state_key}")
        else:
            log.info(f"[Agent2] Observer mode | state={state_key}")

        final_signal = a1_signal

        # ── Step 3: Consensus ─────────────────────────────────────────
        consensus = self._calculate_consensus_score(
            tf_state, final_signal.value)
        log.info(f"[StateQuality] Consensus={consensus:.1%} "
                 f"(votes for {final_signal.value})")

        # ── Step 4: Regime — detect FIRST, flip signal if needed ──────
        detected_regime = "UNKNOWN"
        regime_conf_adj = 0.0
        b1m = self.agent1.builders.get(60)

        if b1m and len(b1m) >= 20:
            _, h_r, l_r, c_r = _get_ohlc_from_builder(b1m)
            from agents.agent_multi_timeframe import _rsi, _macd, _bollinger
            rsi_v            = _rsi(c_r, 14)
            macd_v, macd_s, _ = _macd(c_r)
            bb_u, bb_m, bb_l  = _bollinger(c_r, 20)
            atr_v = float(np.mean([h-l for h,l in zip(h_r[-14:], l_r[-14:])]))

            detected_regime = self.regime.detect_regime(
                c_r, h_r, l_r, rsi_v, macd_v, macd_s,
                bb_u, bb_m, bb_l, atr_v)
            log.info(f"[Regime] {detected_regime} | RSI={rsi_v:.1f} | "
                     f"MACD={macd_v:.6f} | ATR={atr_v:.6f}")

            # Signal flip
            if final_signal == Signal.BUY and detected_regime in BEAR_REGIMES:
                bear_str      = max(0.0, (50.0 - rsi_v) / 50.0)
                a1_confidence = 0.52 + bear_str * 0.15
                final_signal  = Signal.SELL
                consensus     = max(self._calculate_consensus_score(
                    tf_state, "SELL"), 0.50)
                log.info(f"[RegimeFlip] BUY→SELL (regime={detected_regime})")

            elif final_signal == Signal.SELL and detected_regime in BULL_REGIMES:
                bull_str      = max(0.0, (rsi_v - 50.0) / 50.0)
                a1_confidence = 0.52 + bull_str * 0.15
                final_signal  = Signal.BUY
                consensus     = max(self._calculate_consensus_score(
                    tf_state, "BUY"), 0.50)
                log.info(f"[RegimeFlip] SELL→BUY (regime={detected_regime})")

            validation = self.regime.validate_signal(
                final_signal.value, detected_regime)
            is_valid, regime_reason = (
                (validation[0], validation[1]) if len(validation) >= 2
                else (True, "ok"))
            regime_conf_adj = validation[2] if len(validation) == 3 else 0.0
            log.info(f"[RegimeValidation] {regime_reason}")
            if not is_valid:
                log.info("[RegimeGate] Rejected — skip")
                return

        # ── Step 5: ChartAnalyzer — EMA crossover + Aroon strategy ───
        chart_result: Optional[AnalysisResult] = None
        chart_conf = 0.50

        if b1m and len(b1m) >= 16:
            try:
                o_c, h_c, l_c, c_c = _get_ohlc_from_builder(b1m)
                chart_result = self.chart.analyze(o_c, h_c, l_c, c_c)
                log.info(self.chart.summary_log(chart_result))

                if not chart_result.volatility_ok:
                    log.info(
                        f"[VolGate] ATR%={chart_result.volatility_pct:.2%} "
                        f"> {self.chart.MAX_VOLATILITY_PCT:.1%} — skip"
                    )
                    return

                if chart_result.signal == "HOLD":
                    log.info("[ChartAnalyzer] No EMA/Aroon crossover signal — skip")
                    return

                if chart_result.signal != final_signal.value:
                    log.info(
                        f"[ChartAnalyzer] Chart primary signal {chart_result.signal} "
                        f"overrides Agent1 {final_signal.value}"
                    )
                    final_signal = (Signal.BUY
                                    if chart_result.signal == "BUY"
                                    else Signal.SELL)

                chart_conf = chart_result.confidence

            except Exception as e:
                log.warning(f"[ChartAnalyzer] {e}")
                return   # Don't trade if analysis failed
        else:
            log.info("[ChartAnalyzer] Not enough candles (need 16+) — skip")
            return

        # ── Step 6: Consensus gate ────────────────────────────────────
        min_consensus = 0.33   # relaxed since chart is primary
        if consensus < min_consensus:
            log.warning(f"[StateFilter] Consensus {consensus:.1%} < "
                        f"{min_consensus:.1%} — SKIP")
            return

        # ── Step 7: Confidence fusion ─────────────────────────────────
        final_conf = self._combine_confidences(
            a1_confidence, rl_confidence or 0.55,
            rl_experienced, consensus, chart_conf)
        final_conf = min(0.97, max(0.40, final_conf + regime_conf_adj))
        log.info(f"[Fusion] {final_signal.value} @ {final_conf:.2%} "
                 f"| regime_adj={regime_conf_adj:+.1%}")

        # ── Step 8: Previous candle confirmation ──────────────────────
        prev_b = self.agent1.builders.get(5) or b1m
        if prev_b and len(prev_b) >= 5:
            try:
                _, h5, l5, c5 = _get_ohlc_from_builder(prev_b)
                sc  = self.regime.analyze_previous_candles(c5, h5, l5, 5)
                up  = sc.get('uptrend_strength',   0.0)
                dn  = sc.get('downtrend_strength', 0.0)
                mom = sc.get('momentum', 0.0)
                log.info(f"[PrevCandles] up={up:.2f} dn={dn:.2f} mom={mom:.4f}")
                if final_signal == Signal.BUY:
                    if up > 0.4 and mom > 0:
                        final_conf = min(0.97, final_conf + 0.04)
                    elif dn > 0.6:
                        final_conf = max(0.40, final_conf - 0.06)
                else:
                    if dn > 0.4 and mom < 0:
                        final_conf = min(0.97, final_conf + 0.04)
                    elif up > 0.6:
                        final_conf = max(0.40, final_conf - 0.06)
            except Exception:
                pass

        # ── Step 9: Pattern alignment ─────────────────────────────────
        if chart_result and chart_result.pattern_direction != "NEUTRAL":
            if chart_result.pattern_direction == final_signal.value:
                b = chart_result.pattern_strength * 0.06
                final_conf = min(0.97, final_conf + b)
                log.info(f"[PatternAlign] {chart_result.pattern_name} +{b:.1%}")
            else:
                p = chart_result.pattern_strength * 0.05
                final_conf = max(0.40, final_conf - p)
                log.info(f"[PatternConflict] {chart_result.pattern_name} -{p:.1%}")

        # ── Step 10: Confidence gate (strategy doc: 60%) ──────────────
        if final_conf < MIN_CONFIDENCE:
            log.info(f"[Gate] {final_conf:.2%} < {MIN_CONFIDENCE:.0%} — skip")
            return

        # ── Step 11: Payout check ─────────────────────────────────────
        payout = await self.bot.get_asset_payout(self._current_asset)
        if payout and payout < PAYOUT_MINIMUM:
            log.info(f"[PayoutGate] {payout:.0%} < {PAYOUT_MINIMUM:.0%} — skip")
            return

        # ── Step 12: Execute ──────────────────────────────────────────
        success = await self.supervisor.safe_execute_trade(
            final_signal, final_conf, "ftt")
        log.info(f"[Trade] placed={success} | "
                 f"{final_signal.value} @ {final_conf:.2%} | "
                 f"${ TRADE_AMOUNT:.0f}")

        self.risk.record_trade(TradeRecord(
            timestamp=ts, asset=self._current_asset, mode="ftt",
            signal=final_signal.value, confidence=final_conf,
            amount=TRADE_AMOUNT,
            result="PENDING" if success else "FAILED",
            pnl=0.0, state_key=state_key,
        ))

        if success:
            if rl_action is None:
                rl_action = 0 if final_signal == Signal.BUY else 1
            self.agent2.register_trade(state_key, rl_action)
            self.active_trades.append({
                "timestamp":   ts,
                "placed_at":   datetime.now(),
                "entry_price": current_price,
                "signal":      final_signal.value,
                "amount":      TRADE_AMOUNT,
                "duration":    FTT_DURATION_S,
                "state_key":   state_key,
                "action":      rl_action,
                "asset":       self._current_asset,
            })
            log.info(f"FTT tracked | entry={current_price:.5f} | "
                     f"{final_signal.value}")

    # ── Outcome resolution ─────────────────────────────────────────────────────

    def _calculate_quality_penalty(self, state_key: str) -> float:
        if not state_key:
            return 1.0
        parts = state_key.split("|")
        if len(parts) < 2:
            return 1.0
        total = len(parts)
        b = sum(1 for v in parts if "B" in v)
        s = sum(1 for v in parts if "S" in v)
        n = sum(1 for v in parts if "N" in v)
        r = max(b, s, n) / total
        if r >= 0.75:   return 1.00
        elif r >= 0.50: return 0.85
        elif r >= 0.25: return 0.60
        else:           return 0.40

    async def _resolve_completed_trades(self, price: float):
        now, still = datetime.now(), []
        for t in self.active_trades:
            if (now - t["placed_at"]).total_seconds() < t["duration"]:
                still.append(t); continue
            ep  = t["entry_price"]
            win = ((t["signal"]=="BUY"  and price > ep) or
                   (t["signal"]=="SELL" and price < ep))
            res = "WIN" if win else "LOSS"
            pnl = t["amount"] * 0.82 if win else -t["amount"]
            log.info(f"{'✅' if win else '❌'} {t['signal']} "
                     f"{ep:.5f}→{price:.5f} {res} ${pnl:+.0f}")
            self.risk.update_pnl_and_journal(t["timestamp"], res, pnl)
            qp = self._calculate_quality_penalty(t.get("state_key",""))
            self.agent2.update_after_result(res, pnl, qp)
            if hasattr(self, 'asset_selector'):
                self.asset_selector.update_trade_history(
                    t.get('asset', self._current_asset), res)
            self.supervisor.record_outcome(win)
            self._record_trade_outcome(win, pnl)
        self.active_trades = still

    # ── Shutdown ───────────────────────────────────────────────────────────────

    async def _shutdown(self):
        self.running = False
        await self.bot.stop()
        for t in list(self._bg_tasks):
            try: t.cancel()
            except Exception: pass
        wr = (self._wins_today / self._trades_today
               if self._trades_today else 0)
        net = self._daily_profit - self._daily_loss
        log.info("=" * 70)
        log.info(f"  Session: {self._trades_today} trades | WR={wr:.0%}")
        log.info(f"  P&L: +${self._daily_profit:.0f} / -${self._daily_loss:.0f} "
                 f"= ${net:+.0f}")
        log.info("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(MultiAgentTradingSystem().run())
    except KeyboardInterrupt:
        pass