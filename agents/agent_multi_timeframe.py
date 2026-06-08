"""
agents/agent_multi_timeframe.py  — v4
Key fixes:
  • Removed TF_5S, TF_10S, TF_15S — OlympTrade doesn't provide these
  • Only use TF_30S, TF_1M, TF_5M for stable candle data
  • Faster consensus: 2/3 TFs agreement to trade
"""

import logging
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

from models.predictor import Signal

log = logging.getLogger("agent1_mtf")

TF_30S  = 30
TF_1M   = 60
TF_5M   = 300
ALL_TIMEFRAMES = [TF_30S, TF_1M, TF_5M]

MIN_CANDLES = {
    TF_30S: 5,
    TF_1M:  4,
    TF_5M:  3,
}
SR_HISTORY_SIZE = 60

PREFERRED_ASSETS_ORDERED = [
    "AUDUSD-OTC", "EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC",
    "EURGBP-OTC", "USDCAD-OTC", "NZDUSD-OTC",
    "BTCUSD", "ETHUSD", "LTCUSD",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
]


# ─── Candle Builder ───────────────────────────────────────────────
class CandleBuilder:
    def __init__(self, period: int, max_candles: int = 500):
        self.period   = period
        self.candles: deque = deque(maxlen=max_candles)
        self._current: Optional[dict] = None
        self._bar_end: float = 0.0

    def add_tick(self, price: float, ts: float) -> bool:
        closed = False
        if self._current is None or ts >= self._bar_end:
            if self._current is not None:
                self.candles.append(self._current)
                closed = True
            bar_start    = (ts // self.period) * self.period
            self._bar_end = bar_start + self.period
            self._current = {"open": price, "high": price,
                              "low": price, "close": price, "ts": bar_start}
        else:
            self._current["high"]  = max(self._current["high"], price)
            self._current["low"]   = min(self._current["low"],  price)
            self._current["close"] = price
        return closed

    def get_closes(self) -> np.ndarray:
        arr = [c["close"] for c in self.candles]
        if self._current:
            arr.append(self._current["close"])
        return np.array(arr, dtype=float)

    def get_highs(self) -> np.ndarray:
        arr = [c["high"] for c in self.candles]
        if self._current:
            arr.append(self._current["high"])
        return np.array(arr, dtype=float)

    def get_lows(self) -> np.ndarray:
        arr = [c["low"] for c in self.candles]
        if self._current:
            arr.append(self._current["low"])
        return np.array(arr, dtype=float)

    def __len__(self):
        return len(self.candles) + (1 if self._current else 0)


# ─── Indicators ───────────────────────────────────────────────────
def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full_like(arr, np.nan)
    k, v = 2.0 / (p + 1), arr[0]
    for i, x in enumerate(arr):
        v = x * k + v * (1 - k); out[i] = v
    return out

def _rsi(c: np.ndarray, p: int = 14) -> float:
    if len(c) < p + 1: return 50.0
    d = np.diff(c[-(p + 1):])
    g = d[d > 0].mean() if (d > 0).any() else 1e-9
    l = -d[d < 0].mean() if (d < 0).any() else 1e-9
    return 100 - 100 / (1 + g / l)

def _macd(c: np.ndarray) -> Tuple[float, float, float]:
    if len(c) < 26:
        return 0.0, 0.0, 0.0
    m = _ema(c, 12) - _ema(c, 26)
    s = _ema(m[~np.isnan(m)], 9)
    macd_v = float(m[-1])
    macd_s = float(s[-1])
    hist = macd_v - macd_s
    return macd_v, macd_s, float(hist)

def _bollinger(c: np.ndarray, p: int = 20) -> Tuple[float, float, float]:
    if len(c) < p: m = c[-1]; return m, m, m
    w = c[-p:]; mid = w.mean(); std = w.std()
    return mid + 2 * std, mid, mid - 2 * std

def _stoch(h: np.ndarray, l: np.ndarray, c: np.ndarray, p: int = 14) -> float:
    if len(c) < p: return 50.0
    hi, lo = h[-p:].max(), l[-p:].min()
    return 50.0 if hi == lo else 100 * (c[-1] - lo) / (hi - lo)

def _ema_cross(c: np.ndarray) -> int:
    if len(c) < 21: return 0
    f, s = _ema(c, 9), _ema(c, 21)
    return 1 if f[-1] > s[-1] else (-1 if f[-1] < s[-1] else 0)


# ─── Support / Resistance ─────────────────────────────────────────
class SREngine:
    def __init__(self, tol: float = 0.0005):
        self.tol = tol
        self.supports: List[float]    = []
        self.resistances: List[float] = []

    def update(self, h: np.ndarray, l: np.ndarray, c: np.ndarray):
        if len(c) < 5: return
        ph, pl = [], []
        for i in range(2, len(c) - 2):
            if h[i] > h[i-1] and h[i] > h[i-2] and h[i] > h[i+1] and h[i] > h[i+2]:
                ph.append(h[i])
            if l[i] < l[i-1] and l[i] < l[i-2] and l[i] < l[i+1] and l[i] < l[i+2]:
                pl.append(l[i])
        self.resistances = self._cluster(ph)
        self.supports    = self._cluster(pl)

    def _cluster(self, pts: list) -> List[float]:
        if not pts: return []
        pts = sorted(pts); clusters, grp = [], [pts[0]]
        for p in pts[1:]:
            if abs(p - grp[-1]) / max(grp[-1], 1e-9) < self.tol:
                grp.append(p)
            else:
                clusters.append(float(np.mean(grp))); grp = [p]
        clusters.append(float(np.mean(grp)))
        return clusters

    def zone_bias(self, price: float) -> str:
        all_z = [(s, "support") for s in self.supports] + \
                [(r, "resistance") for r in self.resistances]
        if not all_z: return "neutral"
        closest = min(all_z, key=lambda z: abs(z[0] - price))
        if abs(closest[0] - price) / max(closest[0], 1e-9) > 0.003:
            return "neutral"
        return "bullish" if closest[1] == "support" else "bearish"


# ─── Main Agent ───────────────────────────────────────────────────
class MultiTimeframeAgent:
    def __init__(self, asset: str):
        self.asset = asset
        self.reset(asset)

    def reset(self, asset: str = None):
        if asset is not None:
            self.asset = asset
        self.builders: Dict[int, CandleBuilder] = {
            TF_30S: CandleBuilder(TF_30S, 200),
            TF_1M:  CandleBuilder(TF_1M,  70),
            TF_5M:  CandleBuilder(TF_5M,  40),
        }
        self.sr_engine  = SREngine()
        self._last_price = 0.0
        self._tick_count = 0
        self._last_tick_time = 0.0
        self._last_tf_state: Dict = {}

    def seed_from_dataframe(self, df, noise_pct: float = 0.25):
        if df is None or df.empty:
            return
        df = df.tail(60)
        now = time.time()
        total = len(df)
        for i, row in enumerate(df.itertuples()):
            t_base = now - (total - i) * 60.0
            noise_scale = abs(row.high - row.low) * noise_pct
            for sub in range(60):
                frac  = sub / 60.0
                price = row.open + frac * (row.close - row.open)
                if noise_scale > 0:
                    price += np.random.normal(0, noise_scale)
                ts    = t_base + sub
                for builder in self.builders.values():
                    builder.add_tick(float(price), ts)
        log.info(f"[Agent1] Seeded builders from {total} M1 candles (last 60) with {noise_pct*100:.0f}% noise.")
        b1m = self.builders[TF_1M]
        if len(b1m) >= 5:
            self.sr_engine.update(b1m.get_highs(), b1m.get_lows(), b1m.get_closes())

    def bootstrap(self, base_price: float, n_ticks: int = 200):
        log.info(f"[Agent1] Bootstrapping from price {base_price}")
        now = time.time()
        price = base_price
        for i in range(n_ticks, 0, -1):
            price = price * (1 + np.random.normal(0, 0.0003))
            ts    = now - i * 0.5
            for builder in self.builders.values():
                builder.add_tick(price, ts)
        self._last_price = base_price
        log.info("[Agent1] Bootstrap complete.")

    def add_tick(self, price: float):
        if price <= 0:
            return
        ts = time.time()
        self._last_price = price
        self._last_tick_time = ts
        self._tick_count += 1
        for builder in self.builders.values():
            builder.add_tick(price, ts)
        if self._tick_count % 10 == 0:
            b1m = self.builders[TF_1M]
            if len(b1m) >= 5:
                self.sr_engine.update(b1m.get_highs(), b1m.get_lows(), b1m.get_closes())

    def analyze(self) -> Tuple[Signal, float]:
        if self._last_price <= 0 or (time.time() - getattr(self, '_last_tick_time', 0.0) > 15.0):
            return Signal.HOLD, 0.0

        votes_buy = votes_sell = 0
        total_conf = 0.0
        tf_details = {}
        active_tfs = 0

        for tf in ALL_TIMEFRAMES:
            b = self.builders[tf]
            if len(b) < MIN_CANDLES[tf]:
                log.debug(f"[Agent1] TF={tf}s: {len(b)}/{MIN_CANDLES[tf]} candles")
                continue
            active_tfs += 1
            vote, conf = self._analyze_tf(b.get_closes(), b.get_highs(), b.get_lows(), tf)
            tf_details[tf] = {"vote": vote, "confidence": conf}
            if vote == "BUY":
                votes_buy  += 1; total_conf += conf
            elif vote == "SELL":
                votes_sell += 1; total_conf += conf

        self._last_tf_state = tf_details

        def _emit(direction: str, votes: int, avg_raw: float) -> Tuple["Signal", float]:
            if avg_raw == 0.50:
                return Signal.HOLD, 0.0

            bias = self.sr_engine.zone_bias(self._last_price)
            if direction == "BUY":
                sr_adj = +0.04 if bias == "bullish" else (-0.05 if bias == "bearish" else 0.0)
            else:
                sr_adj = +0.04 if bias == "bearish" else (-0.05 if bias == "bullish" else 0.0)
                
            # Fix 5: Agent1 bias contradiction audit
            if (direction == "BUY" and bias == "bearish") or (direction == "SELL" and bias == "bullish"):
                avg_raw = min(avg_raw, 0.50)
                
            if votes >= 3:
                floor = 0.50
            elif votes == 2:
                floor = 0.45
            else:
                floor = 0.40
            conf = max(floor, min(0.97, avg_raw + sr_adj))

            # --- Bollinger Bands Integration ---
            b1m = self.builders[TF_1M]
            c1m = b1m.get_closes()
            upper, mid, lower = _bollinger(c1m)
            prev_p = c1m[-2] if len(c1m) > 1 else c1m[-1]
            p = self._last_price
            
            bb_state = 'NORMAL'
            if p > upper:
                bb_state = 'ABOVE_UPPER'
            elif p < lower:
                bb_state = 'BELOW_LOWER'
            else:
                br = max(upper - lower, 1e-9)
                if (upper - p) / br < 0.05: bb_state = 'UPPER_TOUCH'
                elif (p - lower) / br < 0.05: bb_state = 'LOWER_TOUCH'
                elif prev_p < mid <= p: bb_state = 'MID_CROSS_UP'
                elif prev_p > mid >= p: bb_state = 'MID_CROSS_DOWN'

            if bb_state == 'UPPER_TOUCH' and direction == 'SELL':
                conf += 0.08
            elif bb_state == 'LOWER_TOUCH' and direction == 'BUY':
                conf += 0.08
            elif bb_state == 'ABOVE_UPPER' and direction == 'SELL':
                conf += 0.12
            elif bb_state == 'BELOW_LOWER' and direction == 'BUY':
                conf += 0.12
            elif bb_state == 'MID_CROSS_UP' and direction == 'BUY':
                conf += 0.04
            elif bb_state == 'MID_CROSS_DOWN' and direction == 'SELL':
                conf += 0.04

            conf = min(0.97, conf)

            sig  = Signal.BUY if direction == "BUY" else Signal.SELL
            log.info(f"[Agent1] {direction} {votes}/{active_tfs} active TFs | raw={avg_raw:.2%} "
                     f"sr={sr_adj:+.2%} floor={floor:.2%} → conf={conf:.2%} | bias={bias} | BB={bb_state}")
            return sig, conf

        if votes_buy >= 2:
            return _emit("BUY",  votes_buy,  total_conf / votes_buy)
        if votes_sell >= 2:
            return _emit("SELL", votes_sell, total_conf / votes_sell)

        if active_tfs >= 1 and (votes_buy >= 1 or votes_sell >= 1):
            if votes_buy >= 1:
                return _emit("BUY",  votes_buy,  total_conf / votes_buy)
            else:
                return _emit("SELL", votes_sell, total_conf / votes_sell)

        log.debug(f"[Agent1] No consensus BUY={votes_buy} SELL={votes_sell} from {active_tfs} active TFs")
        return Signal.HOLD, 0.0

    def get_latest_multi_tf_state(self) -> Dict:
        return self._last_tf_state

    def evaluate_chart_strength(self) -> float:
        if not self._last_tf_state:
            return 0.0
        score = 0.0
        weight = 0.0
        for tf, data in self._last_tf_state.items():
            conf = data.get("confidence", 0.0)
            vote = data.get("vote")
            tf_weight = 1.0
            if tf >= TF_1M:
                tf_weight = 1.25
            elif tf >= TF_30S:
                tf_weight = 1.1
            if vote == "BUY":
                score += conf * tf_weight
            elif vote == "SELL":
                score -= conf * tf_weight
            weight += tf_weight

        if weight == 0:
            return 0.0

        bias = self.sr_engine.zone_bias(self._last_price)
        if bias == "bullish":
            score += 0.1
        elif bias == "bearish":
            score -= 0.1

        return float(score / max(weight, 1.0))

    def _analyze_tf(self, c: np.ndarray, h: np.ndarray,
                    l: np.ndarray, tf: int) -> Tuple[str, float]:
        scores = []

        rsi = _rsi(c)
        if   rsi < 30:  scores.append(+1.0)
        elif rsi > 70:  scores.append(-1.0)
        elif rsi <= 45: scores.append(+0.5)
        elif rsi >= 55: scores.append(-0.5)
        else:           scores.append(0.0)

        ml, sl, _ = _macd(c)
        if   ml > sl and ml > 0: scores.append(+1.0)
        elif ml < sl and ml < 0: scores.append(-1.0)
        elif ml > sl:            scores.append(+0.5)
        elif ml < sl:            scores.append(-0.5)
        else:                    scores.append(0.0)

        up, mid, lo = _bollinger(c)
        p = c[-1]
        if   p <= lo:  scores.append(+1.0)
        elif p >= up:  scores.append(-1.0)
        elif p < mid:  scores.append(+0.3)
        elif p > mid:  scores.append(-0.3)
        else:          scores.append(0.0)

        scores.append(_ema_cross(c) * 0.8)

        sk = _stoch(h, l, c)
        if   sk < 20: scores.append(+1.0)
        elif sk > 80: scores.append(-1.0)
        elif sk < 40: scores.append(+0.3)
        elif sk > 60: scores.append(-0.3)
        else:         scores.append(0.0)

        norm  = sum(scores) / 5.0
        abs_n = abs(norm)
        conf  = 0.45 + abs_n * 0.47

        if   norm >=  0.20: return "BUY",  conf
        elif norm <= -0.20: return "SELL", conf
        return "NEUTRAL", conf