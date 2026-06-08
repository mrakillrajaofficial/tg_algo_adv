"""
data/fetcher.py  — v4
Key changes:
  - cache TTL reduced to 5s so yfinance prices update frequently
  - price-scale sanity check: if yfinance returns wrong-scale data
    (e.g., EUR/USD at 300 instead of 1.08) fall back to synthetic
"""

import logging
import time
from typing import Dict

import numpy as np
import pandas as pd

log = logging.getLogger("data.fetcher")

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    log.warning("yfinance not installed — using synthetic data")
    _YF_AVAILABLE = False

_TF_TO_YF_INTERVAL = {
    "M1": "1m", "M5": "5m", "M15": "15m", "M30": "30m",
    "H1": "1h", "H4": "4h", "D1":  "1d",
    5: "1m", 15: "1m", 30: "1m", 60: "1m",
}
_TF_TO_YF_PERIOD = {
    "1m": "1d", "5m": "5d", "15m": "60d",
    "30m": "60d", "1h": "730d", "4h": "730d", "1d": "5y",
}
_SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",   "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",   "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",   "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",   "BTCUSD": "BTC-USD",
    "ETHUSD": "ETH-USD",    "LTCUSD": "LTC-USD",
    "EURUSD-OTC": "EURUSD=X", "EUR/USD OTC": "EURUSD=X",
    "GBPUSD-OTC": "GBPUSD=X", "GBP/USD OTC": "GBPUSD=X",
    "USDJPY-OTC": "USDJPY=X", "AUDUSD-OTC": "AUDUSD=X",
    "EURGBP-OTC": "EURGBP=X", "USDCAD-OTC": "USDCAD=X",
    "NZDUSD-OTC": "NZDUSD=X",
}

# ── Cache: 5-second TTL so indicators see real movement ──────────
_cache: Dict[str, tuple] = {}
_CACHE_TTL = 5.0          # was 30 — this was the root cause of flat candles

# ── Price-scale bounds per asset class ───────────────────────────
# Format: (min_price, max_price) for a sane close value.
# If yfinance returns data outside these bounds, it's the wrong
# instrument or raw pip data — fall back to synthetic.
_PRICE_BOUNDS = {
    "EURUSD": (0.80, 1.60),  "GBPUSD": (1.00, 2.00),
    "AUDUSD": (0.50, 1.10),  "NZDUSD": (0.40, 1.00),
    "USDCAD": (0.90, 1.80),  "EURGBP": (0.70, 1.20),
    "USDJPY": (80.0, 180.0), "BTCUSD": (1000., 200000.),
    "ETHUSD": (50.0, 20000.), "LTCUSD": (5.0, 5000.),
}
# Fallback bounds for unrecognised assets (generous)
_DEFAULT_PRICE_BOUNDS = (0.0001, 100_000.0)


def _price_bounds(asset: str):
    key = asset.upper().replace("-OTC", "").replace("/", "").replace(" ", "")
    return _PRICE_BOUNDS.get(key, _DEFAULT_PRICE_BOUNDS)


def _yf_symbol(asset: str) -> str:
    return _SYMBOL_MAP.get(asset, _SYMBOL_MAP.get(asset.upper(), asset + "=X"))


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema9"]  = c.ewm(span=9,  adjust=False).mean()
    df["ema21"] = c.ewm(span=21, adjust=False).mean()
    df["ema50"] = c.ewm(span=50, adjust=False).mean()

    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - 100 / (1 + rs)

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]

    roll20 = c.rolling(20)
    df["bb_mid"]   = roll20.mean()
    df["bb_upper"] = df["bb_mid"] + 2 * roll20.std()
    df["bb_lower"] = df["bb_mid"] - 2 * roll20.std()

    lo14 = df["low"].rolling(14).min()
    hi14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (c - lo14) / (hi14 - lo14 + 1e-12)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    prev_c = c.shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.rolling(14).mean()
    return df


def fetch_candles(asset: str, timeframe: str = "M1", n: int = 200) -> pd.DataFrame:
    """
    Fetch n candles for asset at timeframe.
    Now uses 5-second cache TTL (was 30s) so live indicators see real movement.
    """
    cache_key = f"{asset}_{timeframe}_{n}"
    cached = _cache.get(cache_key)
    if cached and (time.time() - cached[1]) < _CACHE_TTL:
        return cached[0]

    yf_interval = _TF_TO_YF_INTERVAL.get(timeframe, "1m")
    yf_period   = _TF_TO_YF_PERIOD.get(yf_interval, "1d")
    symbol      = _yf_symbol(asset)

    if _YF_AVAILABLE:
        try:
            raw = yf.Ticker(symbol).history(period=yf_period, interval=yf_interval)
            if raw.empty:
                log.error(f"[Fetcher] yfinance returned empty DataFrame for {symbol} — HALTING instead of synthetic fallback")
                raise ValueError(f"Empty DataFrame from yfinance for {symbol}")
            else:
                raw = raw.rename(columns={
                    "Open": "open", "High": "high",
                    "Low": "low", "Close": "close", "Volume": "volume",
                })[["open", "high", "low", "close", "volume"]].tail(n)
                # ── Price-scale sanity check ──────────────────────────
                # Detect wrong-instrument or un-normalised tick data
                # (e.g. EUR/USD returning 300 instead of 1.08).
                lo_bound, hi_bound = _price_bounds(asset)
                median_close = float(raw["close"].median())
                if not (lo_bound <= median_close <= hi_bound):
                    log.error(
                        f"[Fetcher] PRICE SCALE ERROR for {asset} ({symbol}): "
                        f"median_close={median_close:.4f} is outside "
                        f"expected range [{lo_bound}, {hi_bound}]. "
                        f"HALTING instead of synthetic fallback."
                    )
                    raise ValueError(f"Price scale error for {asset}: {median_close} not in [{lo_bound}, {hi_bound}]")
        except Exception as e:
            log.error(f"yfinance error for {symbol}: {e}")
            raise
    else:
        log.error("yfinance not available — HALTING instead of synthetic fallback.")
        raise RuntimeError("yfinance not available")

    raw = raw.dropna(subset=["close"])
    raw = _compute_indicators(raw)
    _cache[cache_key] = (raw, time.time())
    return raw


def fetch_multi_timeframe(asset: str, timeframes=None) -> Dict[str, pd.DataFrame]:
    if timeframes is None:
        timeframes = ["M1", "M5", "M15", "H1"]
    return {tf: fetch_candles(asset, tf, n=200) for tf in timeframes}


def _synthetic(n: int) -> pd.DataFrame:
    base   = 1.1000
    closes = [base]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1 + np.random.normal(0, 0.0005)))
    closes = np.array(closes)
    noise  = np.abs(np.random.normal(0, 0.0002, n))
    return pd.DataFrame({
        "open":   closes * (1 - noise / 2),
        "high":   closes * (1 + noise),
        "low":    closes * (1 - noise),
        "close":  closes,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })