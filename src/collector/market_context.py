"""
Market Context Engine
=====================

Computes market context from HTF klines: trend, regime, volatility.

All indicators implemented from scratch (no external TA libraries).

Indicators:
    - EMA(20, 50) on 5m, 15m
    - EMA(50, 200) on 1h
    - ATR(14) on 15m, 1h
    - RSI(14) on 15m
    - ADX(14) on 15m

Output:
    - trend_5m, trend_15m, trend_1h: UP/DOWN/FLAT
    - trend_alignment_score: 0..100
    - regime_15m: TREND/RANGE
    - atr_15m, atr_1h
    - rsi_15m
    - ema_spread_15m_bps
    - vol_regime: LOW/NORMAL/HIGH

Usage:
    engine = MarketContextEngine(klines_store, "BTCUSDT")
    engine.update()
    ctx = engine.snapshot()
"""

import logging
from dataclasses import dataclass
from typing import Optional

from collector.context_klines_store import Kline, KlinesStore
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


# =============================================================================
# Indicator Helpers (Pure Python implementations)
# =============================================================================

def calc_ema(prices: list[float], period: int) -> list[float]:
    """
    Calculate Exponential Moving Average.
    
    EMA_t = price_t * k + EMA_{t-1} * (1-k)
    where k = 2 / (period + 1)
    
    Args:
        prices: List of prices (oldest first)
        period: EMA period
        
    Returns:
        List of EMA values (same length as prices, first period-1 are warm-up)
    """
    if len(prices) < period:
        return []
    
    k = 2 / (period + 1)
    ema_values = []
    
    # First EMA is SMA of first `period` values
    sma = sum(prices[:period]) / period
    ema_values.extend([sma] * period)  # Warm-up values
    
    ema = sma
    for i in range(period, len(prices)):
        ema = prices[i] * k + ema * (1 - k)
        ema_values.append(ema)
    
    return ema_values


def calc_sma(values: list[float], period: int) -> list[float]:
    """
    Calculate Simple Moving Average.
    
    Args:
        values: List of values (oldest first)
        period: SMA period
        
    Returns:
        List of SMA values (first period-1 are None/NaN conceptually, we use 0)
    """
    if len(values) < period:
        return []
    
    sma_values = [0.0] * (period - 1)
    
    window_sum = sum(values[:period])
    sma_values.append(window_sum / period)
    
    for i in range(period, len(values)):
        window_sum = window_sum - values[i - period] + values[i]
        sma_values.append(window_sum / period)
    
    return sma_values


def calc_true_range(klines: list[Kline]) -> list[float]:
    """
    Calculate True Range series.
    
    TR = max(high - low, |high - prev_close|, |low - prev_close|)
    
    First value uses high - low only (no previous close).
    
    Args:
        klines: List of Kline objects (oldest first)
        
    Returns:
        List of TR values
    """
    if not klines:
        return []
    
    tr_values = []
    
    # First candle: TR = high - low
    tr_values.append(klines[0].high - klines[0].low)
    
    for i in range(1, len(klines)):
        high = klines[i].high
        low = klines[i].low
        prev_close = klines[i - 1].close
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        tr_values.append(tr)
    
    return tr_values


def calc_atr(klines: list[Kline], period: int = 14) -> Optional[float]:
    """
    Calculate Average True Range (ATR).
    
    Uses EMA smoothing of True Range.
    
    Args:
        klines: List of Kline objects (oldest first)
        period: ATR period
        
    Returns:
        Current ATR value or None if insufficient data
    """
    if len(klines) < period + 1:
        return None
    
    tr_values = calc_true_range(klines)
    atr_values = calc_ema(tr_values, period)
    
    if not atr_values:
        return None
    
    return atr_values[-1]


def calc_atr_series(klines: list[Kline], period: int = 14) -> list[float]:
    """
    Calculate ATR series for percentile analysis.
    
    Args:
        klines: List of Kline objects (oldest first)
        period: ATR period
        
    Returns:
        List of ATR values
    """
    if len(klines) < period + 1:
        return []
    
    tr_values = calc_true_range(klines)
    return calc_ema(tr_values, period)


def calc_rsi(prices: list[float], period: int = 14) -> Optional[float]:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI = 100 - 100 / (1 + RS)
    where RS = avg_gain / avg_loss
    
    Uses Wilder's smoothing (similar to EMA with alpha = 1/period).
    
    Args:
        prices: List of close prices (oldest first)
        period: RSI period
        
    Returns:
        Current RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    deltas = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
    
    # Separate gains and losses
    gains = [max(d, 0) for d in deltas]
    losses = [abs(min(d, 0)) for d in deltas]
    
    # First average: SMA of first `period` values
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    
    # Wilder's smoothing for subsequent values
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - 100 / (1 + rs)
    
    return rsi


def calc_adx(klines: list[Kline], period: int = 14) -> Optional[float]:
    """
    Calculate Average Directional Index (ADX).
    
    Steps:
    1. Calculate +DM and -DM (Directional Movement)
    2. Smooth +DM and -DM with Wilder's smoothing
    3. Calculate +DI and -DI as percentage of ATR
    4. Calculate DX = |+DI - -DI| / (+DI + -DI) * 100
    5. Smooth DX with Wilder's smoothing to get ADX
    
    Args:
        klines: List of Kline objects (oldest first)
        period: ADX period
        
    Returns:
        Current ADX value (0-100) or None if insufficient data
    """
    # Need at least 2*period + 1 candles for stable ADX
    if len(klines) < 2 * period + 1:
        return None
    
    # Calculate +DM and -DM
    plus_dm = []
    minus_dm = []
    tr_values = []
    
    for i in range(1, len(klines)):
        high = klines[i].high
        low = klines[i].low
        prev_high = klines[i - 1].high
        prev_low = klines[i - 1].low
        prev_close = klines[i - 1].close
        
        # True Range
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close),
        )
        tr_values.append(tr)
        
        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low
        
        if up_move > down_move and up_move > 0:
            plus_dm.append(up_move)
        else:
            plus_dm.append(0)
        
        if down_move > up_move and down_move > 0:
            minus_dm.append(down_move)
        else:
            minus_dm.append(0)
    
    if len(tr_values) < period:
        return None
    
    # Wilder's smoothing (similar to EMA with alpha = 1/period)
    def wilder_smooth(values: list[float], p: int) -> list[float]:
        if len(values) < p:
            return []
        
        # First value is sum of first p values
        smoothed = [sum(values[:p])]
        
        for i in range(p, len(values)):
            prev = smoothed[-1]
            new_val = prev - prev / p + values[i]
            smoothed.append(new_val)
        
        return smoothed
    
    # Smooth TR, +DM, -DM
    smoothed_tr = wilder_smooth(tr_values, period)
    smoothed_plus_dm = wilder_smooth(plus_dm, period)
    smoothed_minus_dm = wilder_smooth(minus_dm, period)
    
    if not smoothed_tr or smoothed_tr[-1] == 0:
        return None
    
    # Calculate +DI and -DI series
    dx_values = []
    
    min_len = min(len(smoothed_tr), len(smoothed_plus_dm), len(smoothed_minus_dm))
    
    for i in range(min_len):
        if smoothed_tr[i] == 0:
            continue
        
        plus_di = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
        minus_di = 100 * smoothed_minus_dm[i] / smoothed_tr[i]
        
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0)
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
            dx_values.append(dx)
    
    if len(dx_values) < period:
        return None
    
    # Smooth DX to get ADX
    adx_values = wilder_smooth(dx_values, period)
    
    if not adx_values:
        return None
    
    # ADX is the smoothed value divided by period (since we accumulated)
    adx = adx_values[-1] / period
    
    return min(adx, 100.0)  # Cap at 100


def calc_percentile(values: list[float], percentile: float) -> float:
    """
    Calculate percentile of a list of values.
    
    Args:
        values: List of values
        percentile: Percentile (0-100)
        
    Returns:
        Value at the given percentile
    """
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if n == 1:
        return sorted_values[0]
    
    # Linear interpolation
    k = (percentile / 100) * (n - 1)
    f = int(k)
    c = f + 1
    
    if c >= n:
        return sorted_values[-1]
    
    d = k - f
    return sorted_values[f] * (1 - d) + sorted_values[c] * d


# =============================================================================
# Trend Detection
# =============================================================================

def determine_trend(
    ema_fast: Optional[float],
    ema_slow: Optional[float],
    threshold_bps: float = 10.0,
) -> str:
    """
    Determine trend based on EMA crossover.
    
    Args:
        ema_fast: Fast EMA value
        ema_slow: Slow EMA value
        threshold_bps: Minimum spread in basis points for UP/DOWN
        
    Returns:
        "UP", "DOWN", or "FLAT"
    """
    if ema_fast is None or ema_slow is None or ema_slow == 0:
        return "FLAT"
    
    spread_bps = (ema_fast - ema_slow) / ema_slow * 10000
    
    if spread_bps > threshold_bps:
        return "UP"
    elif spread_bps < -threshold_bps:
        return "DOWN"
    else:
        return "FLAT"


# =============================================================================
# Market Context Engine
# =============================================================================

@dataclass
class MarketContext:
    """Market context snapshot."""
    
    # Trends
    trend_5m: str = "FLAT"
    trend_15m: str = "FLAT"
    trend_1h: str = "FLAT"
    trend_alignment_score: int = 0
    
    # Regime
    regime_15m: str = "RANGE"
    adx_15m: Optional[float] = None
    
    # Volatility
    atr_15m: Optional[float] = None
    atr_1h: Optional[float] = None
    vol_regime: str = "NORMAL"
    
    # Momentum
    rsi_15m: Optional[float] = None
    
    # Spreads
    ema_spread_15m_bps: Optional[float] = None
    
    # Metadata
    last_update_ms: int = 0
    data_quality: str = "OK"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "trend_5m": self.trend_5m,
            "trend_15m": self.trend_15m,
            "trend_1h": self.trend_1h,
            "trend_alignment_score": self.trend_alignment_score,
            "regime_15m": self.regime_15m,
            "adx_15m": round(self.adx_15m, 2) if self.adx_15m else None,
            "atr_15m": round(self.atr_15m, 2) if self.atr_15m else None,
            "atr_1h": round(self.atr_1h, 2) if self.atr_1h else None,
            "vol_regime": self.vol_regime,
            "rsi_15m": round(self.rsi_15m, 2) if self.rsi_15m else None,
            "ema_spread_15m_bps": round(self.ema_spread_15m_bps, 2) if self.ema_spread_15m_bps else None,
            "last_update_ms": self.last_update_ms,
            "data_quality": self.data_quality,
        }


class MarketContextEngine:
    """
    Engine that computes market context from HTF klines.
    
    Pulls klines from KlinesStore and computes various indicators
    to determine trend, regime, and volatility state.
    
    Args:
        store: KlinesStore with klines data
        symbol: Trading pair symbol
        market: Market to analyze ("futures" or "spot")
    """
    
    # Minimum bars required for each indicator
    MIN_BARS_EMA_SHORT = 50    # For EMA(20, 50)
    MIN_BARS_EMA_LONG = 200    # For EMA(50, 200)
    MIN_BARS_ATR = 30          # For ATR(14)
    MIN_BARS_RSI = 30          # For RSI(14)
    MIN_BARS_ADX = 50          # For ADX(14)
    MIN_BARS_VOL_REGIME = 200  # For volatility percentile analysis
    
    def __init__(
        self,
        store: KlinesStore,
        symbol: str,
        market: str = "futures",
    ) -> None:
        self._store = store
        self._symbol = symbol
        self._market = market
        
        self._context = MarketContext()
        self._update_count = 0
        
        logger.info(
            "market_context_engine_initialized",
            extra={
                "symbol": symbol,
                "market": market,
            },
        )
    
    def _get_closes(self, tf: str, limit: Optional[int] = None) -> list[float]:
        """Get close prices from store."""
        klines = self._store.get(self._market, tf)
        if limit:
            klines = klines[-limit:]
        return [k.close for k in klines]
    
    def _get_klines(self, tf: str, limit: Optional[int] = None) -> list[Kline]:
        """Get klines from store."""
        klines = self._store.get(self._market, tf)
        if limit:
            klines = klines[-limit:]
        return klines
    
    def _calc_ema_pair(
        self,
        tf: str,
        fast_period: int,
        slow_period: int,
    ) -> tuple[Optional[float], Optional[float]]:
        """Calculate fast and slow EMA for a timeframe."""
        closes = self._get_closes(tf)
        
        if len(closes) < slow_period:
            return None, None
        
        ema_fast = calc_ema(closes, fast_period)
        ema_slow = calc_ema(closes, slow_period)
        
        if not ema_fast or not ema_slow:
            return None, None
        
        return ema_fast[-1], ema_slow[-1]
    
    def update(self) -> None:
        """
        Update market context by computing all indicators.
        
        Pulls latest klines from store and recomputes indicators.
        """
        issues = []
        
        # ===== 5m Trend (EMA20 vs EMA50) =====
        ema20_5m, ema50_5m = self._calc_ema_pair("5m", 20, 50)
        if ema20_5m and ema50_5m:
            self._context.trend_5m = determine_trend(ema20_5m, ema50_5m, threshold_bps=10)
        else:
            self._context.trend_5m = "FLAT"
            issues.append("5m_ema_insufficient")
        
        # ===== 15m Trend (EMA20 vs EMA50) =====
        ema20_15m, ema50_15m = self._calc_ema_pair("15m", 20, 50)
        if ema20_15m and ema50_15m:
            self._context.trend_15m = determine_trend(ema20_15m, ema50_15m, threshold_bps=10)
            # Also calculate spread
            if ema50_15m != 0:
                self._context.ema_spread_15m_bps = (ema20_15m - ema50_15m) / ema50_15m * 10000
        else:
            self._context.trend_15m = "FLAT"
            self._context.ema_spread_15m_bps = None
            issues.append("15m_ema_insufficient")
        
        # ===== 1h Trend (EMA50 vs EMA200) =====
        ema50_1h, ema200_1h = self._calc_ema_pair("1h", 50, 200)
        if ema50_1h and ema200_1h:
            self._context.trend_1h = determine_trend(ema50_1h, ema200_1h, threshold_bps=10)
        else:
            self._context.trend_1h = "FLAT"
            issues.append("1h_ema_insufficient")
        
        # ===== Trend Alignment Score =====
        self._context.trend_alignment_score = self._calc_alignment_score()
        
        # ===== ATR 15m =====
        klines_15m = self._get_klines("15m")
        if len(klines_15m) >= self.MIN_BARS_ATR:
            self._context.atr_15m = calc_atr(klines_15m, 14)
        else:
            self._context.atr_15m = None
            issues.append("15m_atr_insufficient")
        
        # ===== ATR 1h =====
        klines_1h = self._get_klines("1h")
        if len(klines_1h) >= self.MIN_BARS_ATR:
            self._context.atr_1h = calc_atr(klines_1h, 14)
        else:
            self._context.atr_1h = None
            issues.append("1h_atr_insufficient")
        
        # ===== RSI 15m =====
        closes_15m = self._get_closes("15m")
        if len(closes_15m) >= self.MIN_BARS_RSI:
            self._context.rsi_15m = calc_rsi(closes_15m, 14)
        else:
            self._context.rsi_15m = None
            issues.append("15m_rsi_insufficient")
        
        # ===== ADX 15m =====
        if len(klines_15m) >= self.MIN_BARS_ADX:
            self._context.adx_15m = calc_adx(klines_15m, 14)
            
            # Determine regime
            if self._context.adx_15m is not None and self._context.adx_15m >= 20:
                self._context.regime_15m = "TREND"
            else:
                self._context.regime_15m = "RANGE"
        else:
            self._context.adx_15m = None
            self._context.regime_15m = "RANGE"
            issues.append("15m_adx_insufficient")
        
        # ===== Volatility Regime =====
        if len(klines_15m) >= self.MIN_BARS_VOL_REGIME:
            atr_series = calc_atr_series(klines_15m, 14)
            if len(atr_series) >= 100:
                current_atr = atr_series[-1]
                p30 = calc_percentile(atr_series[-200:], 30)
                p70 = calc_percentile(atr_series[-200:], 70)
                
                if current_atr < p30:
                    self._context.vol_regime = "LOW"
                elif current_atr > p70:
                    self._context.vol_regime = "HIGH"
                else:
                    self._context.vol_regime = "NORMAL"
        else:
            self._context.vol_regime = "NORMAL"
            issues.append("vol_regime_insufficient")
        
        # ===== Update metadata =====
        self._context.last_update_ms = now_ms()
        self._context.data_quality = "OK" if not issues else "DEGRADED"
        
        self._update_count += 1
        
        # Log periodically (every 12 updates = ~1 min at 5s interval)
        if self._update_count % 12 == 1:
            logger.info(
                "market_context_updated",
                extra={
                    "trend_5m": self._context.trend_5m,
                    "trend_15m": self._context.trend_15m,
                    "trend_1h": self._context.trend_1h,
                    "alignment_score": self._context.trend_alignment_score,
                    "regime_15m": self._context.regime_15m,
                    "adx_15m": round(self._context.adx_15m, 2) if self._context.adx_15m else None,
                    "vol_regime": self._context.vol_regime,
                    "rsi_15m": round(self._context.rsi_15m, 2) if self._context.rsi_15m else None,
                    "data_quality": self._context.data_quality,
                    "issues": issues if issues else None,
                },
            )
    
    def _calc_alignment_score(self) -> int:
        """
        Calculate trend alignment score.
        
        Returns:
            0-100 score:
            - 100: All trends UP or all DOWN
            - 60: 2 of 3 trends aligned
            - 20: All different or all FLAT
        """
        trends = [
            self._context.trend_5m,
            self._context.trend_15m,
            self._context.trend_1h,
        ]
        
        # Count each direction
        up_count = trends.count("UP")
        down_count = trends.count("DOWN")
        flat_count = trends.count("FLAT")
        
        # All aligned in same direction
        if up_count == 3 or down_count == 3:
            return 100
        
        # 2 of 3 aligned
        if up_count == 2 or down_count == 2:
            return 60
        
        # All flat
        if flat_count == 3:
            return 20
        
        # Mixed (1 up, 1 down, 1 flat or similar)
        return 20
    
    def snapshot(self) -> dict:
        """
        Get current market context as dictionary.
        
        Returns:
            Dictionary with all context fields
        """
        return self._context.to_dict()
    
    def get_trend_summary(self) -> str:
        """
        Get short trend summary string.
        
        Returns:
            String like "UP-UP-UP" or "DOWN-FLAT-UP"
        """
        return f"{self._context.trend_5m}-{self._context.trend_15m}-{self._context.trend_1h}"
