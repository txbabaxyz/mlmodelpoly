"""
TAAPI Context Engine
====================

Transforms raw TAAPI indicators into meaningful context fields.

Provides:
    - bias_1h / bias_15m / bias_5m (UP / DOWN / NEUTRAL)
    - regime_15m (TREND / RANGE)
    - micro_stage_1m (ACCEL_UP / ACCEL_DOWN / OVERHEATED / NEUTRAL)
    - vol_regime (HIGH / NORMAL / LOW)
    - alignment_score (0-100)

Usage:
    engine = TaapiContextEngine(store)
    context = engine.snapshot()
"""

import logging
from typing import Optional
from collections import deque

from collector.taapi.store import TaapiContextStore
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


# Thresholds
ADX_TREND_THRESHOLD = 20       # ADX >= 20 means trending
RSI_OVERBOUGHT = 70            # RSI > 70 = overbought
RSI_OVERSOLD = 30              # RSI < 30 = oversold
EMA_CROSS_THRESHOLD_BPS = 5    # 5 bps threshold for EMA cross

# ATR history for volatility regime
ATR_HISTORY_SIZE = 20


class TaapiContextEngine:
    """
    Interprets TAAPI indicators into meaningful context.
    
    Transforms raw indicator values into:
    - Bias (directional tendency)
    - Regime (trending vs ranging)
    - Micro stage (acceleration/deceleration)
    - Volatility regime
    - Alignment score
    
    Args:
        store: TaapiContextStore with indicator values
    """
    
    def __init__(self, store: TaapiContextStore) -> None:
        self._store = store
        
        # ATR history for volatility percentiles
        self._atr_history_15m: deque[float] = deque(maxlen=ATR_HISTORY_SIZE)
        self._atr_history_1h: deque[float] = deque(maxlen=ATR_HISTORY_SIZE)
        
        logger.info("taapi_context_engine_initialized")
    
    def snapshot(self) -> dict:
        """
        Generate context snapshot from current indicators.
        
        Returns:
            Dict with interpreted context fields
        """
        ts = now_ms()
        
        # Get bias per timeframe
        bias_1h = self._compute_bias("1h")
        bias_15m = self._compute_bias("15m")
        bias_5m = self._compute_bias("5m")
        bias_1m = self._compute_bias("1m")
        
        # Get regime
        regime_15m = self._compute_regime("15m")
        regime_5m = self._compute_regime("5m")
        
        # Get micro stage
        micro_stage = self._compute_micro_stage()
        
        # Get volatility regime
        vol_regime = self._compute_vol_regime()
        
        # Get alignment score
        alignment_score = self._compute_alignment_score(
            bias_1h, bias_15m, bias_5m, bias_1m
        )
        
        # Supertrend direction
        supertrend_5m = self._get_supertrend_direction("5m")
        
        # RSI values
        rsi_1m = self._store.get_value("1m", "rsi_14")
        rsi_15m = self._store.get_value("15m", "rsi_14")
        
        # CMF (money flow)
        cmf_15m = self._store.get_value("15m", "cmf_20")
        
        # MACD signals
        macd_1m = self._get_macd_signal("1m")
        macd_15m = self._get_macd_signal("15m")
        macd_1h = self._get_macd_signal("1h")
        
        # Data quality
        data_ages = self._get_data_ages()
        
        return {
            # Bias (directional tendency)
            "bias_1h": bias_1h,
            "bias_15m": bias_15m,
            "bias_5m": bias_5m,
            "bias_1m": bias_1m,
            
            # Regime
            "regime_15m": regime_15m,
            "regime_5m": regime_5m,
            
            # Micro stage
            "micro_stage": micro_stage,
            
            # Volatility
            "vol_regime": vol_regime,
            
            # Alignment
            "alignment_score": alignment_score,
            
            # Additional signals
            "supertrend_5m": supertrend_5m,
            "rsi_1m": round(rsi_1m, 1) if rsi_1m else None,
            "rsi_15m": round(rsi_15m, 1) if rsi_15m else None,
            "cmf_15m": round(cmf_15m, 4) if cmf_15m else None,
            "macd_signal_1m": macd_1m,
            "macd_signal_15m": macd_15m,
            "macd_signal_1h": macd_1h,
            
            # Metadata
            "source": "taapi",
            "last_update_ms": ts,
            "data_ages": data_ages,
        }
    
    def _compute_bias(self, tf: str) -> str:
        """
        Compute directional bias for a timeframe.
        
        Logic:
            - 1h: EMA50 vs EMA200
            - 15m/5m/1m: EMA20 vs EMA50
        
        Args:
            tf: Timeframe
            
        Returns:
            "UP", "DOWN", or "NEUTRAL"
        """
        if tf == "1h":
            ema_fast = self._store.get_value("1h", "ema_50")
            ema_slow = self._store.get_value("1h", "ema_200")
        else:
            ema_fast = self._store.get_value(tf, "ema_20")
            ema_slow = self._store.get_value(tf, "ema_50")
        
        if ema_fast is None or ema_slow is None:
            return "NEUTRAL"
        
        if ema_slow == 0:
            return "NEUTRAL"
        
        # Calculate spread in bps
        spread_bps = (ema_fast - ema_slow) / ema_slow * 10000
        
        if spread_bps > EMA_CROSS_THRESHOLD_BPS:
            return "UP"
        elif spread_bps < -EMA_CROSS_THRESHOLD_BPS:
            return "DOWN"
        else:
            return "NEUTRAL"
    
    def _compute_regime(self, tf: str) -> str:
        """
        Compute market regime (trending vs ranging).
        
        Logic:
            ADX >= 20 → TREND
            ADX < 20 → RANGE
        
        Args:
            tf: Timeframe (15m or 5m)
            
        Returns:
            "TREND" or "RANGE"
        """
        adx = self._store.get_value(tf, "adx_14")
        
        if adx is None:
            return "RANGE"  # Default to range when unknown
        
        if adx >= ADX_TREND_THRESHOLD:
            return "TREND"
        else:
            return "RANGE"
    
    def _compute_micro_stage(self) -> str:
        """
        Compute micro stage from 1m indicators.
        
        Logic:
            ACCEL_UP: ROC > 0 AND MACD_hist > 0 AND EMA20 > EMA50
            ACCEL_DOWN: ROC < 0 AND MACD_hist < 0 AND EMA20 < EMA50
            OVERHEATED: RSI > 70 OR RSI < 30
            NEUTRAL: otherwise
        
        Returns:
            "ACCEL_UP", "ACCEL_DOWN", "OVERHEATED", or "NEUTRAL"
        """
        # Get 1m indicators
        roc = self._store.get_value("1m", "roc_9")
        rsi = self._store.get_value("1m", "rsi_14")
        ema20 = self._store.get_value("1m", "ema_20")
        ema50 = self._store.get_value("1m", "ema_50")
        
        # Get MACD histogram
        macd_data = self._store.get("1m")
        macd_hist = None
        if macd_data and "macd" in macd_data:
            macd_result = macd_data["macd"]
            if isinstance(macd_result, dict):
                macd_hist = macd_result.get("valueMACDHist")
        
        # Check for overheated first
        if rsi is not None:
            if rsi > RSI_OVERBOUGHT or rsi < RSI_OVERSOLD:
                return "OVERHEATED"
        
        # Check for acceleration
        if roc is not None and macd_hist is not None and ema20 is not None and ema50 is not None:
            if roc > 0 and macd_hist > 0 and ema20 > ema50:
                return "ACCEL_UP"
            elif roc < 0 and macd_hist < 0 and ema20 < ema50:
                return "ACCEL_DOWN"
        
        return "NEUTRAL"
    
    def _compute_vol_regime(self) -> str:
        """
        Compute volatility regime from ATR.
        
        Compares current ATR to historical median.
        
        Returns:
            "HIGH", "NORMAL", or "LOW"
        """
        atr_15m = self._store.get_value("15m", "atr_14")
        
        if atr_15m is None:
            return "NORMAL"
        
        # Add to history
        self._atr_history_15m.append(atr_15m)
        
        if len(self._atr_history_15m) < 5:
            return "NORMAL"  # Not enough data
        
        # Calculate percentile
        sorted_atr = sorted(self._atr_history_15m)
        median_atr = sorted_atr[len(sorted_atr) // 2]
        
        if median_atr == 0:
            return "NORMAL"
        
        ratio = atr_15m / median_atr
        
        if ratio > 1.3:  # 30% above median
            return "HIGH"
        elif ratio < 0.7:  # 30% below median
            return "LOW"
        else:
            return "NORMAL"
    
    def _compute_alignment_score(
        self,
        bias_1h: str,
        bias_15m: str,
        bias_5m: str,
        bias_1m: str,
    ) -> int:
        """
        Compute alignment score (0-100).
        
        Measures how well timeframes agree on direction.
        
        Args:
            bias_1h: 1h bias
            bias_15m: 15m bias
            bias_5m: 5m bias
            bias_1m: 1m bias
            
        Returns:
            Score 0-100
        """
        # Weight by timeframe importance
        weights = {
            "1h": 40,    # Background has highest weight
            "15m": 30,   # Main window
            "5m": 20,    # Local structure
            "1m": 10,    # Micro (least weight)
        }
        
        biases = {
            "1h": bias_1h,
            "15m": bias_15m,
            "5m": bias_5m,
            "1m": bias_1m,
        }
        
        # Determine dominant direction
        up_weight = sum(weights[tf] for tf, b in biases.items() if b == "UP")
        down_weight = sum(weights[tf] for tf, b in biases.items() if b == "DOWN")
        
        # Score is the dominant direction's weight
        return max(up_weight, down_weight)
    
    def _get_supertrend_direction(self, tf: str) -> Optional[str]:
        """
        Get Supertrend direction.
        
        Args:
            tf: Timeframe
            
        Returns:
            "LONG", "SHORT", or None
        """
        data = self._store.get(tf)
        if not data:
            return None
        
        supertrend = data.get("supertrend")
        if not supertrend:
            return None
        
        if isinstance(supertrend, dict):
            advice = supertrend.get("valueAdvice")
            if advice:
                return advice.upper()
        
        return None
    
    def _get_macd_signal(self, tf: str) -> Optional[str]:
        """
        Get MACD signal (histogram direction).
        
        Args:
            tf: Timeframe
            
        Returns:
            "BULLISH", "BEARISH", or None
        """
        data = self._store.get(tf)
        if not data:
            return None
        
        macd = data.get("macd")
        if not macd:
            return None
        
        if isinstance(macd, dict):
            hist = macd.get("valueMACDHist")
            if hist is not None:
                return "BULLISH" if hist > 0 else "BEARISH"
        
        return None
    
    def _get_data_ages(self) -> dict:
        """
        Get age of data per timeframe.
        
        Returns:
            Dict of tf -> age_sec
        """
        return {
            tf: round(self._store.get_age_sec(tf), 1) if self._store.get_age_sec(tf) else None
            for tf in ["1m", "5m", "15m", "1h"]
        }
    
    def get_summary(self) -> dict:
        """
        Get short summary for logging.
        
        Returns:
            Dict with key context fields
        """
        snapshot = self.snapshot()
        
        return {
            "bias_1h": snapshot["bias_1h"],
            "regime_15m": snapshot["regime_15m"],
            "micro_stage": snapshot["micro_stage"],
            "vol_regime": snapshot["vol_regime"],
            "alignment": snapshot["alignment_score"],
        }
