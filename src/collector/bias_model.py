"""
Bias Model (S4)
===============

Computes slow directional bias from price data across multiple timeframes.
Updates infrequently (every 10-30s or on candle close) to avoid noise.

Why this exists:
    Bias represents the "slow probability" of direction - a trend filter
    that helps avoid trading against the dominant market direction.
    
    Unlike fair model (which reacts to every tick), bias changes slowly
    and provides context for decision confidence.

Components:
    1. Slope (bps): Linear regression slope of log-prices
    2. EMA spread: EMA(fast) - EMA(slow) normalized
    3. Stage: TREND vs RANGE based on |slope| / sigma

Output:
    - dir: "UP" | "DOWN" | "NEUTRAL"
    - strength: 0..1 (how strong the bias is)
    - bias_up_prob: 0..1 (probability-like score)
    - breakdown by TF: 1m, 5m, 15m, 1h

Configuration:
    BIAS_SLOPE_LOOKBACK: 20 bars for slope calculation
    BIAS_EMA_FAST: 12 bars
    BIAS_EMA_SLOW: 26 bars
    BIAS_SLOPE_WEIGHT: 0.6
    BIAS_EMA_WEIGHT: 0.4
    BIAS_UP_THRESHOLD: 0.55
    BIAS_DOWN_THRESHOLD: 0.45

Usage:
    bias = BiasModel()
    
    # Update with 1m closes (will aggregate to higher TFs)
    bias.update_1m(close_px, ts_ms)
    
    # Or update specific TF
    bias.update_tf("5m", close_px, ts_ms)
    
    # Get snapshot
    snap = bias.snapshot()
    # {
    #   "dir": "UP",
    #   "strength": 0.7,
    #   "bias_up_prob": 0.65,
    #   "tf_breakdown": {"1m": {...}, "5m": {...}, ...}
    # }
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_SLOPE_LOOKBACK = 20
DEFAULT_EMA_FAST = 12
DEFAULT_EMA_SLOW = 26
DEFAULT_SLOPE_WEIGHT = 0.6
DEFAULT_EMA_WEIGHT = 0.4
DEFAULT_UP_THRESHOLD = 0.55
DEFAULT_DOWN_THRESHOLD = 0.45


@dataclass
class TFState:
    """State for a single timeframe."""
    closes: deque = field(default_factory=lambda: deque(maxlen=100))
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    last_update_ms: Optional[int] = None
    last_close: Optional[float] = None


class BiasModel:
    """
    Multi-timeframe bias model.
    
    Computes directional bias from price slope and EMA spread.
    Updates slowly to avoid noise.
    """
    
    def __init__(
        self,
        slope_lookback: int = DEFAULT_SLOPE_LOOKBACK,
        ema_fast: int = DEFAULT_EMA_FAST,
        ema_slow: int = DEFAULT_EMA_SLOW,
        slope_weight: float = DEFAULT_SLOPE_WEIGHT,
        ema_weight: float = DEFAULT_EMA_WEIGHT,
        up_threshold: float = DEFAULT_UP_THRESHOLD,
        down_threshold: float = DEFAULT_DOWN_THRESHOLD,
    ):
        self.slope_lookback = slope_lookback
        self.ema_fast_period = ema_fast
        self.ema_slow_period = ema_slow
        self.slope_weight = slope_weight
        self.ema_weight = ema_weight
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold
        
        # State per timeframe
        self._tf_states: dict[str, TFState] = {
            "1m": TFState(closes=deque(maxlen=100)),
            "5m": TFState(closes=deque(maxlen=100)),
            "15m": TFState(closes=deque(maxlen=100)),
            "1h": TFState(closes=deque(maxlen=100)),
        }
        
        # Aggregation buffers (for building 5m/15m/1h from 1m)
        self._1m_buffer: list[float] = []
        self._1m_count = 0
        
        # Cache for slow updates
        self._last_snapshot_ms: Optional[int] = None
        self._cached_snapshot: Optional[dict] = None
        self._snapshot_interval_ms = 10_000  # Update every 10s
        
        logger.info(
            "bias_model_initialized",
            extra={
                "slope_lookback": slope_lookback,
                "ema_fast": ema_fast,
                "ema_slow": ema_slow,
            },
        )
    
    def update_1m(self, close_px: float, ts_ms: int) -> None:
        """
        Update with 1-minute close.
        
        Automatically aggregates to higher timeframes.
        """
        if close_px <= 0:
            return
        
        # Update 1m
        self._update_tf("1m", close_px, ts_ms)
        
        # Aggregate to higher TFs
        self._1m_buffer.append(close_px)
        self._1m_count += 1
        
        # Every 5 bars -> 5m
        if self._1m_count % 5 == 0:
            self._update_tf("5m", close_px, ts_ms)
        
        # Every 15 bars -> 15m
        if self._1m_count % 15 == 0:
            self._update_tf("15m", close_px, ts_ms)
        
        # Every 60 bars -> 1h
        if self._1m_count % 60 == 0:
            self._update_tf("1h", close_px, ts_ms)
    
    def update_tf(self, tf: str, close_px: float, ts_ms: int) -> None:
        """Update specific timeframe directly."""
        if close_px <= 0:
            return
        self._update_tf(tf, close_px, ts_ms)
    
    def _update_tf(self, tf: str, close_px: float, ts_ms: int) -> None:
        """Internal update for a timeframe."""
        if tf not in self._tf_states:
            return
        
        state = self._tf_states[tf]
        state.closes.append(close_px)
        state.last_close = close_px
        state.last_update_ms = ts_ms
        
        # Update EMAs
        if state.ema_fast is None:
            state.ema_fast = close_px
            state.ema_slow = close_px
        else:
            alpha_fast = 2.0 / (self.ema_fast_period + 1)
            alpha_slow = 2.0 / (self.ema_slow_period + 1)
            state.ema_fast = alpha_fast * close_px + (1 - alpha_fast) * state.ema_fast
            state.ema_slow = alpha_slow * close_px + (1 - alpha_slow) * state.ema_slow
        
        # Invalidate cache
        self._cached_snapshot = None
    
    def warmup_tf(self, tf: str, closes: list[float], ts_ms: int) -> int:
        """
        Warmup a timeframe from historical closes.
        
        Args:
            tf: Timeframe ("1m", "5m", etc.)
            closes: List of close prices in chronological order
            ts_ms: Timestamp for last update
            
        Returns:
            Number of closes processed
        """
        if tf not in self._tf_states:
            return 0
        
        state = self._tf_states[tf]
        state.closes.clear()
        state.ema_fast = None
        state.ema_slow = None
        
        count = 0
        for close_px in closes:
            if close_px <= 0:
                continue
            state.closes.append(close_px)
            state.last_close = close_px
            
            # Update EMAs
            if state.ema_fast is None:
                state.ema_fast = close_px
                state.ema_slow = close_px
            else:
                alpha_fast = 2.0 / (self.ema_fast_period + 1)
                alpha_slow = 2.0 / (self.ema_slow_period + 1)
                state.ema_fast = alpha_fast * close_px + (1 - alpha_fast) * state.ema_fast
                state.ema_slow = alpha_slow * close_px + (1 - alpha_slow) * state.ema_slow
            
            count += 1
        
        state.last_update_ms = ts_ms
        
        logger.info(
            "bias_model_warmup",
            extra={"tf": tf, "closes_count": len(closes), "processed": count},
        )
        
        return count
    
    def _compute_slope_bps(self, closes: list[float]) -> Optional[float]:
        """
        Compute slope in bps using linear regression on log-prices.
        
        Returns slope per bar in bps.
        """
        n = len(closes)
        if n < 3:
            return None
        
        # Use last slope_lookback bars
        use_n = min(n, self.slope_lookback)
        prices = closes[-use_n:]
        
        # Log prices
        log_prices = []
        for p in prices:
            if p <= 0:
                return None
            log_prices.append(math.log(p))
        
        # Linear regression: y = a + b*x
        # b = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
        n = len(log_prices)
        sum_x = sum(range(n))
        sum_y = sum(log_prices)
        sum_xy = sum(i * y for i, y in enumerate(log_prices))
        sum_x2 = sum(i * i for i in range(n))
        
        denom = n * sum_x2 - sum_x * sum_x
        if abs(denom) < 1e-10:
            return 0.0
        
        slope = (n * sum_xy - sum_x * sum_y) / denom
        
        # Convert to bps per bar
        return slope * 10000
    
    def _compute_tf_bias(self, tf: str) -> dict:
        """
        Compute bias for a single timeframe.
        
        Returns dict with slope_bps, ema_spread_bps, bias_score, dir.
        """
        state = self._tf_states.get(tf)
        if state is None or len(state.closes) < 5:
            return {
                "ready": False,
                "slope_bps": None,
                "ema_spread_bps": None,
                "bias_score": 0.5,
                "dir": "NEUTRAL",
            }
        
        closes = list(state.closes)
        
        # Compute slope
        slope_bps = self._compute_slope_bps(closes)
        
        # Compute EMA spread
        ema_spread_bps = None
        if state.ema_fast is not None and state.ema_slow is not None and state.ema_slow > 0:
            ema_spread_bps = (state.ema_fast - state.ema_slow) / state.ema_slow * 10000
        
        # Normalize to [-1, 1] range
        # FIX-BIAS-002: Previous normalization was too aggressive!
        # Real data shows slope: -1 to +3 bps, spread: -7 to +10 bps
        # Old: slope/30, spread/80 → always NEUTRAL
        # New: slope/3, spread/15 → proper sensitivity
        slope_norm = 0.0
        if slope_bps is not None:
            slope_norm = max(-1.0, min(1.0, slope_bps / 3.0))
        
        # EMA spread: observed -7 to +10 bps → normalize by 15
        ema_norm = 0.0
        if ema_spread_bps is not None:
            ema_norm = max(-1.0, min(1.0, ema_spread_bps / 15.0))
        
        # Combine
        combined = self.slope_weight * slope_norm + self.ema_weight * ema_norm
        
        # Convert to probability-like score [0.05, 0.95]
        bias_score = 0.5 + combined * 0.45
        bias_score = max(0.05, min(0.95, bias_score))
        
        # Direction
        if bias_score > self.up_threshold:
            direction = "UP"
        elif bias_score < self.down_threshold:
            direction = "DOWN"
        else:
            direction = "NEUTRAL"
        
        return {
            "ready": True,
            "slope_bps": round(slope_bps, 2) if slope_bps else None,
            "ema_spread_bps": round(ema_spread_bps, 2) if ema_spread_bps else None,
            "ema_fast": round(state.ema_fast, 2) if state.ema_fast else None,
            "ema_slow": round(state.ema_slow, 2) if state.ema_slow else None,
            "bias_score": round(bias_score, 3),
            "dir": direction,
            "n_bars": len(state.closes),
            "last_update_ms": state.last_update_ms,
        }
    
    def _taapi_bias_to_score(self, bias_str: str) -> float:
        """Convert TAAPI bias string to numeric score."""
        if bias_str == "UP":
            return 0.7
        elif bias_str == "DOWN":
            return 0.3
        else:
            return 0.5
    
    def snapshot(self, force: bool = False, taapi_context: Optional[dict] = None) -> dict:
        """
        Get bias snapshot with optional TAAPI integration.
        
        Uses cached value if within snapshot_interval to reduce noise.
        
        Args:
            force: If True, recompute even if cached
            taapi_context: Optional TAAPI context with bias_1h, bias_15m, etc.
        
        Returns:
            Dictionary with overall bias and per-TF breakdown
        """
        ts = now_ms()
        
        # Use cache if recent enough (but only if no taapi_context provided)
        if not force and self._cached_snapshot is not None and taapi_context is None:
            if self._last_snapshot_ms and (ts - self._last_snapshot_ms) < self._snapshot_interval_ms:
                return self._cached_snapshot
        
        # Compute per-TF bias from our own data
        tf_breakdown = {}
        own_weighted_score = 0.0
        own_total_weight = 0.0
        
        # Weights per TF (higher TFs have more weight)
        tf_weights = {"1m": 0.1, "5m": 0.2, "15m": 0.3, "1h": 0.4}
        
        for tf in ["1m", "5m", "15m", "1h"]:
            tf_bias = self._compute_tf_bias(tf)
            tf_breakdown[tf] = tf_bias
            
            if tf_bias["ready"]:
                w = tf_weights.get(tf, 0.1)
                own_weighted_score += tf_bias["bias_score"] * w
                own_total_weight += w
        
        # Our own bias score
        if own_total_weight > 0:
            own_score = own_weighted_score / own_total_weight
        else:
            own_score = 0.5
        
        # =========================================
        # TAAPI INTEGRATION: Combine with TAAPI bias
        # =========================================
        taapi_score = None
        taapi_dir = None
        taapi_weight = 0.0
        
        if taapi_context:
            # Extract TAAPI bias per TF
            taapi_biases = {
                "1h": taapi_context.get("bias_1h", "NEUTRAL"),
                "15m": taapi_context.get("bias_15m", "NEUTRAL"),
                "5m": taapi_context.get("bias_5m", "NEUTRAL"),
                "1m": taapi_context.get("bias_1m", "NEUTRAL"),
            }
            
            # Convert to weighted score (same weights as our own)
            taapi_weighted = 0.0
            taapi_total = 0.0
            for tf, bias_str in taapi_biases.items():
                w = tf_weights.get(tf, 0.1)
                score = self._taapi_bias_to_score(bias_str)
                taapi_weighted += score * w
                taapi_total += w
                
                # Add TAAPI info to breakdown
                if tf in tf_breakdown:
                    tf_breakdown[tf]["taapi_bias"] = bias_str
                    tf_breakdown[tf]["taapi_score"] = score
            
            if taapi_total > 0:
                taapi_score = taapi_weighted / taapi_total
                taapi_weight = 0.5  # Give TAAPI 50% weight when available
                
                # Determine TAAPI direction
                if taapi_score > self.up_threshold:
                    taapi_dir = "UP"
                elif taapi_score < self.down_threshold:
                    taapi_dir = "DOWN"
                else:
                    taapi_dir = "NEUTRAL"
        
        # =========================================
        # COMBINED SCORE: Own + TAAPI
        # =========================================
        if taapi_score is not None and taapi_weight > 0:
            # Hybrid: combine own and TAAPI
            own_weight = 1.0 - taapi_weight
            overall_score = own_score * own_weight + taapi_score * taapi_weight
        else:
            # Fallback: use only our own score
            overall_score = own_score
        
        # Overall direction
        if overall_score > self.up_threshold:
            overall_dir = "UP"
        elif overall_score < self.down_threshold:
            overall_dir = "DOWN"
        else:
            overall_dir = "NEUTRAL"
        
        # Strength: 0..1 based on distance from 0.5
        strength = abs(overall_score - 0.5) * 2
        
        result = {
            "dir": overall_dir,
            "strength": round(strength, 3),
            "bias_up_prob": round(overall_score, 3),
            "bias_down_prob": round(1.0 - overall_score, 3),
            "tf_breakdown": tf_breakdown,
            "last_update_ms": ts,
            # Source tracking
            "own_score": round(own_score, 3),
            "taapi_score": round(taapi_score, 3) if taapi_score is not None else None,
            "taapi_dir": taapi_dir,
            "taapi_integrated": taapi_score is not None,
        }
        
        # Cache (only if no taapi_context to avoid stale hybrid data)
        if taapi_context is None:
            self._cached_snapshot = result
            self._last_snapshot_ms = ts
        
        return result
