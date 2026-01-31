"""
Volatility Estimator (S2 Multi-scale)
=====================================

Estimates realized volatility from 1-minute bar closes with multiple scales:
- sigma_fast_15m: Fast-reacting volatility (default 60 minutes)
- sigma_slow_15m: Slow/smoothed volatility (default 360 minutes / 6 hours)
- sigma_blend_15m: Adaptive blend based on market activity (rvol)
- sigma_15m: Legacy field (= sigma_fast_15m for backward compatibility)

Why multiple scales:
    - Fast sigma reacts quickly to volatility spikes (good for risk management)
    - Slow sigma provides stable baseline (good for fair value estimation)
    - Blend adapts: more weight to fast during high activity, more to slow when calm

Methodology:
    1. Collect close prices from 1-minute bars (store up to slow_minutes + buffer)
    2. Compute log returns: r_t = ln(close_t / close_{t-1})
    3. sigma_fast_1m = std(r_t) over fast window
    4. sigma_slow_1m = std(r_t) over slow window
    5. Scale to 15m: sigma_*_15m = sigma_*_1m * sqrt(15)
    6. Blend: sigma_blend = w * sigma_fast + (1-w) * sigma_slow
       where w depends on rvol_5s (higher activity -> more weight to fast)

Configuration (config.py):
    VOL_FAST_MINUTES: 60 (1 hour)
    VOL_SLOW_MINUTES: 360 (6 hours)
    VOL_MIN_BARS: 20 (warmup threshold)
    SIGMA_BLEND_W_RVOL_K: 0.5 (sensitivity to rvol)
    SIGMA_BLEND_W_CLAMP_MIN: 0.2
    SIGMA_BLEND_W_CLAMP_MAX: 0.9

Usage:
    from collector.config import settings
    
    vol = VolEstimator(
        fast_minutes=settings.VOL_FAST_MINUTES,
        slow_minutes=settings.VOL_SLOW_MINUTES,
        min_bars=settings.VOL_MIN_BARS,
    )
    
    # Feed 1m bar closes
    vol.update_1m_close(105000.0, ts_ms)
    
    # Get snapshot with optional rvol for blend weight
    snap = vol.snapshot(rvol_5s=1.5)
    # {
    #   "sigma_fast_15m": 0.0023,
    #   "sigma_slow_15m": 0.0019,
    #   "sigma_blend_15m": 0.0021,
    #   "sigma_15m": 0.0023,  # legacy = fast
    #   "blend_w": 0.65,
    #   "n_bars": 120,
    #   "reason": "ok"
    # }
"""

import logging
import math
from collections import deque
from typing import Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# Default values (can be overridden via config)
DEFAULT_FAST_MINUTES = 60
DEFAULT_SLOW_MINUTES = 360
DEFAULT_MIN_BARS = 20

# Blend weight parameters
DEFAULT_BLEND_W_RVOL_K = 0.5
DEFAULT_BLEND_W_CLAMP_MIN = 0.2
DEFAULT_BLEND_W_CLAMP_MAX = 0.9


class VolEstimator:
    """
    Multi-scale volatility estimator from 1-minute bar closes.
    
    Computes:
    - sigma_fast_15m: Volatility from recent fast_minutes bars
    - sigma_slow_15m: Volatility from recent slow_minutes bars
    - sigma_blend_15m: Adaptive blend based on rvol
    - sigma_15m: Legacy field (= sigma_fast_15m)
    
    Attributes:
        fast_minutes: Window size for fast sigma (default 60)
        slow_minutes: Window size for slow sigma (default 360)
        min_bars: Minimum bars for valid sigma calculation
    """
    
    def __init__(
        self,
        fast_minutes: int = DEFAULT_FAST_MINUTES,
        slow_minutes: int = DEFAULT_SLOW_MINUTES,
        min_bars: int = DEFAULT_MIN_BARS,
        # Legacy parameters for backward compatibility
        lookback: Optional[int] = None,
        min_returns: Optional[int] = None,
    ):
        """
        Initialize the volatility estimator.
        
        Args:
            fast_minutes: Fast sigma window (default 60 = 1 hour)
            slow_minutes: Slow sigma window (default 360 = 6 hours)
            min_bars: Minimum bars for warmup (default 20)
            lookback: Legacy parameter (ignored, uses fast_minutes)
            min_returns: Legacy parameter (maps to min_bars)
        """
        # Handle legacy parameters
        if lookback is not None:
            fast_minutes = lookback
        if min_returns is not None:
            min_bars = min_returns
        
        self.fast_minutes = fast_minutes
        self.slow_minutes = slow_minutes
        self.min_bars = min_bars
        
        # Legacy aliases for backward compatibility
        self.lookback = fast_minutes
        self.min_returns = min_bars
        
        # Store (ts_ms, close_px) tuples for flexibility
        # Need slow_minutes + buffer to compute slow sigma
        max_size = max(slow_minutes + 10, 500)
        self._closes: deque[tuple[int, float]] = deque(maxlen=max_size)
        
        # Also store log returns for backward compatibility
        self._returns: deque[float] = deque(maxlen=fast_minutes)
        
        # Last close for computing return
        self._last_close: Optional[float] = None
        self._last_ts_ms: Optional[int] = None
        
        # Update count
        self._update_count = 0
        
        # Blend weight config (load from settings if available)
        try:
            from collector.config import settings
            self._blend_w_rvol_k = settings.SIGMA_BLEND_W_RVOL_K
            self._blend_w_clamp_min = settings.SIGMA_BLEND_W_CLAMP_MIN
            self._blend_w_clamp_max = settings.SIGMA_BLEND_W_CLAMP_MAX
        except Exception:
            self._blend_w_rvol_k = DEFAULT_BLEND_W_RVOL_K
            self._blend_w_clamp_min = DEFAULT_BLEND_W_CLAMP_MIN
            self._blend_w_clamp_max = DEFAULT_BLEND_W_CLAMP_MAX
        
        logger.info(
            "vol_estimator_initialized",
            extra={
                "fast_minutes": fast_minutes,
                "slow_minutes": slow_minutes,
                "min_bars": min_bars,
                "max_closes": max_size,
            },
        )
    
    def update_1m_close(self, close_px: float, ts_ms: int) -> Optional[float]:
        """
        Update with a new 1-minute bar close price.
        
        Args:
            close_px: Close price of the 1-minute bar
            ts_ms: Timestamp of the bar close in milliseconds
        
        Returns:
            The log return if computed, None if first price
        """
        if close_px <= 0:
            logger.warning(
                "vol_estimator_invalid_price",
                extra={"close_px": close_px, "ts_ms": ts_ms},
            )
            return None
        
        log_return = None
        
        # Compute log return from previous close
        if self._last_close is not None and self._last_close > 0:
            log_return = math.log(close_px / self._last_close)
            self._returns.append(log_return)
            self._update_count += 1
        
        # Store (ts_ms, close_px) for multi-scale computation
        self._closes.append((int(ts_ms), float(close_px)))
        
        self._last_close = close_px
        self._last_ts_ms = ts_ms
        
        return log_return
    
    def _compute_log_returns(self, n: int) -> Optional[list[float]]:
        """
        Compute log returns from last n+1 closes.
        
        Args:
            n: Number of returns to compute (needs n+1 closes)
        
        Returns:
            List of log returns, or None if not enough data
        """
        if len(self._closes) < n + 1:
            return None
        
        # Get last n+1 closes
        closes = [c for _, c in list(self._closes)[-(n + 1):]]
        
        returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] <= 0 or closes[i] <= 0:
                continue
            returns.append(math.log(closes[i] / closes[i - 1]))
        
        return returns if returns else None
    
    def _stdev(self, xs: list[float]) -> Optional[float]:
        """
        Compute sample standard deviation.
        
        Args:
            xs: List of values
        
        Returns:
            Standard deviation, or None if not enough data
        """
        if not xs or len(xs) < 2:
            return None
        
        n = len(xs)
        mean = sum(xs) / n
        variance = sum((x - mean) ** 2 for x in xs) / (n - 1)  # Sample std
        
        return math.sqrt(variance)
    
    def get_sigma_1m(self) -> Optional[float]:
        """
        Get 1-minute volatility (backward compatibility).
        Uses fast window.
        """
        if len(self._returns) < self.min_bars:
            return None
        
        n = len(self._returns)
        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / n
        
        return math.sqrt(variance)
    
    def get_sigma_15m(self) -> Optional[float]:
        """
        Get 15-minute volatility (backward compatibility).
        Uses fast sigma scaled.
        """
        sigma_1m = self.get_sigma_1m()
        if sigma_1m is None:
            return None
        return sigma_1m * math.sqrt(15)
    
    def snapshot(self, rvol_5s: Optional[float] = None) -> dict:
        """
        Get current volatility estimates with multi-scale sigmas.
        
        Args:
            rvol_5s: Optional relative volume (for blend weight calculation)
                     rvol ~1.0 is neutral, >1 means higher activity
        
        Returns:
            Dictionary with:
                - sigma_fast_15m: Fast volatility (scaled to 15m)
                - sigma_slow_15m: Slow volatility (scaled to 15m)
                - sigma_blend_15m: Blended volatility
                - sigma_15m: Legacy field (= sigma_fast_15m)
                - sigma_1m: Legacy 1-minute volatility
                - blend_w: Weight used for blending
                - n_bars: Number of closes stored
                - n_returns: Number of returns (legacy)
                - reason: Status ("ok", "sigma_warmup", etc.)
                - ready: True if valid sigma available
                - last_update_ms, last_update_age_sec: Timing info
        """
        n_bars = len(self._closes)
        n_returns = len(self._returns)
        
        # Compute age
        last_update_age_sec = None
        if self._last_ts_ms:
            last_update_age_sec = round((now_ms() - self._last_ts_ms) / 1000, 1)
        
        # Base result with diagnostics
        result = {
            "n_bars": n_bars,
            "n_returns": n_returns,
            "lookback": self.lookback,
            "min_returns": self.min_returns,
            "last_close": self._last_close,
            "last_update_ms": self._last_ts_ms,
            "last_update_age_sec": last_update_age_sec,
            "update_count": self._update_count,
        }
        
        # Check warmup
        if n_bars < self.min_bars + 1:
            result.update({
                "sigma_fast_15m": None,
                "sigma_slow_15m": None,
                "sigma_blend_15m": None,
                "sigma_15m": None,
                "sigma_1m": None,
                "blend_w": None,
                "reason": "sigma_warmup",
                "ready": False,
            })
            return result
        
        # Compute fast sigma from fast_minutes returns
        fast_rets = self._compute_log_returns(self.fast_minutes)
        slow_rets = self._compute_log_returns(self.slow_minutes)
        
        sig_fast_1m = self._stdev(fast_rets) if fast_rets else None
        sig_slow_1m = self._stdev(slow_rets) if slow_rets else None
        
        reason = "ok"
        
        # Handle missing sigmas
        if sig_fast_1m is None and sig_slow_1m is None:
            result.update({
                "sigma_fast_15m": None,
                "sigma_slow_15m": None,
                "sigma_blend_15m": None,
                "sigma_15m": None,
                "sigma_1m": None,
                "blend_w": None,
                "reason": "sigma_missing",
                "ready": False,
            })
            return result
        
        # Fallback: if slow not ready but fast is, use fast for slow
        if sig_slow_1m is None and sig_fast_1m is not None:
            sig_slow_1m = sig_fast_1m
            reason = "slow_fallback_to_fast"
        
        # Scale 1m -> 15m using sqrt(15) rule
        sqrt_15 = math.sqrt(15)
        sig_fast_15m = sig_fast_1m * sqrt_15 if sig_fast_1m is not None else None
        sig_slow_15m = sig_slow_1m * sqrt_15 if sig_slow_1m is not None else None
        
        # Compute blend weight based on rvol_5s
        # rvol ~1 is neutral, >1 means more activity -> more weight to fast
        w = 0.5
        if rvol_5s is not None:
            w = 0.5 + (rvol_5s - 1.0) * self._blend_w_rvol_k
        w = max(self._blend_w_clamp_min, min(self._blend_w_clamp_max, w))
        
        # Compute blend
        if sig_fast_15m is None:
            sig_blend_15m = sig_slow_15m
            if reason == "ok":
                reason = "blend_uses_slow_only"
        elif sig_slow_15m is None:
            sig_blend_15m = sig_fast_15m
            if reason == "ok":
                reason = "blend_uses_fast_only"
        else:
            sig_blend_15m = w * sig_fast_15m + (1 - w) * sig_slow_15m
        
        # Round values
        def safe_round(x, decimals=6):
            return round(x, decimals) if x is not None else None
        
        result.update({
            "sigma_fast_15m": safe_round(sig_fast_15m),
            "sigma_slow_15m": safe_round(sig_slow_15m),
            "sigma_blend_15m": safe_round(sig_blend_15m),
            "sigma_15m": safe_round(sig_fast_15m),  # Legacy = fast
            "sigma_1m": safe_round(sig_fast_1m),
            "blend_w": round(w, 3),
            "reason": reason,
            "ready": True,
        })
        
        return result
    
    def warmup_from_closes(self, closes: list[float], ts_ms: int) -> int:
        """
        Warmup estimator from historical close prices.
        
        Args:
            closes: List of close prices in chronological order
            ts_ms: Timestamp to set as last update
            
        Returns:
            Number of returns computed
        """
        if not closes:
            return 0
        
        # Clear existing state
        self._closes.clear()
        self._returns.clear()
        self._last_close = None
        self._last_ts_ms = None
        
        # Process closes
        returns_added = 0
        # Use ts_ms as base, assume 1 minute apart
        for i, close_px in enumerate(closes):
            if close_px <= 0:
                continue
            
            # Fake timestamp (1 minute apart)
            fake_ts = ts_ms - (len(closes) - i - 1) * 60_000
            self._closes.append((fake_ts, close_px))
            
            if self._last_close is not None and self._last_close > 0:
                log_return = math.log(close_px / self._last_close)
                self._returns.append(log_return)
                returns_added += 1
            
            self._last_close = close_px
        
        self._last_ts_ms = ts_ms
        self._update_count += returns_added
        
        logger.info(
            "vol_estimator_warmup",
            extra={
                "closes_count": len(closes),
                "returns_added": returns_added,
                "n_bars": len(self._closes),
                "n_returns": len(self._returns),
                "ready": self.is_ready(),
            },
        )
        
        return returns_added
    
    def is_ready(self) -> bool:
        """Check if estimator has enough data for valid sigma."""
        return len(self._closes) >= self.min_bars + 1
    
    def get_recent_returns(self, n: int = 10) -> list[float]:
        """Get most recent log returns for debugging."""
        return list(self._returns)[-n:]
