"""
ROC Calculator (Rate of Change)
===============================

Calculates short-term price momentum by tracking price history
and computing rate of change over configurable windows.

Why this exists:
    ROC captures momentum - whether price is accelerating up or down.
    This is critical for "countertrend" triggers:
    - When Z > 0 (BTC above ref) but ROC < 0 (falling) → buy DOWN
    - When Z < 0 (BTC below ref) but ROC > 0 (rising) → buy UP

Output:
    - roc_30s: ROC over 30 seconds in bps
    - roc_60s: ROC over 60 seconds in bps
    - roc_direction: "UP" | "DOWN" | "FLAT"
    - roc_strength: 0..1 (how strong the momentum is)

Usage:
    roc = RocCalculator(window_sec=30)
    roc.update(price=89500.0, ts_ms=1234567890)
    snap = roc.snapshot()
    # {"roc_30s": -5.2, "roc_60s": -8.1, "direction": "DOWN", "strength": 0.4}
"""

import logging
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Thresholds
ROC_FLAT_THRESHOLD_BPS = 1.0  # Below this, consider flat
ROC_STRONG_THRESHOLD_BPS = 5.0  # Above this, consider strong


@dataclass
class PricePoint:
    """Single price observation."""
    price: float
    ts_ms: int


class RocCalculator:
    """
    Rate of Change calculator for BTC price.
    
    Maintains a rolling window of prices and computes ROC.
    """
    
    def __init__(
        self,
        window_30s: int = 30,
        window_60s: int = 60,
        max_history_sec: int = 120,
    ):
        """
        Initialize ROC calculator.
        
        Args:
            window_30s: Window for short ROC (seconds)
            window_60s: Window for longer ROC (seconds)
            max_history_sec: Maximum history to keep (seconds)
        """
        self.window_30s = window_30s
        self.window_60s = window_60s
        self.max_history_sec = max_history_sec
        
        # Price history (sorted by time)
        self._history: deque[PricePoint] = deque(maxlen=max_history_sec * 10)  # ~10 updates/sec
        
        # Cached values
        self._last_roc_30s: Optional[float] = None
        self._last_roc_60s: Optional[float] = None
        self._last_update_ms: Optional[int] = None
        
        logger.info(
            "roc_calculator_initialized",
            extra={
                "window_30s": window_30s,
                "window_60s": window_60s,
                "max_history_sec": max_history_sec,
            },
        )
    
    def update(self, price: float, ts_ms: int) -> None:
        """
        Update with new price observation.
        
        Args:
            price: Current BTC price
            ts_ms: Timestamp in milliseconds
        """
        if price <= 0:
            return
        
        self._history.append(PricePoint(price=price, ts_ms=ts_ms))
        self._last_update_ms = ts_ms
        
        # Prune old data
        cutoff_ms = ts_ms - (self.max_history_sec * 1000)
        while self._history and self._history[0].ts_ms < cutoff_ms:
            self._history.popleft()
        
        # Compute ROC
        self._compute_roc(price, ts_ms)
    
    def _compute_roc(self, current_price: float, current_ts_ms: int) -> None:
        """Compute ROC values."""
        # Find price N seconds ago
        def get_price_at_offset(offset_sec: int) -> Optional[float]:
            target_ts = current_ts_ms - (offset_sec * 1000)
            best_point = None
            best_diff = float('inf')
            
            for p in self._history:
                diff = abs(p.ts_ms - target_ts)
                if diff < best_diff:
                    best_diff = diff
                    best_point = p
            
            # Only accept if within 5 seconds of target
            if best_point and best_diff < 5000:
                return best_point.price
            return None
        
        # ROC 30s
        price_30s_ago = get_price_at_offset(self.window_30s)
        if price_30s_ago and price_30s_ago > 0:
            self._last_roc_30s = (current_price - price_30s_ago) / price_30s_ago * 10000
        else:
            self._last_roc_30s = None
        
        # ROC 60s
        price_60s_ago = get_price_at_offset(self.window_60s)
        if price_60s_ago and price_60s_ago > 0:
            self._last_roc_60s = (current_price - price_60s_ago) / price_60s_ago * 10000
        else:
            self._last_roc_60s = None
    
    def snapshot(self) -> dict:
        """
        Get current ROC snapshot.
        
        Returns:
            Dictionary with ROC values and metadata
        """
        roc_30s = self._last_roc_30s
        roc_60s = self._last_roc_60s
        
        # Determine direction from 30s ROC
        if roc_30s is None:
            direction = "UNKNOWN"
            strength = 0.0
        elif abs(roc_30s) < ROC_FLAT_THRESHOLD_BPS:
            direction = "FLAT"
            strength = 0.0
        elif roc_30s > 0:
            direction = "UP"
            strength = min(1.0, abs(roc_30s) / ROC_STRONG_THRESHOLD_BPS)
        else:
            direction = "DOWN"
            strength = min(1.0, abs(roc_30s) / ROC_STRONG_THRESHOLD_BPS)
        
        # Check if ready (have enough history)
        ready = roc_30s is not None
        
        return {
            "roc_30s": round(roc_30s, 2) if roc_30s is not None else None,
            "roc_60s": round(roc_60s, 2) if roc_60s is not None else None,
            "direction": direction,
            "strength": round(strength, 3),
            "ready": ready,
            "history_count": len(self._history),
            "last_update_ms": self._last_update_ms,
        }
    
    def get_countertrend_signal(self, z_score: float, threshold_bps: float = 2.0) -> dict:
        """
        Check for countertrend signal.
        
        A countertrend signal occurs when:
        - Z > 0 (BTC above ref) but ROC < -threshold (falling) → SELL_SIGNAL (buy DOWN)
        - Z < 0 (BTC below ref) but ROC > +threshold (rising) → BUY_SIGNAL (buy UP)
        
        Args:
            z_score: Current Z-score (BTC vs ref)
            threshold_bps: Minimum ROC magnitude for signal
        
        Returns:
            Dictionary with signal info
        """
        roc = self._last_roc_30s
        
        if roc is None:
            return {
                "signal": "NONE",
                "reason": "roc_not_ready",
                "z_score": z_score,
                "roc_30s": None,
            }
        
        signal = "NONE"
        reason = "no_countertrend"
        
        # Countertrend DOWN: Z > 0 but falling
        if z_score > 0 and roc < -threshold_bps:
            signal = "BUY_DOWN"
            reason = f"z_positive({z_score:.3f})_but_falling({roc:.1f}bps)"
        
        # Countertrend UP: Z < 0 but rising
        elif z_score < 0 and roc > threshold_bps:
            signal = "BUY_UP"
            reason = f"z_negative({z_score:.3f})_but_rising({roc:.1f}bps)"
        
        return {
            "signal": signal,
            "reason": reason,
            "z_score": round(z_score, 3),
            "roc_30s": round(roc, 2),
            "threshold_bps": threshold_bps,
        }
