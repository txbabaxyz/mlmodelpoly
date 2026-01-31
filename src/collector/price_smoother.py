"""
Price Smoother (S1)
===================

Time-aware EMA smoother for BTC price.

Provides S_now_smooth alongside S_now_raw to reduce noise in fair value calculations.

The EMA uses time-based alpha: alpha = 1 - exp(-dt / tau)
This ensures correct smoothing regardless of update frequency.

Usage:
    smoother = EmaSmoother(ema_sec=20.0)
    
    # On each price update
    smooth_price = smoother.update(raw_price, ts_ms)
    
    # Get snapshot
    snap = smoother.snapshot()
    # {"S_now_smooth": 89500.5, "smooth_last_update_ms": ..., "ema_sec": 20.0}

Configuration:
    PRICE_EMA_SEC: EMA time constant in seconds (default: 20.0)
    - Lower = faster response, more noise
    - Higher = slower response, smoother
"""

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EmaSmoother:
    """
    Time-aware Exponential Moving Average smoother.
    
    Uses time-based alpha calculation to handle irregular update intervals:
        alpha = 1 - exp(-dt / tau)
    
    This ensures that:
    - If dt << tau: alpha ≈ dt/tau (small update)
    - If dt >> tau: alpha ≈ 1 (full reset to new value)
    - Regardless of update frequency, smoothing behaves consistently
    
    Attributes:
        ema_sec: Time constant (tau) in seconds
        value: Current smoothed value
        last_ts_ms: Timestamp of last update
    """
    ema_sec: float
    value: Optional[float] = field(default=None)
    last_ts_ms: Optional[int] = field(default=None)
    
    def update(self, x: float, ts_ms: int) -> float:
        """
        Update EMA with new value.
        
        Args:
            x: New raw value
            ts_ms: Timestamp in milliseconds
            
        Returns:
            Updated smoothed value
        """
        # First update: initialize to raw value
        if self.value is None or self.last_ts_ms is None:
            self.value = float(x)
            self.last_ts_ms = int(ts_ms)
            return self.value
        
        # Compute time delta
        dt_sec = max(1, ts_ms - self.last_ts_ms) / 1000.0
        
        # Time-based alpha: alpha = 1 - exp(-dt / tau)
        # This gives correct weighting regardless of update frequency
        alpha = 1.0 - math.exp(-dt_sec / max(1e-6, self.ema_sec))
        
        # Update EMA: new = (1 - alpha) * old + alpha * x
        self.value = (1.0 - alpha) * self.value + alpha * float(x)
        self.last_ts_ms = int(ts_ms)
        
        return self.value
    
    def get_value(self) -> Optional[float]:
        """Get current smoothed value."""
        return self.value
    
    def snapshot(self) -> dict:
        """
        Get snapshot of smoother state.
        
        Returns:
            Dictionary with S_now_smooth, last_update_ms, ema_sec
        """
        return {
            "S_now_smooth": round(self.value, 2) if self.value is not None else None,
            "smooth_last_update_ms": self.last_ts_ms,
            "ema_sec": self.ema_sec,
        }
    
    def reset(self) -> None:
        """Reset smoother state."""
        self.value = None
        self.last_ts_ms = None
