"""
Polymarket Dip Detector
=======================

Detects dips in Polymarket UP/DOWN token prices for accumulation signals.

Why this exists:
    When UP token price dips below its recent high, it may be a buying
    opportunity (if fundamentals haven't changed). Similarly for DOWN.
    
    We track:
    - Rolling max of mid price over 60 seconds
    - Dip = (current_mid - max_60s) / max_60s * 10000 (in bps)
    - Flag when dip exceeds threshold (e.g., -80 bps)

Logic:
    - max_60s_up = max(UP mid prices over last 60s)
    - dip_bps_up = (mid_up - max_60s_up) / max_60s_up * 10000
    - up_dip = dip_bps_up < -threshold (price dropped from recent high)

Usage:
    detector = PolymarketDipDetector(window_sec=60, threshold_bps=80)
    
    # On each Polymarket update:
    detector.update_up(mid_up, ts_ms)
    detector.update_down(mid_down, ts_ms)
    
    result = detector.snapshot()
    # result = {"up_dip_bps": -120, "up_dip": True, ...}

Configuration:
    window_sec: Rolling window for max tracking (default 60s)
    threshold_bps: Minimum dip in bps to trigger flag (default 80)
"""

import logging
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_WINDOW_SEC = 60  # 60 second rolling window
DEFAULT_THRESHOLD_BPS = 80  # 80 bps = 0.8% dip


class RollingMaxTracker:
    """
    Tracks rolling maximum of a value over a time window.
    
    Maintains a deque of (timestamp, value) pairs and computes
    the maximum value within the time window.
    """
    
    def __init__(self, window_ms: int):
        """
        Initialize tracker.
        
        Args:
            window_ms: Window size in milliseconds
        """
        self.window_ms = window_ms
        self._history: deque[tuple[int, float]] = deque()
        self._current_max: Optional[float] = None
    
    def update(self, value: float, ts_ms: int) -> float:
        """
        Update with new value and return current rolling max.
        
        Args:
            value: New value to add
            ts_ms: Timestamp in milliseconds
        
        Returns:
            Rolling maximum over the window
        """
        # Remove old entries outside window
        cutoff = ts_ms - self.window_ms
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()
        
        # Add new entry
        self._history.append((ts_ms, value))
        
        # Recompute max (could be optimized with monotonic deque)
        self._current_max = max(v for _, v in self._history)
        
        return self._current_max
    
    def get_max(self) -> Optional[float]:
        """Get current rolling maximum."""
        return self._current_max
    
    def get_count(self) -> int:
        """Get number of values in window."""
        return len(self._history)


class PolymarketDipDetector:
    """
    Detects dips in Polymarket UP/DOWN token prices.
    
    Tracks rolling maximum of each token's mid price and flags
    when current price drops significantly below recent high.
    
    Attributes:
        window_sec: Rolling window for max tracking
        threshold_bps: Minimum dip to trigger flag
    """
    
    def __init__(
        self,
        window_sec: float = DEFAULT_WINDOW_SEC,
        threshold_bps: float = DEFAULT_THRESHOLD_BPS,
    ):
        """
        Initialize dip detector.
        
        Args:
            window_sec: Rolling window in seconds
            threshold_bps: Dip threshold in basis points
        """
        self.window_sec = window_sec
        self.threshold_bps = threshold_bps
        window_ms = int(window_sec * 1000)
        
        # Separate trackers for UP and DOWN
        self._up_max_tracker = RollingMaxTracker(window_ms)
        self._down_max_tracker = RollingMaxTracker(window_ms)
        
        # Latest values
        self._up_mid: Optional[float] = None
        self._down_mid: Optional[float] = None
        self._up_max: Optional[float] = None
        self._down_max: Optional[float] = None
        
        # Computed dips
        self._up_dip_bps: Optional[float] = None
        self._down_dip_bps: Optional[float] = None
        self._up_dip: bool = False
        self._down_dip: bool = False
        
        logger.info(
            "pm_dip_detector_initialized",
            extra={"window_sec": window_sec, "threshold_bps": threshold_bps},
        )
    
    def update_up(self, mid: float, ts_ms: int) -> None:
        """
        Update UP token price.
        
        Args:
            mid: Current UP mid price
            ts_ms: Timestamp in milliseconds
        """
        if mid is None or mid <= 0:
            return
        
        self._up_mid = mid
        self._up_max = self._up_max_tracker.update(mid, ts_ms)
        
        # Compute dip
        self._up_dip_bps = self._compute_dip_bps(mid, self._up_max)
        self._up_dip = self._up_dip_bps is not None and self._up_dip_bps < -self.threshold_bps
        
        if self._up_dip:
            logger.info(
                "pm_dip_detected_up",
                extra={"mid": mid, "max_60s": self._up_max, "dip_bps": self._up_dip_bps},
            )
    
    def update_down(self, mid: float, ts_ms: int) -> None:
        """
        Update DOWN token price.
        
        Args:
            mid: Current DOWN mid price
            ts_ms: Timestamp in milliseconds
        """
        if mid is None or mid <= 0:
            return
        
        self._down_mid = mid
        self._down_max = self._down_max_tracker.update(mid, ts_ms)
        
        # Compute dip
        self._down_dip_bps = self._compute_dip_bps(mid, self._down_max)
        self._down_dip = self._down_dip_bps is not None and self._down_dip_bps < -self.threshold_bps
        
        if self._down_dip:
            logger.info(
                "pm_dip_detected_down",
                extra={"mid": mid, "max_60s": self._down_max, "dip_bps": self._down_dip_bps},
            )
    
    def update_both(self, up_mid: Optional[float], down_mid: Optional[float], ts_ms: int) -> None:
        """
        Update both UP and DOWN prices.
        
        Args:
            up_mid: Current UP mid price
            down_mid: Current DOWN mid price
            ts_ms: Timestamp in milliseconds
        """
        if up_mid is not None:
            self.update_up(up_mid, ts_ms)
        if down_mid is not None:
            self.update_down(down_mid, ts_ms)
    
    def _compute_dip_bps(self, current: float, max_val: Optional[float]) -> Optional[float]:
        """
        Compute dip in basis points.
        
        dip_bps = (current - max) / max * 10000
        
        Negative value means price dropped from max.
        """
        if max_val is None or max_val <= 0:
            return None
        
        return (current - max_val) / max_val * 10000
    
    def snapshot(self) -> dict:
        """
        Get current dip detection state.
        
        Returns:
            Dictionary with dip information for both sides
        """
        return {
            "up_mid": self._up_mid,
            "up_max_60s": round(self._up_max, 4) if self._up_max else None,
            "up_dip_bps": round(self._up_dip_bps, 1) if self._up_dip_bps is not None else None,
            "up_dip": self._up_dip,
            "down_mid": self._down_mid,
            "down_max_60s": round(self._down_max, 4) if self._down_max else None,
            "down_dip_bps": round(self._down_dip_bps, 1) if self._down_dip_bps is not None else None,
            "down_dip": self._down_dip,
            "up_history_count": self._up_max_tracker.get_count(),
            "down_history_count": self._down_max_tracker.get_count(),
        }
