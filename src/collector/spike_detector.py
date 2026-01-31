"""
Binance Spike Detector (5-second)
=================================

Detects sudden price spikes on Binance to trigger accumulation signals.

Why this exists:
    When BTC drops sharply in 5 seconds, it may be a good time to
    accumulate UP tokens (buying the dip). Conversely, sharp rises
    may signal accumulation of DOWN tokens.
    
    We detect spikes by:
    1. Computing 5-second returns
    2. Computing z-score vs rolling history
    3. Flagging when |z| > threshold

Logic:
    - ret_5s = (close - open) / open for each 5s bar
    - z_ret_5s = (ret_5s - mean(history)) / std(history)
    - down_spike_5s = z_ret_5s < -threshold (sharp drop → buy UP)
    - up_spike_5s = z_ret_5s > +threshold (sharp rise → buy DOWN)

Usage:
    detector = BinanceSpikeDetector(lookback=60, threshold=2.0)
    
    # On each 5s bar:
    result = detector.update(ret_5s)
    # result = {"z_ret_5s": -2.5, "down_spike_5s": True, "up_spike_5s": False}

Configuration:
    lookback: Number of 5s bars to keep for statistics (default 60 = 5 min)
    threshold: Z-score threshold for spike detection (default 2.0)
"""

import logging
import math
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_LOOKBACK = 60  # 60 * 5s = 5 minutes
DEFAULT_THRESHOLD = 2.0  # 2 standard deviations
MIN_HISTORY = 12  # Need at least 1 minute of data


class BinanceSpikeDetector:
    """
    Detects 5-second price spikes using z-score analysis.
    
    Maintains rolling history of 5s returns and computes z-scores
    to identify statistically significant moves.
    
    Attributes:
        lookback: Number of returns to keep in history
        threshold: Z-score threshold for spike detection
    """
    
    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Initialize spike detector.
        
        Args:
            lookback: Number of 5s bars for rolling statistics
            threshold: Z-score threshold for spike flags
        """
        self.lookback = lookback
        self.threshold = threshold
        
        # Rolling history of 5s returns
        self._returns: deque[float] = deque(maxlen=lookback)
        
        # Latest values
        self._latest_ret: Optional[float] = None
        self._latest_z: Optional[float] = None
        self._down_spike: bool = False
        self._up_spike: bool = False
        
        # Stats
        self._update_count = 0
        
        logger.info(
            "spike_detector_initialized",
            extra={"lookback": lookback, "threshold": threshold},
        )
    
    def update(self, ret_5s: float) -> dict:
        """
        Update with new 5-second return and compute spike flags.
        
        Args:
            ret_5s: 5-second return (close - open) / open
        
        Returns:
            Dictionary with:
                - ret_5s_bps: Return in basis points
                - z_ret_5s: Z-score of the return
                - down_spike_5s: True if sharp downward move
                - up_spike_5s: True if sharp upward move
        
        Example:
            >>> detector.update(-0.002)  # -0.2% move
            {"ret_5s_bps": -20.0, "z_ret_5s": -2.3, "down_spike_5s": True, ...}
        """
        self._update_count += 1
        self._latest_ret = ret_5s
        
        # Compute z-score before adding to history
        z_score = self._compute_z_score(ret_5s)
        self._latest_z = z_score
        
        # Add to history
        self._returns.append(ret_5s)
        
        # Determine spike flags
        self._down_spike = False
        self._up_spike = False
        
        if z_score is not None:
            if z_score < -self.threshold:
                self._down_spike = True
                logger.info(
                    "spike_detected_down",
                    extra={"z_score": z_score, "ret_bps": ret_5s * 10000},
                )
            elif z_score > self.threshold:
                self._up_spike = True
                logger.info(
                    "spike_detected_up",
                    extra={"z_score": z_score, "ret_bps": ret_5s * 10000},
                )
        
        return self.snapshot()
    
    def _compute_z_score(self, ret: float) -> Optional[float]:
        """
        Compute z-score of return vs rolling history.
        
        z = (ret - mean) / std
        
        Args:
            ret: Return to compute z-score for
        
        Returns:
            Z-score or None if not enough history
        """
        if len(self._returns) < MIN_HISTORY:
            return None
        
        # Compute rolling mean and std
        n = len(self._returns)
        mean = sum(self._returns) / n
        
        variance = sum((r - mean) ** 2 for r in self._returns) / n
        std = math.sqrt(variance) if variance > 0 else 0
        
        if std < 1e-10:
            # No variance - can't compute z-score
            return None
        
        return (ret - mean) / std
    
    def snapshot(self) -> dict:
        """
        Get current spike detection state.
        
        Returns:
            Dictionary with current spike information
        """
        ret_bps = None
        if self._latest_ret is not None:
            ret_bps = round(self._latest_ret * 10000, 2)
        
        return {
            "ret_5s_bps": ret_bps,
            "z_ret_5s": round(self._latest_z, 2) if self._latest_z is not None else None,
            "down_spike_5s": self._down_spike,
            "up_spike_5s": self._up_spike,
            "history_len": len(self._returns),
            "ready": len(self._returns) >= MIN_HISTORY,
        }
    
    def is_ready(self) -> bool:
        """Check if detector has enough history."""
        return len(self._returns) >= MIN_HISTORY
    
    def get_stats(self) -> dict:
        """Get rolling statistics for debugging."""
        if len(self._returns) < MIN_HISTORY:
            return {"ready": False}
        
        n = len(self._returns)
        mean = sum(self._returns) / n
        variance = sum((r - mean) ** 2 for r in self._returns) / n
        std = math.sqrt(variance)
        
        return {
            "mean_bps": round(mean * 10000, 2),
            "std_bps": round(std * 10000, 2),
            "n": n,
            "threshold": self.threshold,
        }
