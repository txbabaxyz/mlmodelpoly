"""
Market Reference Tracker
========================

Tracks the reference price (ref_px) at the start of each 15-minute window
and computes tau (time remaining until window end).

Why this exists:
    For Polymarket 15-minute prediction markets, we need to know:
    1. What was the BTC price at the start of the window? (ref_px)
    2. How much time is left in the window? (tau_sec)
    
    This allows us to:
    - Compare current price to ref_px to estimate UP/DOWN probability
    - Adjust our confidence based on tau (less time = less uncertainty)

Window Structure:
    - Windows are 15 minutes (900 seconds) aligned to UTC
    - window_id = floor(now_ms / 900000)
    - Example: window_id 1966212 → starts at 1769590800000 ms
    
Usage:
    tracker = MarketRefTracker()
    
    # In your loop:
    now_ms = utils_time.now_ms()
    current_price = 105000.0  # from Binance
    
    tracker.maybe_roll_window(now_ms)
    tracker.set_ref_if_missing(current_price, now_ms)
    
    snap = tracker.snapshot(now_ms)
    # snap = {
    #     "window_id": 1966212,
    #     "t0_ms": 1769590800000,
    #     "t_end_ms": 1769591700000,
    #     "ref_px": 104850.5,
    #     "tau_sec": 542.3
    # }

Configuration:
    WINDOW_LEN_SEC: Window length in seconds (default 900 = 15 minutes)
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default window length: 15 minutes = 900 seconds
DEFAULT_WINDOW_LEN_SEC = 900


class MarketRefTracker:
    """
    Tracks reference price at window start and time remaining (tau).
    
    The reference price (ref_px) is fixed at the first price observation
    after a new window starts. It does not change until the next window.
    
    Attributes:
        window_id: Current window identifier (floor(now_ms / window_len_ms))
        t0_ms: Window start timestamp in milliseconds
        t_end_ms: Window end timestamp in milliseconds
        ref_px: Reference price fixed at window start
        window_len_sec: Window length in seconds (default 900)
    """
    
    def __init__(self, window_len_sec: int = DEFAULT_WINDOW_LEN_SEC):
        """
        Initialize the tracker.
        
        Args:
            window_len_sec: Window length in seconds (default 900 = 15 min)
        """
        self.window_len_sec = window_len_sec
        self.window_len_ms = window_len_sec * 1000
        
        # Current window state
        self.window_id: Optional[int] = None
        self.t0_ms: Optional[int] = None
        self.t_end_ms: Optional[int] = None
        self.ref_px: Optional[float] = None
        
        # Track when ref_px was set
        self._ref_px_set_at_ms: Optional[int] = None
        
        logger.info(
            "market_ref_tracker_initialized",
            extra={"window_len_sec": window_len_sec},
        )
    
    def maybe_roll_window(self, now_ms: int) -> bool:
        """
        Check if we need to roll to a new window and do so if needed.
        
        Window rolling happens when:
            floor(now_ms / window_len_ms) != current window_id
        
        When rolling:
            - Reset ref_px to None (will be set on next price update)
            - Update t0_ms and t_end_ms
            - Update window_id
        
        Args:
            now_ms: Current timestamp in milliseconds
        
        Returns:
            True if window was rolled, False otherwise
        
        Example:
            >>> tracker = MarketRefTracker()
            >>> tracker.maybe_roll_window(1769590800000)  # Start of window
            True
            >>> tracker.maybe_roll_window(1769590800100)  # Same window
            False
            >>> tracker.maybe_roll_window(1769591700000)  # Next window
            True
        """
        new_window_id = now_ms // self.window_len_ms
        
        if new_window_id == self.window_id:
            return False
        
        # Roll to new window
        old_window_id = self.window_id
        old_ref_px = self.ref_px
        
        self.window_id = new_window_id
        self.t0_ms = new_window_id * self.window_len_ms
        self.t_end_ms = self.t0_ms + self.window_len_ms
        self.ref_px = None
        self._ref_px_set_at_ms = None
        
        logger.info(
            "market_ref_window_rolled",
            extra={
                "old_window_id": old_window_id,
                "new_window_id": new_window_id,
                "old_ref_px": old_ref_px,
                "t0_ms": self.t0_ms,
                "t_end_ms": self.t_end_ms,
            },
        )
        
        return True
    
    def set_ref_if_missing(self, ref_px: float, now_ms: int) -> bool:
        """
        Set the reference price if not already set for current window.
        
        The ref_px is only set once per window - the first price observation
        after the window starts becomes the reference for the entire window.
        
        Args:
            ref_px: Reference price (typically Binance futures mark/last price)
            now_ms: Current timestamp in milliseconds
        
        Returns:
            True if ref_px was set, False if already set
        
        Example:
            >>> tracker.maybe_roll_window(now_ms)
            >>> tracker.set_ref_if_missing(105000.0, now_ms)  # Sets ref_px
            True
            >>> tracker.set_ref_if_missing(105100.0, now_ms)  # Already set
            False
        """
        if self.ref_px is not None:
            return False
        
        if self.window_id is None:
            # No window initialized yet
            return False
        
        self.ref_px = ref_px
        self._ref_px_set_at_ms = now_ms
        
        # Calculate delay from window start
        delay_ms = now_ms - self.t0_ms if self.t0_ms else 0
        
        logger.info(
            "market_ref_px_set",
            extra={
                "window_id": self.window_id,
                "ref_px": ref_px,
                "delay_from_t0_ms": delay_ms,
            },
        )
        
        return True
    
    def get_tau_sec(self, now_ms: int) -> Optional[float]:
        """
        Get time remaining until window end (tau) in seconds.
        
        Args:
            now_ms: Current timestamp in milliseconds
        
        Returns:
            Seconds remaining until window end, or None if no window
        
        Notes:
            - Returns 0 if window has ended (now_ms >= t_end_ms)
            - tau decreases as time passes within the window
        """
        if self.t_end_ms is None:
            return None
        
        remaining_ms = self.t_end_ms - now_ms
        return max(0.0, remaining_ms / 1000.0)
    
    def snapshot(self, now_ms: int) -> dict:
        """
        Get current tracker state as dictionary.
        
        Args:
            now_ms: Current timestamp in milliseconds
        
        Returns:
            Dictionary with:
                - window_id: Current window identifier
                - t0_ms: Window start timestamp
                - t_end_ms: Window end timestamp
                - ref_px: Reference price (or None if not set)
                - tau_sec: Seconds remaining in window
                - ref_px_age_ms: How long ago ref_px was set (or None)
        
        Example:
            >>> snap = tracker.snapshot(now_ms)
            >>> print(f"Window {snap['window_id']}, tau={snap['tau_sec']:.1f}s")
        """
        tau_sec = self.get_tau_sec(now_ms)
        
        # Calculate how long ago ref_px was set
        ref_px_age_ms = None
        if self._ref_px_set_at_ms is not None:
            ref_px_age_ms = now_ms - self._ref_px_set_at_ms
        
        return {
            "window_id": self.window_id,
            "t0_ms": self.t0_ms,
            "t_end_ms": self.t_end_ms,
            "ref_px": self.ref_px,
            "tau_sec": round(tau_sec, 1) if tau_sec is not None else None,
            "ref_px_age_ms": ref_px_age_ms,
        }
    
    def is_ready(self) -> bool:
        """
        Check if tracker has valid window and ref_px.
        
        Returns:
            True if both window_id and ref_px are set
        """
        return self.window_id is not None and self.ref_px is not None
