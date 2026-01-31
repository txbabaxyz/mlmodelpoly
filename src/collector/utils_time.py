"""
Time Utilities Module
=====================

Utility functions for working with timestamps.
All timestamps are in milliseconds (Binance standard).
"""

import time


def now_ms() -> int:
    """
    Get current timestamp in milliseconds.
    
    Returns:
        Current Unix timestamp in milliseconds.
    
    Example:
        >>> ts = now_ms()
        >>> print(ts)  # 1706356800000
    """
    return int(time.time() * 1000)


def floor_ts(ts_ms: int, interval_ms: int) -> int:
    """
    Floor timestamp to the nearest interval boundary.
    
    Rounds down the timestamp to the start of the interval.
    Useful for aggregating data into time bars.
    
    Args:
        ts_ms: Timestamp in milliseconds.
        interval_ms: Interval size in milliseconds.
        
    Returns:
        Floored timestamp in milliseconds.
    
    Example:
        >>> # Floor to 1-minute bars (60000 ms)
        >>> floor_ts(1706356845123, 60000)
        1706356800000
        
        >>> # Floor to 5-second bars (5000 ms)
        >>> floor_ts(1706356847500, 5000)
        1706356845000
    """
    if interval_ms <= 0:
        raise ValueError("interval_ms must be positive")
    return (ts_ms // interval_ms) * interval_ms


def ms_to_sec(ts_ms: int) -> float:
    """
    Convert milliseconds to seconds.
    
    Args:
        ts_ms: Timestamp in milliseconds.
        
    Returns:
        Timestamp in seconds (float).
    """
    return ts_ms / 1000.0


def sec_to_ms(ts_sec: float) -> int:
    """
    Convert seconds to milliseconds.
    
    Args:
        ts_sec: Timestamp in seconds.
        
    Returns:
        Timestamp in milliseconds (int).
    """
    return int(ts_sec * 1000)


# Common interval constants (in milliseconds)
INTERVAL_1S = 1_000
INTERVAL_5S = 5_000
INTERVAL_10S = 10_000
INTERVAL_30S = 30_000
INTERVAL_1M = 60_000
INTERVAL_5M = 300_000
INTERVAL_15M = 900_000
INTERVAL_1H = 3_600_000
