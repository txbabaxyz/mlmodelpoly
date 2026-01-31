"""
TAAPI Context Store
===================

In-memory storage for TAAPI indicator values.

Features:
- Stores indicators per timeframe
- Tracks last update timestamps
- Provides snapshot for features integration
- Thread-safe via simple dict operations

Usage:
    store = TaapiContextStore()
    store.update("1h", {"ema_50": 88000, "adx_14": 25})
    data = store.get("1h")
"""

import logging
from typing import Optional
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


class TaapiContextStore:
    """
    Storage for TAAPI indicator values.
    
    Stores indicator values per timeframe with timestamps.
    
    Structure:
        _data[tf] = {
            "ema_20": value,
            "ema_50": value,
            "rsi_14": value,
            "macd": {"valueMACD": ..., "valueMACDSignal": ..., "valueMACDHist": ...},
            ...
            "_ts_ms": timestamp_of_last_update,
        }
    """
    
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}
        self._ready: bool = False
        self._bootstrap_ts_ms: Optional[int] = None
        
        logger.info("taapi_context_store_initialized")
    
    def update(self, tf: str, indicators: dict) -> None:
        """
        Update indicators for a timeframe.
        
        Args:
            tf: Timeframe (1m, 5m, 15m, 1h)
            indicators: Dict of indicator_id -> value
        """
        ts = now_ms()
        
        # Merge with existing data or create new
        if tf not in self._data:
            self._data[tf] = {}
        
        for key, value in indicators.items():
            self._data[tf][key] = value
        
        self._data[tf]["_ts_ms"] = ts
        
        logger.debug(
            "taapi_store_updated",
            extra={
                "tf": tf,
                "indicators_count": len(indicators),
            },
        )
    
    def get(self, tf: str) -> Optional[dict]:
        """
        Get indicators for a timeframe.
        
        Args:
            tf: Timeframe
            
        Returns:
            Dict of indicators or None if not available
        """
        return self._data.get(tf)
    
    def get_value(self, tf: str, indicator_id: str) -> Optional[float]:
        """
        Get single indicator value.
        
        Args:
            tf: Timeframe
            indicator_id: Indicator ID (e.g., "ema_20", "rsi_14")
            
        Returns:
            Indicator value or None
        """
        tf_data = self._data.get(tf)
        if not tf_data:
            return None
        
        value = tf_data.get(indicator_id)
        
        # Handle simple values
        if isinstance(value, (int, float)):
            return value
        
        # Handle dict values (like MACD)
        if isinstance(value, dict):
            # Return primary value if exists
            return value.get("value")
        
        return None
    
    def get_age_sec(self, tf: str) -> Optional[float]:
        """
        Get age of data for timeframe in seconds.
        
        Args:
            tf: Timeframe
            
        Returns:
            Age in seconds or None if no data
        """
        tf_data = self._data.get(tf)
        if not tf_data:
            return None
        
        ts_ms = tf_data.get("_ts_ms")
        if not ts_ms:
            return None
        
        return (now_ms() - ts_ms) / 1000.0
    
    def mark_ready(self) -> None:
        """Mark store as ready (bootstrap complete)."""
        self._ready = True
        self._bootstrap_ts_ms = now_ms()
        logger.info("taapi_context_store_ready")
    
    def is_ready(self) -> bool:
        """Check if store is ready (bootstrap complete)."""
        return self._ready
    
    def get_timeframes(self) -> list[str]:
        """Get list of timeframes with data."""
        return list(self._data.keys())
    
    def snapshot(self) -> dict:
        """
        Get full snapshot for API/logging.
        
        Returns:
            Dict with all timeframes and metadata
        """
        ts = now_ms()
        
        result = {
            "ready": self._ready,
            "bootstrap_ts_ms": self._bootstrap_ts_ms,
            "timeframes": {},
        }
        
        for tf, data in self._data.items():
            tf_ts = data.get("_ts_ms", 0)
            age_sec = (ts - tf_ts) / 1000.0 if tf_ts else None
            
            # Extract indicator values (exclude internal fields)
            indicators = {
                k: v for k, v in data.items()
                if not k.startswith("_")
            }
            
            result["timeframes"][tf] = {
                "age_sec": round(age_sec, 1) if age_sec else None,
                "indicators_count": len(indicators),
                "indicators": indicators,
            }
        
        return result
    
    def get_summary(self) -> dict:
        """
        Get short summary for logging.
        
        Returns:
            Dict with ready status and indicator counts per TF
        """
        return {
            "ready": self._ready,
            "timeframes": {
                tf: {
                    "count": len([k for k in data.keys() if not k.startswith("_")]),
                    "age_sec": round(self.get_age_sec(tf), 1) if self.get_age_sec(tf) else None,
                }
                for tf, data in self._data.items()
            },
        }
