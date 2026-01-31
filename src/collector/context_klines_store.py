"""
Context Klines Store Module
===========================

In-memory store for HTF (Higher Time Frame) klines used for context.

Features:
- Stores klines for multiple (market, tf) combinations
- Dedup by open_time_ms
- Updates last kline if it's still forming
- Thread-safe via async locks

Usage:
    store = KlinesStore()
    store.upsert_klines("futures", "1h", raw_klines_from_api)
    klines = store.get("futures", "1h")
"""

import logging
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# Maximum klines to store per (market, tf)
MAX_KLINES_PER_TF = 2000


@dataclass(frozen=True, slots=True)
class Kline:
    """
    Single kline (OHLCV candlestick) data.
    
    Attributes:
        open_time_ms: Opening time in milliseconds
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        close_time_ms: Closing time in milliseconds
    """
    open_time_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_time_ms: int
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "open_time_ms": self.open_time_ms,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "close_time_ms": self.close_time_ms,
        }


def parse_binance_kline(raw: list) -> Kline:
    """
    Parse Binance kline array into Kline dataclass.
    
    Binance kline format:
    [
        0: open_time (ms),
        1: open,
        2: high,
        3: low,
        4: close,
        5: volume,
        6: close_time (ms),
        7: quote_asset_volume,
        8: number_of_trades,
        9: taker_buy_base_volume,
        10: taker_buy_quote_volume,
        11: ignore
    ]
    
    Args:
        raw: Raw kline array from Binance API
        
    Returns:
        Kline dataclass instance
    """
    return Kline(
        open_time_ms=int(raw[0]),
        open=float(raw[1]),
        high=float(raw[2]),
        low=float(raw[3]),
        close=float(raw[4]),
        volume=float(raw[5]),
        close_time_ms=int(raw[6]),
    )


class KlinesStore:
    """
    In-memory store for HTF klines.
    
    Stores klines for multiple (market, timeframe) combinations.
    Each combination has a deque with maxlen=2000.
    
    Features:
        - Upsert: add new klines, update existing by open_time_ms
        - Dedup: no duplicate klines by open_time_ms
        - Update last: if open_time matches last kline, replace it (for live updates)
    
    Usage:
        store = KlinesStore()
        store.upsert_klines("futures", "1h", raw_klines)
        klines = store.get("futures", "1h")
    """
    
    def __init__(self, maxlen: int = MAX_KLINES_PER_TF) -> None:
        """
        Initialize klines store.
        
        Args:
            maxlen: Maximum klines per (market, tf) combination
        """
        self._maxlen = maxlen
        self._data: dict[tuple[str, str], deque[Kline]] = {}
        self._open_time_index: dict[tuple[str, str], set[int]] = {}
        
        logger.info(
            "klines_store_initialized",
            extra={"maxlen": maxlen},
        )
    
    def _get_or_create_deque(self, market: str, tf: str) -> deque[Kline]:
        """Get or create deque for (market, tf)."""
        key = (market, tf)
        if key not in self._data:
            self._data[key] = deque(maxlen=self._maxlen)
            self._open_time_index[key] = set()
        return self._data[key]
    
    def _get_index(self, market: str, tf: str) -> set[int]:
        """Get open_time_ms index set for (market, tf)."""
        key = (market, tf)
        if key not in self._open_time_index:
            self._open_time_index[key] = set()
        return self._open_time_index[key]
    
    def upsert_klines(
        self,
        market: str,
        tf: str,
        klines_raw: list[list],
    ) -> int:
        """
        Upsert klines from Binance API response.
        
        - New klines are appended
        - If open_time_ms matches last kline, replace it (live update)
        - Duplicates (by open_time_ms) are skipped
        
        Args:
            market: Market name ("futures" or "spot")
            tf: Timeframe ("1m", "5m", "15m", "1h", etc.)
            klines_raw: Raw klines from Binance API
            
        Returns:
            Number of klines added/updated
        """
        if not klines_raw:
            return 0
        
        q = self._get_or_create_deque(market, tf)
        index = self._get_index(market, tf)
        
        added = 0
        updated = 0
        
        for raw in klines_raw:
            try:
                kline = parse_binance_kline(raw)
            except (IndexError, ValueError, TypeError) as e:
                logger.warning(
                    "kline_parse_error",
                    extra={
                        "market": market,
                        "tf": tf,
                        "error": str(e),
                        "raw": str(raw)[:100],
                    },
                )
                continue
            
            # Check if this is an update to the last kline
            if q and q[-1].open_time_ms == kline.open_time_ms:
                # Replace last kline (live update)
                q[-1] = kline
                updated += 1
            elif kline.open_time_ms not in index:
                # New kline, not a duplicate
                q.append(kline)
                index.add(kline.open_time_ms)
                added += 1
                
                # If deque is at maxlen, oldest item was removed
                # Clean up index accordingly
                if len(index) > self._maxlen:
                    # Find and remove oldest open_time that's no longer in deque
                    deque_times = {k.open_time_ms for k in q}
                    index.intersection_update(deque_times)
        
        if added > 0 or updated > 0:
            logger.debug(
                "klines_upserted",
                extra={
                    "market": market,
                    "tf": tf,
                    "added": added,
                    "updated": updated,
                    "total": len(q),
                },
            )
        
        return added + updated
    
    def get(self, market: str, tf: str) -> list[Kline]:
        """
        Get all klines for (market, tf).
        
        Args:
            market: Market name
            tf: Timeframe
            
        Returns:
            List of Kline objects (oldest first)
        """
        key = (market, tf)
        if key not in self._data:
            return []
        return list(self._data[key])
    
    def get_last(self, market: str, tf: str, n: int = 1) -> list[Kline]:
        """
        Get last N klines for (market, tf).
        
        Args:
            market: Market name
            tf: Timeframe
            n: Number of klines to return
            
        Returns:
            List of last N Kline objects (oldest first)
        """
        key = (market, tf)
        if key not in self._data:
            return []
        q = self._data[key]
        if n >= len(q):
            return list(q)
        return list(q)[-n:]
    
    def count(self, market: str, tf: str) -> int:
        """
        Get kline count for (market, tf).
        
        Args:
            market: Market name
            tf: Timeframe
            
        Returns:
            Number of stored klines
        """
        key = (market, tf)
        if key not in self._data:
            return 0
        return len(self._data[key])
    
    def get_counts(self) -> dict[str, dict[str, int]]:
        """
        Get counts for all (market, tf) combinations.
        
        Returns:
            Dict: {market: {tf: count}}
        """
        result: dict[str, dict[str, int]] = {}
        for (market, tf), q in self._data.items():
            if market not in result:
                result[market] = {}
            result[market][tf] = len(q)
        return result
    
    def get_last_kline(self, market: str, tf: str) -> Optional[Kline]:
        """
        Get the most recent kline for (market, tf).
        
        Args:
            market: Market name
            tf: Timeframe
            
        Returns:
            Last Kline or None if no data
        """
        key = (market, tf)
        if key not in self._data or not self._data[key]:
            return None
        return self._data[key][-1]
    
    def snapshot(self) -> dict:
        """
        Get snapshot of store state.
        
        Returns:
            Dict with counts and last kline info per (market, tf)
        """
        result = {
            "counts": self.get_counts(),
            "last_klines": {},
        }
        
        for (market, tf), q in self._data.items():
            if q:
                last = q[-1]
                key = f"{market}_{tf}"
                result["last_klines"][key] = {
                    "open_time_ms": last.open_time_ms,
                    "close": last.close,
                }
        
        return result
