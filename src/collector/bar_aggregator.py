"""
Bar Aggregator Module
=====================

Aggregates normalized aggTrade events into OHLCV+delta bars.
Bars are closed strictly on time grid (e.g., every 5 seconds).

Features:
- Event time based (not local time)
- Handles late events gracefully
- No empty bars (only bars with trades)
- Delta volume calculation (buy - sell)
"""

import logging
from typing import Optional

from collector.types import NormalizedEvent, SCHEMA_VERSION
from collector.utils_time import floor_ts

logger = logging.getLogger(__name__)


class BarAggregator:
    """
    Aggregates aggTrade events into time-based OHLCV+delta bars.
    
    Only processes events matching the configured market and type="aggTrade".
    Bars close when an event arrives with a timestamp in the next bucket.
    
    Usage:
        aggregator = BarAggregator(tf_seconds=5, market="futures", symbol="BTCUSDT")
        
        for event in events:
            bar = aggregator.update(event)
            if bar:
                # Bar closed, process it
                print(bar)
    """
    
    def __init__(self, tf_seconds: int, market: str, symbol: str):
        """
        Initialize bar aggregator.
        
        Args:
            tf_seconds: Timeframe in seconds (e.g., 5 for 5s bars)
            market: Market to aggregate ("futures" or "spot")
            symbol: Symbol to aggregate (uppercase, e.g., "BTCUSDT")
        """
        self.tf_seconds = tf_seconds
        self.tf_ms = tf_seconds * 1000
        # Human-readable timeframe label
        if tf_seconds == 60:
            self.tf_label = "1m"
        elif tf_seconds >= 60 and tf_seconds % 60 == 0:
            self.tf_label = f"{tf_seconds // 60}m"
        else:
            self.tf_label = f"{tf_seconds}s"
        self.market = market
        self.symbol = symbol
        
        # Current bar state
        self._current_bucket_ms: Optional[int] = None
        self._open: Optional[float] = None
        self._high: Optional[float] = None
        self._low: Optional[float] = None
        self._close: Optional[float] = None
        self._volume_total: float = 0.0
        self._trades_count: int = 0
        self._buy_vol: float = 0.0
        self._sell_vol: float = 0.0
        
        # Stats
        self._bars_closed: int = 0
        self._events_processed: int = 0
        self._late_events_ignored: int = 0
    
    def update(self, event: NormalizedEvent) -> Optional[dict]:
        """
        Process a normalized event and potentially close a bar.
        
        Args:
            event: Normalized event (must be aggTrade for aggregation)
        
        Returns:
            Closed bar dict if bar was closed, None otherwise.
        """
        # Filter: only process matching market and aggTrade type
        if event.market != self.market:
            return None
        
        if event.type != "aggTrade":
            return None
        
        # Extract data
        ts_event_ms = event.ts_event_ms
        data = event.data
        price = data.get("price")
        qty = data.get("qty")
        side_aggressor = data.get("side_aggressor")
        
        # Validate required fields
        if ts_event_ms is None or price is None or qty is None:
            logger.warning(
                "bar_aggregator_invalid_event",
                extra={
                    "market": self.market,
                    "tf": self.tf_label,
                    "reason": "missing_fields",
                },
            )
            return None
        
        # Calculate bucket for this event
        event_bucket_ms = floor_ts(ts_event_ms, self.tf_ms)
        
        # Check for late event (belongs to past bucket)
        if self._current_bucket_ms is not None and event_bucket_ms < self._current_bucket_ms:
            self._late_events_ignored += 1
            logger.warning(
                "bar_aggregator_late_event",
                extra={
                    "market": self.market,
                    "tf": self.tf_label,
                    "event_bucket_ms": event_bucket_ms,
                    "current_bucket_ms": self._current_bucket_ms,
                    "lag_buckets": (self._current_bucket_ms - event_bucket_ms) // self.tf_ms,
                },
            )
            return None
        
        self._events_processed += 1
        closed_bar: Optional[dict] = None
        
        # Check if we need to close current bar and start new one
        if self._current_bucket_ms is not None and event_bucket_ms > self._current_bucket_ms:
            # Close current bar
            closed_bar = self._close_bar()
        
        # Initialize new bar if needed
        if self._current_bucket_ms is None or event_bucket_ms > self._current_bucket_ms:
            self._start_new_bar(event_bucket_ms)
        
        # Aggregate this event into current bar
        self._aggregate_trade(price, qty, side_aggressor)
        
        return closed_bar
    
    def _start_new_bar(self, bucket_ms: int) -> None:
        """Start a new bar for the given bucket."""
        self._current_bucket_ms = bucket_ms
        self._open = None
        self._high = None
        self._low = None
        self._close = None
        self._volume_total = 0.0
        self._trades_count = 0
        self._buy_vol = 0.0
        self._sell_vol = 0.0
    
    def _aggregate_trade(
        self,
        price: float,
        qty: float,
        side_aggressor: Optional[str],
    ) -> None:
        """Add a trade to the current bar."""
        # OHLC
        if self._open is None:
            self._open = price
        
        if self._high is None or price > self._high:
            self._high = price
        
        if self._low is None or price < self._low:
            self._low = price
        
        self._close = price
        
        # Volume
        self._volume_total += qty
        self._trades_count += 1
        
        # Buy/Sell volume
        if side_aggressor == "buy":
            self._buy_vol += qty
        elif side_aggressor == "sell":
            self._sell_vol += qty
    
    def _close_bar(self) -> Optional[dict]:
        """
        Close current bar and return it.
        
        Returns:
            Bar dict if there were trades, None if empty.
        """
        # Don't create empty bars
        if self._trades_count == 0 or self._current_bucket_ms is None:
            return None
        
        bar = {
            "schema_version": SCHEMA_VERSION,
            "market": self.market,
            "symbol": self.symbol,
            "tf": self.tf_label,
            
            "t_open_ms": self._current_bucket_ms,
            "t_close_ms": self._current_bucket_ms + self.tf_ms,
            
            "open": self._open,
            "high": self._high,
            "low": self._low,
            "close": self._close,
            
            "volume_total": round(self._volume_total, 8),
            "trades_count": self._trades_count,
            
            "buy_vol": round(self._buy_vol, 8),
            "sell_vol": round(self._sell_vol, 8),
            "delta_vol": round(self._buy_vol - self._sell_vol, 8),
        }
        
        self._bars_closed += 1
        
        return bar
    
    def flush(self) -> Optional[dict]:
        """
        Force close current bar (e.g., on shutdown).
        
        Returns:
            Current bar if any trades, None otherwise.
        """
        bar = self._close_bar()
        if bar:
            self._start_new_bar(self._current_bucket_ms + self.tf_ms)
        return bar
    
    @property
    def bars_closed(self) -> int:
        """Number of bars closed since start."""
        return self._bars_closed
    
    @property
    def events_processed(self) -> int:
        """Number of aggTrade events processed."""
        return self._events_processed
    
    @property
    def late_events_ignored(self) -> int:
        """Number of late events ignored."""
        return self._late_events_ignored
    
    @property
    def current_bucket_ms(self) -> Optional[int]:
        """Current active bucket timestamp."""
        return self._current_bucket_ms
    
    def get_stats(self) -> dict:
        """Get aggregator statistics."""
        return {
            "market": self.market,
            "tf": self.tf_label,
            "bars_closed": self._bars_closed,
            "events_processed": self._events_processed,
            "late_events_ignored": self._late_events_ignored,
            "current_bucket_ms": self._current_bucket_ms,
            "current_trades_count": self._trades_count,
        }
