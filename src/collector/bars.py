"""
Bars Manager Module
===================

Centralized management of bar aggregators for multiple markets and timeframes.

Manages:
    - Futures: 5s, 15s, 1m bars
    - Spot: 5s, 15s, 1m bars

Provides unified interface for event processing and bar emission.
"""

import logging
from typing import Optional

from collector.bar_aggregator import BarAggregator
from collector.types import NormalizedEvent

logger = logging.getLogger(__name__)

# Timeframes to aggregate (in seconds)
TIMEFRAMES = [5, 15, 60]

# Markets to aggregate
MARKETS = ["futures", "spot"]


class BarsManager:
    """
    Centralized manager for all bar aggregators.
    
    Handles events and distributes them to appropriate aggregators
    based on market. Returns list of closed bars.
    
    Usage:
        manager = BarsManager(symbol="BTCUSDT")
        
        for event in events:
            closed_bars = manager.handle_event(event)
            for bar in closed_bars:
                # Process closed bar
                pass
    """
    
    def __init__(self, symbol: str):
        """
        Initialize bars manager with aggregators for all markets and timeframes.
        
        Args:
            symbol: Trading symbol (uppercase, e.g., "BTCUSDT")
        """
        self.symbol = symbol
        
        # Create aggregators for each (market, tf) combination
        # Key: (market, tf_seconds)
        # Value: BarAggregator
        self._aggregators: dict[tuple[str, int], BarAggregator] = {}
        
        for market in MARKETS:
            for tf_seconds in TIMEFRAMES:
                key = (market, tf_seconds)
                self._aggregators[key] = BarAggregator(
                    tf_seconds=tf_seconds,
                    market=market,
                    symbol=symbol,
                )
        
        logger.info(
            "bars_manager_initialized",
            extra={
                "symbol": symbol,
                "markets": MARKETS,
                "timeframes": TIMEFRAMES,
                "aggregators_count": len(self._aggregators),
            },
        )
    
    def handle_event(self, event: NormalizedEvent) -> list[dict]:
        """
        Process a normalized event through all relevant aggregators.
        
        Args:
            event: Normalized event (only aggTrade events are processed)
        
        Returns:
            List of closed bars (0 to N). Empty list if no bars closed.
        """
        # Only process aggTrade events
        if event.type != "aggTrade":
            return []
        
        market = event.market
        closed_bars: list[dict] = []
        
        # Get aggregators for this market
        for tf_seconds in TIMEFRAMES:
            key = (market, tf_seconds)
            aggregator = self._aggregators.get(key)
            
            if aggregator:
                bar = aggregator.update(event)
                if bar:
                    closed_bars.append(bar)
        
        return closed_bars
    
    def flush_all(self) -> list[dict]:
        """
        Force close all active bars (e.g., on shutdown).
        
        Returns:
            List of all flushed bars.
        """
        flushed_bars: list[dict] = []
        
        for key, aggregator in self._aggregators.items():
            bar = aggregator.flush()
            if bar:
                flushed_bars.append(bar)
        
        return flushed_bars
    
    def get_aggregator(self, market: str, tf_seconds: int) -> Optional[BarAggregator]:
        """
        Get specific aggregator.
        
        Args:
            market: Market name ("futures" or "spot")
            tf_seconds: Timeframe in seconds
        
        Returns:
            BarAggregator or None if not found.
        """
        return self._aggregators.get((market, tf_seconds))
    
    def get_stats(self) -> dict:
        """
        Get statistics for all aggregators.
        
        Returns:
            Dictionary with stats per (market, tf).
        """
        stats = {}
        
        for (market, tf_seconds), aggregator in self._aggregators.items():
            tf_label = aggregator.tf_label
            key = f"{market}_{tf_label}"
            stats[key] = {
                "bars_closed": aggregator.bars_closed,
                "events_processed": aggregator.events_processed,
                "late_events": aggregator.late_events_ignored,
            }
        
        return stats


class LastBarsStore:
    """
    Simple in-memory store for last closed bars.
    
    Used by HTTP API to expose last bars without heavy computation.
    """
    
    def __init__(self):
        """Initialize empty store."""
        # Key: (market, tf_label)
        # Value: bar dict
        self._bars: dict[tuple[str, str], dict] = {}
    
    def update(self, bar: dict) -> None:
        """
        Update store with a closed bar.
        
        Args:
            bar: Closed bar dict with market and tf fields.
        """
        market = bar.get("market")
        tf = bar.get("tf")
        
        if market and tf:
            self._bars[(market, tf)] = bar
    
    def get(self, market: str, tf: str) -> Optional[dict]:
        """
        Get last bar for market/tf.
        
        Returns:
            Last bar dict or None.
        """
        return self._bars.get((market, tf))
    
    def get_all(self) -> dict:
        """
        Get all last bars as flat dict for API.
        
        Returns:
            Dict with keys like "futures_5s", "spot_1m" etc.
        """
        result = {}
        
        for (market, tf), bar in self._bars.items():
            key = f"{market}_{tf}"
            # Return minimal fields for API
            result[key] = {
                "t_open_ms": bar.get("t_open_ms"),
                "t_close_ms": bar.get("t_close_ms"),
                "open": bar.get("open"),
                "high": bar.get("high"),
                "low": bar.get("low"),
                "close": bar.get("close"),
                "volume": bar.get("volume_total"),
                "delta": bar.get("delta_vol"),
                "trades": bar.get("trades_count"),
            }
        
        return result
    
    def clear(self) -> None:
        """Clear all stored bars."""
        self._bars.clear()
