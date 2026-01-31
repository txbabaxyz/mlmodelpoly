"""
Event Pipeline Module
=====================

Routes incoming WebSocket messages to appropriate normalizers
and handles event processing logic.

Supported markets and event types:
    - futures: aggTrade, bookTicker, markPrice, forceOrder, depthUpdate
    - spot: aggTrade, bookTicker
"""

import logging
from typing import Optional
from collections import defaultdict

from collector.normalizers import (
    # Futures
    normalize_futures_aggtrade,
    normalize_futures_bookticker,
    normalize_futures_markprice,
    normalize_futures_liquidation,
    normalize_futures_depth,
    # Spot
    normalize_spot_aggtrade,
    normalize_spot_bookticker,
)
from collector.types import NormalizedEvent

logger = logging.getLogger(__name__)


class EventPipeline:
    """
    Event processing pipeline.
    
    Routes raw WebSocket messages to appropriate normalizers
    based on market (futures/spot) and event type.
    
    Usage:
        pipeline = EventPipeline(symbol="BTCUSDT", topn=10)
        event = pipeline.handle_ws_message(envelope)
        if event:
            # Process normalized event
            pass
    """
    
    def __init__(self, symbol: str, topn: int = 10):
        """
        Initialize pipeline.
        
        Args:
            symbol: Symbol to process (uppercase, e.g., "BTCUSDT")
            topn: Number of order book levels for depth normalization
        """
        self.symbol = symbol
        self.topn = topn
        
        # Counters per market and event type
        # Structure: {market: {event_type: count}}
        self._processed_count: dict[str, dict[str, int]] = {
            "futures": defaultdict(int),
            "spot": defaultdict(int),
        }
        self._error_count: dict[str, dict[str, int]] = {
            "futures": defaultdict(int),
            "spot": defaultdict(int),
        }
    
    def handle_ws_message(self, envelope: dict) -> Optional[NormalizedEvent]:
        """
        Process incoming WebSocket message envelope.
        
        Args:
            envelope: Message envelope from WSClient containing:
                - ws_name: Source client name ("futures", "spot")
                - recv_ts_ms: Receive timestamp
                - msg: Raw Binance message with 'stream' and 'data'
        
        Returns:
            NormalizedEvent if successfully processed, None otherwise.
        """
        try:
            msg = envelope.get("msg", {})
            ws_name = envelope.get("ws_name", "unknown")
            
            # Combined stream format: {"stream": "...", "data": {...}}
            data = msg.get("data", {})
            stream = msg.get("stream", "")
            
            if not data:
                # Might be a subscription response or ping
                return None
            
            # Get event type
            event_type = data.get("e")
            
            # For spot bookTicker, "e" might not be present
            # Detect by presence of bid/ask fields
            if not event_type and data.get("b") and data.get("a") and data.get("s"):
                event_type = "bookTicker"
            
            if not event_type:
                # Not an event message
                return None
            
            # Route based on market (ws_name)
            normalized: Optional[NormalizedEvent] = None
            market = ws_name  # "futures" or "spot"
            
            if ws_name == "futures":
                normalized = self._route_futures(event_type, envelope, stream)
            elif ws_name == "spot":
                normalized = self._route_spot(event_type, envelope, stream)
            else:
                logger.warning(
                    "pipeline_unknown_source",
                    extra={"ws_name": ws_name, "event_type": event_type},
                )
                return None
            
            # Update counters
            if normalized:
                self._processed_count[market][normalized.type] += 1
            else:
                self._error_count[market][event_type] += 1
            
            return normalized
            
        except Exception as e:
            logger.exception(
                "pipeline_error",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return None
    
    def _route_futures(
        self,
        event_type: str,
        envelope: dict,
        stream: str,
    ) -> Optional[NormalizedEvent]:
        """
        Route futures events to appropriate normalizer.
        """
        if event_type == "aggTrade":
            return normalize_futures_aggtrade(envelope, self.symbol)
        
        elif event_type == "bookTicker":
            return normalize_futures_bookticker(envelope, self.symbol)
        
        elif event_type == "markPriceUpdate":
            return normalize_futures_markprice(envelope, self.symbol)
        
        elif event_type == "forceOrder":
            return normalize_futures_liquidation(envelope, self.symbol)
        
        elif event_type == "depthUpdate":
            return normalize_futures_depth(envelope, self.symbol, self.topn)
        
        else:
            logger.warning(
                "pipeline_unknown_event",
                extra={"event_type": event_type, "stream": stream, "market": "futures"},
            )
            return None
    
    def _route_spot(
        self,
        event_type: str,
        envelope: dict,
        stream: str,
    ) -> Optional[NormalizedEvent]:
        """
        Route spot events to appropriate normalizer.
        """
        if event_type == "aggTrade":
            return normalize_spot_aggtrade(envelope, self.symbol)
        
        elif event_type == "bookTicker":
            return normalize_spot_bookticker(envelope, self.symbol)
        
        else:
            logger.warning(
                "pipeline_unknown_event",
                extra={"event_type": event_type, "stream": stream, "market": "spot"},
            )
            return None
    
    @property
    def processed_count(self) -> int:
        """Total number of successfully processed events across all markets."""
        total = 0
        for market_counts in self._processed_count.values():
            total += sum(market_counts.values())
        return total
    
    @property
    def error_count(self) -> int:
        """Total number of processing errors across all markets."""
        total = 0
        for market_counts in self._error_count.values():
            total += sum(market_counts.values())
        return total
    
    def get_counts_by_market(self) -> dict[str, dict[str, int]]:
        """
        Get processed counts organized by market and type.
        
        Returns:
            {
                "futures": {"aggTrade": N, "bookTicker": M, ...},
                "spot": {"aggTrade": N, "bookTicker": M}
            }
        """
        return {
            market: dict(counts)
            for market, counts in self._processed_count.items()
        }
    
    def get_errors_by_market(self) -> dict[str, dict[str, int]]:
        """Get error counts organized by market and type."""
        return {
            market: dict(counts)
            for market, counts in self._error_count.items()
        }
    
    def reset_counters(self) -> tuple[dict, dict]:
        """
        Reset and return current counters.
        
        Returns:
            Tuple of (processed_counts, error_counts) before reset.
        """
        processed = self.get_counts_by_market()
        errors = self.get_errors_by_market()
        
        for market in self._processed_count:
            self._processed_count[market].clear()
        for market in self._error_count:
            self._error_count[market].clear()
        
        return processed, errors


class SamplingLogger:
    """
    Utility for sampling log output per market and event type.
    
    Different event types can have different sample rates.
    Rates are applied per (market, event_type) combination.
    
    Usage:
        sampler = SamplingLogger()
        if sampler.should_log("futures", "aggTrade"):
            logger.info("event_sample", extra={...})
    """
    
    # Default sample rates per event type (applied to all markets)
    DEFAULT_RATES = {
        "aggTrade": 200,
        "bookTicker": 50,
        "markPrice": 50,
        "liquidation": 1,  # Always log liquidations
        "depth": 200,
    }
    
    def __init__(self, sample_rates: Optional[dict[str, int]] = None):
        """
        Initialize sampler.
        
        Args:
            sample_rates: Dict of event_type -> sample_rate (1 = always log)
                          Uses DEFAULT_RATES for missing types.
        """
        self._rates = {**self.DEFAULT_RATES}
        if sample_rates:
            self._rates.update(sample_rates)
        
        # Counters per (market, event_type)
        self._counters: dict[tuple[str, str], int] = defaultdict(int)
    
    def should_log(self, market: str, event_type: str) -> bool:
        """
        Check if current event should be logged.
        
        Args:
            market: Market name ("futures", "spot")
            event_type: Type of event (e.g., "aggTrade")
        
        Returns:
            True if this event should be logged.
        """
        rate = self._rates.get(event_type, 100)  # Default: 1/100
        key = (market, event_type)
        
        self._counters[key] += 1
        
        if self._counters[key] >= rate:
            self._counters[key] = 0
            return True
        
        return False
    
    def reset(self, market: Optional[str] = None, event_type: Optional[str] = None) -> None:
        """
        Reset counter(s).
        
        Args:
            market: Specific market to reset, or None for all.
            event_type: Specific type to reset, or None for all.
        """
        if market and event_type:
            self._counters[(market, event_type)] = 0
        elif market:
            keys_to_reset = [k for k in self._counters if k[0] == market]
            for key in keys_to_reset:
                self._counters[key] = 0
        elif event_type:
            keys_to_reset = [k for k in self._counters if k[1] == event_type]
            for key in keys_to_reset:
                self._counters[key] = 0
        else:
            self._counters.clear()
