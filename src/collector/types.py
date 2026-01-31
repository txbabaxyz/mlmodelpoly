"""
Type Definitions Module
=======================

Base types and data structures for the collector.
These types define the normalized event schema used throughout the system.

Schema version: 1.0
"""

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict


# Schema version for data format compatibility
SCHEMA_VERSION = "1.0"


class NormalizedEventDict(TypedDict):
    """
    Normalized event as TypedDict.
    
    This is the standard format for all events processed by the collector.
    Can be used for type hints when working with raw dictionaries.
    
    Fields:
        schema_version: Version of the event schema (e.g., "1.0")
        ts_event_ms: Original event timestamp from exchange (milliseconds)
        ts_recv_ms: Timestamp when event was received by collector (milliseconds)
        market: Market type ("spot" or "futures")
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        type: Event type (e.g., "trade", "depth", "ticker")
        data: Event-specific payload
    """
    schema_version: str
    ts_event_ms: int
    ts_recv_ms: int
    market: Literal["spot", "futures"]
    symbol: str
    type: str
    data: dict[str, Any]


@dataclass(slots=True, frozen=True)
class NormalizedEvent:
    """
    Normalized event dataclass.
    
    Immutable data structure for processed events.
    Use this when you need type safety and immutability.
    
    Attributes:
        schema_version: Version of the event schema
        ts_event_ms: Original event timestamp from exchange
        ts_recv_ms: Timestamp when event was received
        market: Market type (spot/futures)
        symbol: Trading pair symbol
        type: Event type
        data: Event-specific payload
    """
    schema_version: str
    ts_event_ms: int
    ts_recv_ms: int
    market: Literal["spot", "futures"]
    symbol: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> NormalizedEventDict:
        """
        Convert to dictionary format.
        
        Returns:
            Event as NormalizedEventDict.
        """
        return {
            "schema_version": self.schema_version,
            "ts_event_ms": self.ts_event_ms,
            "ts_recv_ms": self.ts_recv_ms,
            "market": self.market,
            "symbol": self.symbol,
            "type": self.type,
            "data": self.data,
        }
    
    @classmethod
    def create(
        cls,
        ts_event_ms: int,
        ts_recv_ms: int,
        market: Literal["spot", "futures"],
        symbol: str,
        event_type: str,
        data: dict[str, Any] | None = None,
    ) -> "NormalizedEvent":
        """
        Factory method to create a normalized event.
        
        Args:
            ts_event_ms: Event timestamp from exchange.
            ts_recv_ms: Receive timestamp.
            market: Market type.
            symbol: Trading pair.
            event_type: Type of event.
            data: Event payload.
            
        Returns:
            New NormalizedEvent instance.
        """
        return cls(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=ts_recv_ms,
            market=market,
            symbol=symbol,
            type=event_type,
            data=data or {},
        )


# Event type constants
EVENT_TYPE_TRADE = "trade"
EVENT_TYPE_AGG_TRADE = "agg_trade"
EVENT_TYPE_DEPTH = "depth"
EVENT_TYPE_BOOK_TICKER = "book_ticker"
EVENT_TYPE_KLINE = "kline"
EVENT_TYPE_TICKER = "ticker"
EVENT_TYPE_MARK_PRICE = "mark_price"
EVENT_TYPE_LIQUIDATION = "liquidation"

# Market type constants
MARKET_SPOT = "spot"
MARKET_FUTURES = "futures"
