"""
Polymarket Integration Module
=============================

Provides Polymarket prediction market data integration:
- BookStore: Orderbook state management for YES/NO tokens
- WSClient: WebSocket client for real-time orderbook updates
- MarketResolver: API client for market/token resolution
- SpikeProbe: Microspike detection for price improvements

Architecture:
    Polymarket WS → PolymarketBookStore → FeatureEngine.snapshot() → EdgeEngine.decide()
                          ↓
                    SpikeProbe (microspike detection)

Usage:
    from collector.polymarket import PolymarketBookStore, PolymarketWSClient, MarketResolver, SpikeProbe
"""

from collector.polymarket.book_store import PolymarketBookStore
from collector.polymarket.ws_client import PolymarketWSClient
from collector.polymarket.market_resolver import MarketResolver
from collector.polymarket.spike_probe import SpikeProbe

__all__ = [
    "PolymarketBookStore",
    "PolymarketWSClient",
    "MarketResolver",
    "SpikeProbe",
]
