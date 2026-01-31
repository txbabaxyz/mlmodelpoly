"""
Polymarket Book Store
=====================

Single source of truth for Polymarket orderbook state.

The BookStore:
- Holds YES and NO orderbooks separately
- Provides atomic snapshots
- Computes derived metrics (mid, spread, depth)
- Does NOT make trading decisions

Architecture:
    WS Client → update_yes() / update_no() → snapshot() → FeatureEngine

Thread Safety:
    All methods are thread-safe via internal lock.

Usage:
    store = PolymarketBookStore()
    
    # Update from WS
    store.update_yes(bids=[(0.52, 1000), (0.51, 500)], asks=[(0.54, 800)], ts_ms=now_ms())
    
    # Get snapshot
    snap = store.snapshot()
    # {
    #   "yes": {"best_bid": 0.52, "best_ask": 0.54, "mid": 0.53, ...},
    #   "no": {...},
    #   "market_id": "...",
    #   "ts_ms": 1234567890
    # }
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from collector.utils_time import now_ms

if TYPE_CHECKING:
    from collector.polymarket.spike_probe import SpikeProbe
    from collector.event_recorder import EventRecorder

logger = logging.getLogger(__name__)


@dataclass
class OrderbookSide:
    """
    Single orderbook (YES or NO token).
    
    Attributes:
        bids: List of (price, size) tuples, sorted descending by price
        asks: List of (price, size) tuples, sorted ascending by price
        ts_ms: Last update timestamp
    """
    bids: list[tuple[float, float]] = field(default_factory=list)
    asks: list[tuple[float, float]] = field(default_factory=list)
    ts_ms: int = 0
    
    def update(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        ts_ms: int,
    ) -> None:
        """
        Update orderbook with new bids/asks.
        
        Args:
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples  
            ts_ms: Update timestamp
        """
        # Sort bids descending (highest first)
        self.bids = sorted(bids, key=lambda x: x[0], reverse=True)
        # Sort asks ascending (lowest first)
        self.asks = sorted(asks, key=lambda x: x[0])
        self.ts_ms = ts_ms
    
    def best_bid(self) -> Optional[float]:
        """Get best bid price."""
        return self.bids[0][0] if self.bids else None
    
    def best_ask(self) -> Optional[float]:
        """Get best ask price."""
        return self.asks[0][0] if self.asks else None
    
    def best_bid_size(self) -> Optional[float]:
        """Get best bid size."""
        return self.bids[0][1] if self.bids else None
    
    def best_ask_size(self) -> Optional[float]:
        """Get best ask size."""
        return self.asks[0][1] if self.asks else None
    
    def mid(self) -> Optional[float]:
        """Get mid price."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    def spread(self) -> Optional[float]:
        """Get spread (ask - bid)."""
        bid = self.best_bid()
        ask = self.best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    def spread_bps(self) -> Optional[float]:
        """Get spread in basis points."""
        mid = self.mid()
        spread = self.spread()
        if mid is not None and spread is not None and mid > 0:
            return (spread / mid) * 10000
        return None
    
    def depth_top_n(self, n: int = 1) -> float:
        """
        Get total depth (bid_size + ask_size) for top N levels.
        
        Args:
            n: Number of levels to include
            
        Returns:
            Total size across top N bid + ask levels
        """
        bid_depth = sum(size for _, size in self.bids[:n])
        ask_depth = sum(size for _, size in self.asks[:n])
        return bid_depth + ask_depth
    
    def bid_depth_top_n(self, n: int = 1) -> float:
        """Get bid depth for top N levels."""
        return sum(size for _, size in self.bids[:n])
    
    def ask_depth_top_n(self, n: int = 1) -> float:
        """Get ask depth for top N levels."""
        return sum(size for _, size in self.asks[:n])
    
    def imbalance(self, levels: int = 5) -> Optional[float]:
        """
        Calculate orderbook imbalance.
        
        Formula: (bid_size - ask_size) / (bid_size + ask_size)
        
        Returns:
            Imbalance in range [-1, 1], or None if no data
            >0 = bid pressure (bullish)
            <0 = ask pressure (bearish)
        """
        bid_depth = self.bid_depth_top_n(levels)
        ask_depth = self.ask_depth_top_n(levels)
        total = bid_depth + ask_depth
        if total == 0:
            return None
        return (bid_depth - ask_depth) / total
    
    def microprice(self) -> Optional[float]:
        """
        Calculate microprice (size-weighted mid).
        
        Formula: (ask_price * bid_size + bid_price * ask_size) / (bid_size + ask_size)
        """
        bid_px = self.best_bid()
        ask_px = self.best_ask()
        bid_sz = self.best_bid_size()
        ask_sz = self.best_ask_size()
        
        if all(x is not None for x in [bid_px, ask_px, bid_sz, ask_sz]):
            total = bid_sz + ask_sz
            if total > 0:
                return (ask_px * bid_sz + bid_px * ask_sz) / total
        return None
    
    def snapshot(self) -> dict:
        """
        Get atomic snapshot of orderbook state.
        
        Returns:
            Dictionary with all derived metrics
        """
        return {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "best_bid_size": self.best_bid_size(),
            "best_ask_size": self.best_ask_size(),
            "mid": self.mid(),
            "spread": self.spread(),
            "spread_bps": self.spread_bps(),
            "depth_top1": self.depth_top_n(1),
            "depth_top3": self.depth_top_n(3),
            "depth_top5": self.depth_top_n(5),
            "bid_depth_top5": self.bid_depth_top_n(5),
            "ask_depth_top5": self.ask_depth_top_n(5),
            "imbalance": self.imbalance(5),
            "microprice": self.microprice(),
            "ts_ms": self.ts_ms,
        }


class PolymarketBookStore:
    """
    Central store for Polymarket YES/NO orderbooks.
    
    This is the single source of truth for Polymarket market data.
    All downstream consumers (FeatureEngine, EdgeEngine) read from here.
    
    Thread Safety:
        All public methods acquire internal lock for thread safety.
        
    Args:
        market_id: Optional market identifier (e.g., "btc-updown-15m")
        
    Usage:
        store = PolymarketBookStore(market_id="btc-updown-15m-1234567890")
        
        # WS client updates
        store.update_yes(bids, asks, ts_ms)
        store.update_no(bids, asks, ts_ms)
        
        # FeatureEngine reads
        snap = store.snapshot()
    """
    
    def __init__(self, market_id: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._yes = OrderbookSide()
        self._no = OrderbookSide()
        self._market_id = market_id
        self._yes_token_id: Optional[str] = None
        self._no_token_id: Optional[str] = None
        self._connected = False
        self._last_update_ms = 0
        self._spike_probe: Optional["SpikeProbe"] = None
        self._event_recorder: Optional["EventRecorder"] = None
        
        logger.info(
            "polymarket_book_store_initialized",
            extra={"market_id": market_id},
        )
    
    def set_event_recorder(self, recorder: "EventRecorder") -> None:
        """
        Set the EventRecorder for high-frequency logging.
        
        Args:
            recorder: EventRecorder instance
        """
        self._event_recorder = recorder
        logger.info("polymarket_event_recorder_set")
    
    def set_spike_probe(self, probe: "SpikeProbe") -> None:
        """
        Set the SpikeProbe for microspike detection.
        
        Args:
            probe: SpikeProbe instance
        """
        self._spike_probe = probe
        logger.info("polymarket_spike_probe_set")
    
    def set_market(
        self,
        market_id: str,
        yes_token_id: str,
        no_token_id: str,
    ) -> None:
        """
        Set current market and token IDs.
        
        Called when market changes (every 15 minutes for BTC market).
        
        Args:
            market_id: Market identifier (slug)
            yes_token_id: CLOB token ID for YES outcome
            no_token_id: CLOB token ID for NO outcome
        """
        with self._lock:
            self._market_id = market_id
            self._yes_token_id = yes_token_id
            self._no_token_id = no_token_id
            # Reset orderbooks on market change
            self._yes = OrderbookSide()
            self._no = OrderbookSide()
            
            logger.info(
                "polymarket_market_set",
                extra={
                    "market_id": market_id,
                    "yes_token": yes_token_id[:16] + "..." if yes_token_id else None,
                    "no_token": no_token_id[:16] + "..." if no_token_id else None,
                },
            )
    
    def set_connected(self, connected: bool) -> None:
        """Set connection status."""
        with self._lock:
            self._connected = connected
            
        logger.info(
            "polymarket_connection_status",
            extra={"connected": connected},
        )
    
    def update_yes(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        ts_ms: int,
    ) -> None:
        """
        Update YES orderbook.
        
        Args:
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            ts_ms: Update timestamp in milliseconds
        """
        with self._lock:
            self._yes.update(bids, asks, ts_ms)
            self._last_update_ms = ts_ms
            
            # Feed spike probe
            if self._spike_probe:
                best_ask = self._yes.best_ask()
                depth_top1 = self._yes.depth_top_n(1)
                if best_ask is not None:
                    self._spike_probe.on_book("YES", best_ask, depth_top1, ts_ms)
            
            # Record tick event
            if self._event_recorder:
                self._event_recorder.record_pm_tick(
                    side="YES",
                    bid=self._yes.best_bid(),
                    ask=self._yes.best_ask(),
                    mid=self._yes.mid(),
                    bid_size=self._yes.best_bid_size(),
                    ask_size=self._yes.best_ask_size(),
                    spread_bps=self._yes.spread_bps(),
                )
    
    def update_no(
        self,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        ts_ms: int,
    ) -> None:
        """
        Update NO orderbook.
        
        Args:
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            ts_ms: Update timestamp in milliseconds
        """
        with self._lock:
            self._no.update(bids, asks, ts_ms)
            self._last_update_ms = ts_ms
            
            # Feed spike probe
            if self._spike_probe:
                best_ask = self._no.best_ask()
                depth_top1 = self._no.depth_top_n(1)
                if best_ask is not None:
                    self._spike_probe.on_book("NO", best_ask, depth_top1, ts_ms)
            
            # Record tick event
            if self._event_recorder:
                self._event_recorder.record_pm_tick(
                    side="NO",
                    bid=self._no.best_bid(),
                    ask=self._no.best_ask(),
                    mid=self._no.mid(),
                    bid_size=self._no.best_bid_size(),
                    ask_size=self._no.best_ask_size(),
                    spread_bps=self._no.spread_bps(),
                )
    
    def update_by_token_id(
        self,
        token_id: str,
        bids: list[tuple[float, float]],
        asks: list[tuple[float, float]],
        ts_ms: int,
    ) -> bool:
        """
        Update orderbook by token ID.
        
        Args:
            token_id: CLOB token ID
            bids: List of (price, size) tuples
            asks: List of (price, size) tuples
            ts_ms: Update timestamp
            
        Returns:
            True if token matched YES or NO, False otherwise
        """
        # Route to update_yes/update_no to trigger spike probe and event recorder
        if token_id == self._yes_token_id:
            self.update_yes(bids, asks, ts_ms)
            return True
        elif token_id == self._no_token_id:
            self.update_no(bids, asks, ts_ms)
            return True
        return False
    
    def get_token_ids(self) -> tuple[Optional[str], Optional[str]]:
        """Get (yes_token_id, no_token_id) tuple."""
        with self._lock:
            return (self._yes_token_id, self._no_token_id)
    
    def snapshot(self) -> dict:
        """
        Get atomic snapshot of all orderbook state.
        
        Returns:
            Dictionary with YES and NO snapshots plus metadata.
            Structure is guaranteed even if no data (values will be None).
        """
        with self._lock:
            ts = now_ms()
            age_sec = (ts - self._last_update_ms) / 1000 if self._last_update_ms > 0 else None
            
            result = {
                "market_id": self._market_id,
                "yes": self._yes.snapshot(),
                "no": self._no.snapshot(),
                "connected": self._connected,
                "ts_ms": self._last_update_ms,
                "age_sec": round(age_sec, 1) if age_sec is not None else None,
            }
            
            # Add spike stats if probe is set
            if self._spike_probe:
                result["spike_stats"] = self._spike_probe.get_stats_60s()
            
            return result
    
    def get_yes_mid(self) -> Optional[float]:
        """Get YES mid price (thread-safe)."""
        with self._lock:
            return self._yes.mid()
    
    def get_no_mid(self) -> Optional[float]:
        """Get NO mid price (thread-safe)."""
        with self._lock:
            return self._no.mid()
    
    def get_summary(self) -> dict:
        """Get short summary for logging."""
        with self._lock:
            return {
                "market_id": self._market_id,
                "connected": self._connected,
                "yes_mid": self._yes.mid(),
                "no_mid": self._no.mid(),
                "yes_spread_bps": self._yes.spread_bps(),
                "no_spread_bps": self._no.spread_bps(),
            }
    
    def is_ready(self) -> bool:
        """
        Check if store has valid data.
        
        Returns:
            True if connected and has recent YES and NO updates
        """
        with self._lock:
            if not self._connected:
                return False
            if self._yes.ts_ms == 0 or self._no.ts_ms == 0:
                return False
            # Check freshness (within 10 seconds)
            ts = now_ms()
            age_yes = ts - self._yes.ts_ms
            age_no = ts - self._no.ts_ms
            return age_yes < 10_000 and age_no < 10_000
