"""
Metrics Collection Module
=========================

Collects and computes quality metrics for the collector service:
- WebSocket connection status
- Reconnect counts
- Event counts and rates (RPS)
- Lag percentiles (p50, p95) via rolling windows

Thread-safe for use across async tasks.
"""

import logging
import threading
import time
from collections import defaultdict, deque
from typing import Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# Rolling window sizes
RPS_WINDOW_SECONDS = 5
LAG_WINDOW_SECONDS = 60

# Lag sanity bounds (ms)
LAG_MIN_MS = -1000
LAG_MAX_MS = 60000


class Metrics:
    """
    Central metrics collection for the collector service.
    
    Tracks:
        - WS connection status per market
        - Reconnect counts per market
        - Event counts and RPS per (market, event_type)
        - Lag percentiles per (market, event_type)
    
    Thread-safe via lock for concurrent access.
    
    Usage:
        metrics = Metrics()
        metrics.mark_connected("futures", True)
        metrics.observe_event("futures", "aggTrade", lag_ms=150)
        snapshot = metrics.snapshot()
    """
    
    def __init__(self):
        """Initialize metrics storage."""
        self._lock = threading.Lock()
        
        # WS connection status
        self._ws_connected: dict[str, bool] = {
            "futures": False,
            "spot": False,
        }
        
        # Reconnect counters
        self._reconnects_total: dict[str, int] = {
            "futures": 0,
            "spot": 0,
        }
        
        # Event counters (total)
        # Structure: {market: {event_type: count}}
        self._events_total: dict[str, dict[str, int]] = {
            "futures": defaultdict(int),
            "spot": defaultdict(int),
        }
        
        # RPS rolling window
        # Structure: {market: {event_type: deque[(ts_sec, count)]}}
        self._rps_window: dict[str, dict[str, deque]] = {
            "futures": defaultdict(lambda: deque(maxlen=RPS_WINDOW_SECONDS * 100)),
            "spot": defaultdict(lambda: deque(maxlen=RPS_WINDOW_SECONDS * 100)),
        }
        
        # Lag rolling window
        # Structure: {market: {event_type: deque[(ts_sec, lag_ms)]}}
        self._lag_window: dict[str, dict[str, deque]] = {
            "futures": defaultdict(lambda: deque(maxlen=10000)),
            "spot": defaultdict(lambda: deque(maxlen=10000)),
        }
        
        # Bar counters
        # Structure: {market: {tf: count}}
        self._bars_total: dict[str, dict[str, int]] = {
            "futures": defaultdict(int),
            "spot": defaultdict(int),
        }
        
        # Startup time
        self._start_time_ms = now_ms()
    
    def mark_connected(self, market: str, connected: bool) -> None:
        """
        Update WebSocket connection status.
        
        Args:
            market: Market name ("futures" or "spot")
            connected: True if connected, False if disconnected
        """
        with self._lock:
            if market in self._ws_connected:
                self._ws_connected[market] = connected
    
    def inc_reconnect(self, market: str) -> None:
        """
        Increment reconnect counter for a market.
        
        Args:
            market: Market name ("futures" or "spot")
        """
        with self._lock:
            if market in self._reconnects_total:
                self._reconnects_total[market] += 1
    
    def inc_bar(self, market: str, tf: str) -> None:
        """
        Increment bar counter.
        
        Args:
            market: Market name ("futures" or "spot")
            tf: Timeframe label (e.g., "5s", "15s", "1m")
        """
        with self._lock:
            if market in self._bars_total:
                self._bars_total[market][tf] += 1
    
    def observe_event(self, market: str, etype: str, lag_ms: int) -> None:
        """
        Record an event observation.
        
        Args:
            market: Market name ("futures" or "spot")
            etype: Event type (e.g., "aggTrade", "bookTicker")
            lag_ms: Latency in milliseconds (ts_recv - ts_event)
        """
        # Validate lag
        if lag_ms < LAG_MIN_MS or lag_ms > LAG_MAX_MS:
            logger.warning(
                "metrics_invalid_lag",
                extra={"market": market, "etype": etype, "lag_ms": lag_ms},
            )
            # Still count the event, just don't track lag
            lag_ms = None
        
        ts_sec = time.time()
        
        with self._lock:
            # Increment total counter
            if market in self._events_total:
                self._events_total[market][etype] += 1
            
            # Record for RPS calculation
            if market in self._rps_window:
                self._rps_window[market][etype].append((ts_sec, 1))
            
            # Record lag for percentiles (if valid)
            if lag_ms is not None and market in self._lag_window:
                self._lag_window[market][etype].append((ts_sec, lag_ms))
    
    def snapshot(self) -> dict:
        """
        Generate a snapshot of all metrics.
        
        Returns:
            Dictionary with all metrics for serialization.
        """
        current_time = time.time()
        current_time_ms = now_ms()
        
        with self._lock:
            # Copy connection status
            ws_connected = dict(self._ws_connected)
            
            # Copy reconnect totals
            reconnects_total = dict(self._reconnects_total)
            
            # Copy event totals
            events_total = {
                market: dict(counts)
                for market, counts in self._events_total.items()
            }
            
            # Calculate RPS
            events_per_sec = self._calculate_rps(current_time)
            
            # Calculate lag percentiles
            lag_stats = self._calculate_lag_percentiles(current_time)
            
            # Copy bar totals
            bars_total = {
                market: dict(counts)
                for market, counts in self._bars_total.items()
            }
        
        return {
            "server_time_ms": current_time_ms,
            "uptime_ms": current_time_ms - self._start_time_ms,
            "ws_connected": ws_connected,
            "reconnects_total": reconnects_total,
            "events_total": events_total,
            "events_per_sec": events_per_sec,
            "lag_p50_ms": lag_stats["p50"],
            "lag_p95_ms": lag_stats["p95"],
            "lag_samples_count": lag_stats["samples"],
            "bars_total": bars_total,
        }
    
    def _calculate_rps(self, current_time: float) -> dict[str, dict[str, float]]:
        """
        Calculate events per second from rolling window.
        
        Must be called with lock held.
        """
        cutoff = current_time - RPS_WINDOW_SECONDS
        result: dict[str, dict[str, float]] = {}
        
        for market, types in self._rps_window.items():
            result[market] = {}
            for etype, window in types.items():
                # Remove old entries and count recent
                count = 0
                while window and window[0][0] < cutoff:
                    window.popleft()
                
                count = sum(entry[1] for entry in window)
                
                # Calculate rate
                if count > 0:
                    result[market][etype] = round(count / RPS_WINDOW_SECONDS, 2)
        
        return result
    
    def _calculate_lag_percentiles(
        self,
        current_time: float,
    ) -> dict[str, dict[str, dict[str, Optional[float]]]]:
        """
        Calculate p50 and p95 lag percentiles from rolling window.
        
        Must be called with lock held.
        """
        cutoff = current_time - LAG_WINDOW_SECONDS
        
        p50: dict[str, dict[str, Optional[float]]] = {}
        p95: dict[str, dict[str, Optional[float]]] = {}
        samples: dict[str, dict[str, int]] = {}
        
        for market, types in self._lag_window.items():
            p50[market] = {}
            p95[market] = {}
            samples[market] = {}
            
            for etype, window in types.items():
                # Remove old entries
                while window and window[0][0] < cutoff:
                    window.popleft()
                
                # Extract lag values
                lags = [entry[1] for entry in window]
                samples[market][etype] = len(lags)
                
                if lags:
                    lags_sorted = sorted(lags)
                    n = len(lags_sorted)
                    
                    # p50
                    idx50 = int(n * 0.50)
                    p50[market][etype] = lags_sorted[min(idx50, n - 1)]
                    
                    # p95
                    idx95 = int(n * 0.95)
                    p95[market][etype] = lags_sorted[min(idx95, n - 1)]
                else:
                    p50[market][etype] = None
                    p95[market][etype] = None
        
        return {"p50": p50, "p95": p95, "samples": samples}
    
    def get_short_summary(self) -> dict:
        """
        Get a short summary for periodic logging.
        
        Returns minimal essential metrics.
        """
        snap = self.snapshot()
        
        # Extract key metrics
        summary = {
            "ws_connected": snap["ws_connected"],
            "reconnects": snap["reconnects_total"],
        }
        
        # Add RPS for key types
        eps = snap.get("events_per_sec", {})
        if "futures" in eps and "aggTrade" in eps["futures"]:
            summary["futures_aggTrade_rps"] = eps["futures"]["aggTrade"]
        if "spot" in eps and "aggTrade" in eps["spot"]:
            summary["spot_aggTrade_rps"] = eps["spot"]["aggTrade"]
        
        # Add p95 lag for key types
        p95 = snap.get("lag_p95_ms", {})
        if "futures" in p95 and "aggTrade" in p95["futures"]:
            summary["futures_aggTrade_p95_ms"] = p95["futures"]["aggTrade"]
        if "spot" in p95 and "aggTrade" in p95["spot"]:
            summary["spot_aggTrade_p95_ms"] = p95["spot"]["aggTrade"]
        
        # Add bar counts
        bars = snap.get("bars_total", {})
        
        # Futures bars
        if "futures" in bars:
            for tf in ["5s", "15s", "1m"]:
                if tf in bars["futures"]:
                    summary[f"futures_bars_{tf}"] = bars["futures"][tf]
        
        # Spot bars
        if "spot" in bars:
            for tf in ["5s", "15s", "1m"]:
                if tf in bars["spot"]:
                    summary[f"spot_bars_{tf}"] = bars["spot"][tf]
        
        return summary
