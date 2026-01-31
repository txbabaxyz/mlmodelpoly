"""
SpikeProbe — Microspike Detection
=================================

Detects short-lived price improvement windows on Polymarket orderbook.

Purpose:
- Understand if there are executable microspikes worth fast-loop trading
- Track how long price improvements last (milliseconds)
- Measure improvement magnitude (ticks)

A "spike" occurs when best_ask drops by at least 1 tick, creating a 
buying opportunity. We track:
- When it starts (price drops)
- How long it lasts (until price returns)
- How much improvement (in ticks)

Usage:
    probe = SpikeProbe(tick=0.01, min_log_ms=50)
    
    # Call on each book update
    probe.on_book("YES", best_ask=0.52, depth_top1=1000, ts_ms=now_ms())
    
    # Get statistics
    stats = probe.get_stats_60s()
    # {"yes": {"count": 5, "median_duration_ms": 120, ...}, "no": {...}}

Architecture:
    BookStore.update_yes() → SpikeProbe.on_book("YES", ...)
    BookStore.update_no()  → SpikeProbe.on_book("NO", ...)
"""

import logging
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


@dataclass
class SpikeEvent:
    """Completed spike event."""
    side: str  # "YES" | "NO"
    start_ts_ms: int
    end_ts_ms: int
    start_px: float
    low_px: float  # Lowest price during spike
    end_px: float
    depth_at_start: float
    duration_ms: int = field(init=False)
    improve_ticks: float = field(init=False)
    
    def __post_init__(self):
        self.duration_ms = self.end_ts_ms - self.start_ts_ms
        self.improve_ticks = self.start_px - self.low_px


@dataclass
class ActiveSpike:
    """Currently active (ongoing) spike."""
    side: str
    start_ts_ms: int
    start_px: float  # Price before the drop
    low_px: float  # Lowest seen so far
    depth_at_start: float


class SpikeProbe:
    """
    Detects and tracks price improvement microspikes.
    
    A spike is detected when best_ask drops by at least `tick` from the 
    previous level. The spike ends when price returns above the starting level.
    
    Args:
        tick: Minimum price movement to consider a spike (default: 0.01 = 1 cent)
        min_log_ms: Minimum spike duration to log (default: 50ms)
        history_window_ms: Window for stats aggregation (default: 60000ms = 60s)
        
    Thread Safety:
        Methods should be called from a single thread (the WS handler thread).
    """
    
    def __init__(
        self,
        tick: float = 0.01,
        min_log_ms: int = 50,
        history_window_ms: int = 60_000,
    ):
        self.tick = tick
        self.min_log_ms = min_log_ms
        self.history_window_ms = history_window_ms
        
        # Active spikes by side
        self._active: dict[str, Optional[ActiveSpike]] = {
            "YES": None,
            "NO": None,
        }
        
        # Last known best_ask by side (for detecting drops)
        self._last_ask: dict[str, Optional[float]] = {
            "YES": None,
            "NO": None,
        }
        
        # Completed spike events (rolling window)
        self._events: deque[SpikeEvent] = deque(maxlen=1000)
        
        # Counters
        self._total_spikes = {"YES": 0, "NO": 0}
        self._total_logged = {"YES": 0, "NO": 0}
        
        logger.info(
            "spike_probe_initialized",
            extra={
                "tick": tick,
                "min_log_ms": min_log_ms,
                "history_window_ms": history_window_ms,
            },
        )
    
    def on_book(
        self,
        side: str,
        best_ask: float,
        depth_top1: float,
        ts_ms: int,
    ) -> Optional[SpikeEvent]:
        """
        Process a book update and detect/track spikes.
        
        Args:
            side: "YES" or "NO"
            best_ask: Current best ask price
            depth_top1: Current depth at top level
            ts_ms: Timestamp in milliseconds
            
        Returns:
            SpikeEvent if a spike just completed, None otherwise
        """
        if side not in ("YES", "NO"):
            return None
        
        if best_ask is None or best_ask <= 0:
            return None
        
        last_ask = self._last_ask[side]
        active = self._active[side]
        completed: Optional[SpikeEvent] = None
        
        # First update - just record the price
        if last_ask is None:
            self._last_ask[side] = best_ask
            return None
        
        # Check if we have an active spike
        if active is not None:
            # Update low price if we went lower
            if best_ask < active.low_px:
                active.low_px = best_ask
            
            # Check if spike ended (price returned above start level)
            if best_ask >= active.start_px:
                # Spike completed
                event = SpikeEvent(
                    side=side,
                    start_ts_ms=active.start_ts_ms,
                    end_ts_ms=ts_ms,
                    start_px=active.start_px,
                    low_px=active.low_px,
                    end_px=best_ask,
                    depth_at_start=active.depth_at_start,
                )
                
                self._active[side] = None
                self._total_spikes[side] += 1
                
                # Only log/record if duration meets threshold
                if event.duration_ms >= self.min_log_ms:
                    self._events.append(event)
                    self._total_logged[side] += 1
                    completed = event
                    
                    # Log the spike
                    logger.info(
                        "spike_detected",
                        extra={
                            "side": side,
                            "duration_ms": event.duration_ms,
                            "improve_ticks": round(event.improve_ticks / self.tick, 2),
                            "start_px": event.start_px,
                            "low_px": event.low_px,
                            "depth": round(event.depth_at_start, 1),
                        },
                    )
        else:
            # No active spike - check if price dropped enough to start one
            price_drop = last_ask - best_ask
            
            if price_drop >= self.tick:
                # Start new spike
                self._active[side] = ActiveSpike(
                    side=side,
                    start_ts_ms=ts_ms,
                    start_px=last_ask,
                    low_px=best_ask,
                    depth_at_start=depth_top1,
                )
        
        # Update last known price
        self._last_ask[side] = best_ask
        
        return completed
    
    def get_stats_60s(self) -> dict:
        """
        Get spike statistics for the last 60 seconds.
        
        Returns:
            Dictionary with stats for YES and NO sides
        """
        ts_now = now_ms()
        cutoff = ts_now - self.history_window_ms
        
        # Filter events in window
        yes_events = [e for e in self._events if e.side == "YES" and e.end_ts_ms >= cutoff]
        no_events = [e for e in self._events if e.side == "NO" and e.end_ts_ms >= cutoff]
        
        return {
            "yes": self._compute_side_stats(yes_events),
            "no": self._compute_side_stats(no_events),
            "window_ms": self.history_window_ms,
            "ts_ms": ts_now,
        }
    
    def _compute_side_stats(self, events: list[SpikeEvent]) -> dict:
        """Compute statistics for a list of events."""
        if not events:
            return {
                "count": 0,
                "median_duration_ms": None,
                "mean_duration_ms": None,
                "max_duration_ms": None,
                "min_duration_ms": None,
                "median_improve_ticks": None,
                "max_improve_ticks": None,
                "total_improve_ticks": None,
            }
        
        durations = [e.duration_ms for e in events]
        improve_ticks = [e.improve_ticks / self.tick for e in events]
        
        return {
            "count": len(events),
            "median_duration_ms": round(statistics.median(durations), 1),
            "mean_duration_ms": round(statistics.mean(durations), 1),
            "max_duration_ms": max(durations),
            "min_duration_ms": min(durations),
            "median_improve_ticks": round(statistics.median(improve_ticks), 2),
            "max_improve_ticks": round(max(improve_ticks), 2),
            "total_improve_ticks": round(sum(improve_ticks), 2),
        }
    
    def get_active(self) -> dict:
        """Get currently active spikes."""
        result = {}
        ts_now = now_ms()
        
        for side in ("YES", "NO"):
            active = self._active[side]
            if active:
                result[side] = {
                    "active": True,
                    "duration_so_far_ms": ts_now - active.start_ts_ms,
                    "start_px": active.start_px,
                    "low_px": active.low_px,
                    "current_improve_ticks": round((active.start_px - active.low_px) / self.tick, 2),
                }
            else:
                result[side] = {"active": False}
        
        return result
    
    def get_summary(self) -> dict:
        """Get summary for logging."""
        stats = self.get_stats_60s()
        
        return {
            "yes_count_60s": stats["yes"]["count"],
            "yes_median_ms": stats["yes"]["median_duration_ms"],
            "yes_max_ticks": stats["yes"]["max_improve_ticks"],
            "no_count_60s": stats["no"]["count"],
            "no_median_ms": stats["no"]["median_duration_ms"],
            "no_max_ticks": stats["no"]["max_improve_ticks"],
            "total_spikes_yes": self._total_spikes["YES"],
            "total_spikes_no": self._total_spikes["NO"],
        }
    
    def snapshot(self) -> dict:
        """Full snapshot for API."""
        return {
            "stats_60s": self.get_stats_60s(),
            "active": self.get_active(),
            "totals": {
                "yes_total": self._total_spikes["YES"],
                "yes_logged": self._total_logged["YES"],
                "no_total": self._total_spikes["NO"],
                "no_logged": self._total_logged["NO"],
            },
            "config": {
                "tick": self.tick,
                "min_log_ms": self.min_log_ms,
            },
        }
