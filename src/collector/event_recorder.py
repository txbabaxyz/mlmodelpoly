"""
Event Recorder
==============

High-frequency event logger for backtesting and analysis.

Records all price changes and computed metrics to JSONL files:
- Each Polymarket market (15-min window) → separate file
- Binance ticks, klines
- Computed features (fair, edge, spikes, dips)
- Accumulate decisions

Architecture:
    Sources → EventRecorder.record() → async queue → buffered file writer

File naming:
    data/recordings/{market_id}_{start_ts}.jsonl
    e.g., data/recordings/pm_123456_1706448000000.jsonl

Event types:
    - pm_tick: Polymarket orderbook update (YES or NO side)
    - pm_microspike: SpikeProbe detected microspike
    - bn_trade: Binance trade tick
    - bn_kline: Binance kline close
    - features: Computed features snapshot
    - decision: AccumulateEngine decision

Usage:
    recorder = EventRecorder(output_dir="data/recordings")
    await recorder.start()
    
    # Record events
    recorder.record("pm_tick", {...})
    recorder.record("bn_trade", {...})
    
    # On shutdown
    await recorder.stop()
"""

import asyncio
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# Buffer size before flush
DEFAULT_BUFFER_SIZE = 100

# Flush interval (ms) even if buffer not full
DEFAULT_FLUSH_INTERVAL_MS = 1000


@dataclass
class EventBuffer:
    """Buffer for events of a specific market/file."""
    market_id: str
    file_path: Path
    events: list[dict] = field(default_factory=list)
    file_handle: Optional[Any] = None
    events_written: int = 0
    last_flush_ms: int = 0
    
    def append(self, event: dict) -> None:
        self.events.append(event)
    
    def should_flush(self, buffer_size: int, flush_interval_ms: int) -> bool:
        if len(self.events) >= buffer_size:
            return True
        if self.events and (now_ms() - self.last_flush_ms) >= flush_interval_ms:
            return True
        return False


class EventRecorder:
    """
    High-frequency event recorder with async buffered writing.
    
    Features:
        - Separate file per market (Polymarket 15-min window)
        - Async queue to not block WS handlers
        - Buffered writes for performance
        - Graceful shutdown with flush
    
    Args:
        output_dir: Directory to write JSONL files
        buffer_size: Events to buffer before flush (default 100)
        flush_interval_ms: Max time between flushes (default 1000ms)
        enabled: Whether recording is enabled
    """
    
    def __init__(
        self,
        output_dir: str = "data/recordings",
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        flush_interval_ms: int = DEFAULT_FLUSH_INTERVAL_MS,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.buffer_size = buffer_size
        self.flush_interval_ms = flush_interval_ms
        self.enabled = enabled
        
        # Buffers per market_id
        self._buffers: dict[str, EventBuffer] = {}
        
        # Global buffer for non-market events (binance, etc.)
        self._global_market_id: Optional[str] = None
        
        # Async queue for thread-safe recording
        self._queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()
        
        # Background task
        self._writer_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Stats
        self._total_events = 0
        self._total_written = 0
        
        logger.info(
            "event_recorder_initialized",
            extra={
                "output_dir": str(self.output_dir),
                "buffer_size": buffer_size,
                "flush_interval_ms": flush_interval_ms,
                "enabled": enabled,
            },
        )
    
    async def start(self) -> None:
        """Start the background writer task."""
        if not self.enabled:
            logger.info("event_recorder_disabled")
            return
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._running = True
        self._writer_task = asyncio.create_task(self._writer_loop())
        
        logger.info("event_recorder_started")
    
    async def stop(self) -> None:
        """Stop recording and flush all buffers."""
        if not self._running:
            return
        
        self._running = False
        
        # Signal writer to stop
        await self._queue.put(("__STOP__", {}))
        
        # Wait for writer to finish
        if self._writer_task:
            try:
                await asyncio.wait_for(self._writer_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning("event_recorder_stop_timeout")
                self._writer_task.cancel()
        
        # Final flush and close all files
        await self._flush_all()
        self._close_all_files()
        
        logger.info(
            "event_recorder_stopped",
            extra={
                "total_events": self._total_events,
                "total_written": self._total_written,
                "markets": len(self._buffers),
            },
        )
    
    def set_current_market(self, market_id: str) -> None:
        """
        Set current Polymarket market ID.
        
        Called when a new 15-min window starts.
        All subsequent events will go to this market's file.
        
        Args:
            market_id: Unique market identifier (e.g., "pm_1706448000000")
        """
        if not self.enabled:
            return
        
        self._global_market_id = market_id
        
        # Create buffer if not exists
        if market_id not in self._buffers:
            ts_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_name = f"{market_id}_{ts_str}.jsonl"
            file_path = self.output_dir / file_name
            
            self._buffers[market_id] = EventBuffer(
                market_id=market_id,
                file_path=file_path,
                last_flush_ms=now_ms(),
            )
            
            logger.info(
                "event_recorder_new_market",
                extra={
                    "market_id": market_id,
                    "file_path": str(file_path),
                },
            )
    
    def record(self, event_type: str, data: dict, market_id: Optional[str] = None) -> None:
        """
        Record an event (non-blocking).
        
        Args:
            event_type: Type of event (pm_tick, bn_trade, features, etc.)
            data: Event data dictionary
            market_id: Optional market ID (uses current if not specified)
        """
        if not self.enabled or not self._running:
            return
        
        # Use specified market or current
        mid = market_id or self._global_market_id
        if not mid:
            # No market set yet, use "global"
            mid = "global"
            if mid not in self._buffers:
                self.set_current_market(mid)
        
        # Build event
        event = {
            "ts_ms": now_ms(),
            "type": event_type,
            **data,
        }
        
        # Put in queue (non-blocking)
        try:
            self._queue.put_nowait((mid, event))
            self._total_events += 1
        except asyncio.QueueFull:
            logger.warning("event_recorder_queue_full")
    
    def record_pm_tick(
        self,
        side: str,  # "YES" or "NO"
        bid: Optional[float],
        ask: Optional[float],
        mid: Optional[float],
        bid_size: Optional[float] = None,
        ask_size: Optional[float] = None,
        spread_bps: Optional[float] = None,
    ) -> None:
        """Record Polymarket orderbook tick."""
        self.record("pm_tick", {
            "side": side,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "bid_size": bid_size,
            "ask_size": ask_size,
            "spread_bps": spread_bps,
        })
    
    def record_pm_microspike(
        self,
        side: str,
        direction: str,  # "up" or "down"
        magnitude_bps: float,
        prev_mid: float,
        curr_mid: float,
    ) -> None:
        """Record Polymarket microspike event."""
        self.record("pm_microspike", {
            "side": side,
            "direction": direction,
            "magnitude_bps": magnitude_bps,
            "prev_mid": prev_mid,
            "curr_mid": curr_mid,
        })
    
    def record_bn_trade(
        self,
        market: str,  # "futures" or "spot"
        price: float,
        qty: float,
        side: str,  # "BUY" or "SELL"
    ) -> None:
        """Record Binance trade tick."""
        self.record("bn_trade", {
            "market": market,
            "price": price,
            "qty": qty,
            "side": side,
        })
    
    def record_bn_ticker(
        self,
        market: str,
        bid: float,
        ask: float,
        mid: float,
    ) -> None:
        """Record Binance bookTicker (bid/ask/mid)."""
        self.record("bn_ticker", {
            "market": market,
            "bid": bid,
            "ask": ask,
            "mid": mid,
        })
    
    def record_bn_kline(
        self,
        market: str,
        tf: str,
        open_px: float,
        high: float,
        low: float,
        close: float,
        volume: float,
    ) -> None:
        """Record Binance kline close."""
        self.record("bn_kline", {
            "market": market,
            "tf": tf,
            "open": open_px,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        })
    
    def record_features(self, features: dict) -> None:
        """Record computed features snapshot (S7 schema v3)."""
        # Extract key fields
        market_ref = features.get("market_ref") or {}
        vol = features.get("vol") or {}
        fair = features.get("fair") or {}
        fair_status = fair.get("status") or features.get("fair_status") or {}
        fair_fast = fair.get("fast") or {}
        fair_smooth = fair.get("smooth") or {}
        bias = features.get("bias") or {}
        spikes = features.get("spikes") or {}
        pm_dips = features.get("pm_dips") or {}
        pm_updown = features.get("polymarket_up_down") or {}
        
        up_data = pm_updown.get("up") or {}
        down_data = pm_updown.get("down") or {}
        
        self.record("features", {
            # === Market Ref (S1) ===
            "s_now_raw": market_ref.get("S_now_raw"),
            "s_now_smooth": market_ref.get("S_now_smooth"),
            "s_now": market_ref.get("S_now"),  # legacy
            "ref_px": market_ref.get("ref_px"),
            "tau_sec": market_ref.get("tau_sec"),
            
            # === Vol (S2) ===
            "sigma_fast_15m": vol.get("sigma_fast_15m"),
            "sigma_slow_15m": vol.get("sigma_slow_15m"),
            "sigma_blend_15m": vol.get("sigma_blend_15m"),
            "blend_w": vol.get("blend_w"),
            "vol_n_bars": vol.get("n_bars"),
            "vol_reason": vol.get("reason"),
            "sigma_15m": vol.get("sigma_15m"),  # legacy
            
            # === Fair (S3) ===
            "fair_fast_up": fair_fast.get("up"),
            "fair_fast_down": fair_fast.get("down"),
            "fair_fast_z": fair_fast.get("z"),
            "fair_smooth_up": fair_smooth.get("up"),
            "fair_smooth_down": fair_smooth.get("down"),
            "fair_smooth_z": fair_smooth.get("z"),
            "fair_fast_ready": fair_status.get("fast_ready"),
            "fair_smooth_ready": fair_status.get("smooth_ready"),
            "fair_fast_reason": fair_status.get("fast_reason"),
            "fair_smooth_reason": fair_status.get("smooth_reason"),
            "fair_up": fair.get("fair_up"),  # legacy
            "fair_down": fair.get("fair_down"),  # legacy
            "fair_ready": fair_status.get("ready"),  # legacy
            "z_score": fair.get("z_score"),  # legacy
            
            # === Bias (S4) ===
            "bias_dir": bias.get("dir"),
            "bias_strength": bias.get("strength"),
            "bias_up_prob": bias.get("bias_up_prob"),
            
            # === Polymarket ===
            "pm_up_mid": up_data.get("mid"),
            "pm_up_bid": up_data.get("bid"),
            "pm_up_ask": up_data.get("best_ask"),
            "pm_down_mid": down_data.get("mid"),
            "pm_down_bid": down_data.get("bid"),
            "pm_down_ask": down_data.get("best_ask"),
            
            # === Events ===
            "ret_5s_bps": spikes.get("ret_5s_bps"),
            "z_ret_5s": spikes.get("z_ret_5s"),
            "down_spike": spikes.get("down_spike_5s"),
            "up_spike": spikes.get("up_spike_5s"),
            "up_dip_bps": pm_dips.get("up_dip_bps"),
            "down_dip_bps": pm_dips.get("down_dip_bps"),
            "up_dip": pm_dips.get("up_dip"),
            "down_dip": pm_dips.get("down_dip"),
        })
    
    def record_decision(self, decision: dict) -> None:
        """Record accumulate decision (S7 schema v3)."""
        # S6 Decision v2 fields
        net_fast = decision.get("net_edge_fast") or {}
        net_smooth = decision.get("net_edge_smooth") or {}
        
        self.record("decision", {
            # === S6 Decision v2 ===
            "candidate_side": decision.get("candidate_side"),
            "action": decision.get("action"),
            "confidence": decision.get("confidence"),
            "confidence_level": decision.get("confidence_level"),
            "confidence_reasons": ",".join(decision.get("confidence_reasons", [])[:5]),
            
            # Net edge fast/smooth
            "net_edge_fast_up": net_fast.get("up"),
            "net_edge_fast_down": net_fast.get("down"),
            "net_edge_smooth_up": net_smooth.get("up"),
            "net_edge_smooth_down": net_smooth.get("down"),
            
            # Legacy fields
            "fair_up": decision.get("fair_up"),
            "fair_down": decision.get("fair_down"),
            "market_up": decision.get("market_up"),
            "market_down": decision.get("market_down"),
            "edge_up_bps": decision.get("edge_up_bps"),
            "edge_down_bps": decision.get("edge_down_bps"),
            "net_edge_up": decision.get("net_edge_up"),
            "net_edge_down": decision.get("net_edge_down"),
            "triggers": decision.get("triggers"),
            "reasons": decision.get("reasons"),
            "veto_reasons": decision.get("veto_reasons"),
            
            # Bias context
            "bias_dir": decision.get("bias_dir"),
            "bias_strength": decision.get("bias_strength"),
        })
    
    async def _writer_loop(self) -> None:
        """Background task that processes queue and writes to files."""
        while self._running:
            try:
                # Get event from queue with timeout
                try:
                    market_id, event = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=0.1,
                    )
                except asyncio.TimeoutError:
                    # Check if any buffers need flushing
                    await self._check_flush_all()
                    continue
                
                # Stop signal
                if market_id == "__STOP__":
                    break
                
                # Add event to buffer
                if market_id in self._buffers:
                    self._buffers[market_id].append(event)
                    
                    # Check if buffer needs flushing
                    buf = self._buffers[market_id]
                    if buf.should_flush(self.buffer_size, self.flush_interval_ms):
                        await self._flush_buffer(buf)
                
            except Exception as e:
                logger.error(
                    "event_recorder_writer_error",
                    extra={"error": str(e)},
                )
    
    async def _flush_buffer(self, buf: EventBuffer) -> None:
        """Flush buffer to file."""
        if not buf.events:
            return
        
        try:
            # Open file if not open
            if buf.file_handle is None:
                buf.file_handle = open(buf.file_path, "a", encoding="utf-8")
            
            # Write events
            for event in buf.events:
                line = json.dumps(event, separators=(",", ":")) + "\n"
                buf.file_handle.write(line)
            
            buf.file_handle.flush()
            
            written = len(buf.events)
            buf.events_written += written
            self._total_written += written
            buf.events.clear()
            buf.last_flush_ms = now_ms()
            
        except Exception as e:
            logger.error(
                "event_recorder_flush_error",
                extra={
                    "market_id": buf.market_id,
                    "error": str(e),
                },
            )
    
    async def _check_flush_all(self) -> None:
        """Check all buffers for time-based flush."""
        for buf in self._buffers.values():
            if buf.should_flush(self.buffer_size, self.flush_interval_ms):
                await self._flush_buffer(buf)
    
    async def _flush_all(self) -> None:
        """Flush all buffers."""
        for buf in self._buffers.values():
            await self._flush_buffer(buf)
    
    def _close_all_files(self) -> None:
        """Close all open file handles."""
        for buf in self._buffers.values():
            if buf.file_handle:
                try:
                    buf.file_handle.close()
                except Exception:
                    pass
                buf.file_handle = None
    
    def get_stats(self) -> dict:
        """Get recording statistics."""
        return {
            "enabled": self.enabled,
            "running": self._running,
            "total_events": self._total_events,
            "total_written": self._total_written,
            "markets": len(self._buffers),
            "queue_size": self._queue.qsize(),
            "buffers": {
                mid: {
                    "file": str(buf.file_path),
                    "buffered": len(buf.events),
                    "written": buf.events_written,
                }
                for mid, buf in self._buffers.items()
            },
        }
