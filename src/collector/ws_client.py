"""
WebSocket Client Module
=======================

Universal async WebSocket client for Binance streams.
Handles connection, reconnection with backoff, and message routing.
Integrates with Metrics for observability.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

import orjson
import websockets
from websockets.client import WebSocketClientProtocol
from websockets.exceptions import (
    ConnectionClosed,
    ConnectionClosedError,
    ConnectionClosedOK,
)

from collector.utils_time import now_ms

if TYPE_CHECKING:
    from collector.metrics import Metrics

logger = logging.getLogger(__name__)

# Reconnection backoff sequence (seconds)
BACKOFF_SEQUENCE = [1, 2, 5, 10, 30]


class WSClient:
    """
    Universal WebSocket client for Binance streams.
    
    Features:
    - Combined streams endpoint support
    - Auto-reconnect with exponential backoff
    - Message parsing and queuing
    - Metrics integration for observability
    - Graceful shutdown
    
    Usage:
        queue = asyncio.Queue(maxsize=10000)
        metrics = Metrics()
        client = WSClient(
            name="futures",
            base_url="wss://fstream.binance.com",
            streams=["btcusdt@aggTrade"],
            out_queue=queue,
            symbol="BTCUSDT",
            metrics=metrics,
            market="futures"
        )
        await client.run_forever()
    """
    
    def __init__(
        self,
        name: str,
        base_url: str,
        streams: list[str],
        out_queue: asyncio.Queue,
        symbol: str,
        metrics: Optional["Metrics"] = None,
        market: Optional[str] = None,
    ):
        """
        Initialize WebSocket client.
        
        Args:
            name: Client identifier for logging (e.g., "futures", "spot")
            base_url: Base WebSocket URL (e.g., "wss://fstream.binance.com")
            streams: List of stream names (lowercase, e.g., ["btcusdt@aggTrade"])
            out_queue: Queue to put received messages
            symbol: Symbol for reference (uppercase)
            metrics: Optional Metrics instance for observability
            market: Market identifier for metrics ("futures" or "spot").
                    Defaults to name if not specified.
        """
        self.name = name
        self.base_url = base_url
        self.streams = streams
        self.out_queue = out_queue
        self.symbol = symbol
        self._metrics = metrics
        self._market = market or name  # Default to name if market not specified
        
        self._ws: Optional[WebSocketClientProtocol] = None
        self._running = False
        self._backoff_index = 0
        
        # Build combined streams URL
        self._url = self._build_url()
    
    def _build_url(self) -> str:
        """
        Build combined streams WebSocket URL.
        
        Returns:
            Full WebSocket URL with streams parameter.
        """
        streams_param = "/".join(self.streams)
        return f"{self.base_url}/stream?streams={streams_param}"
    
    def _get_backoff_delay(self) -> float:
        """
        Get current backoff delay and advance index.
        
        Returns:
            Delay in seconds before next reconnection attempt.
        """
        delay = BACKOFF_SEQUENCE[min(self._backoff_index, len(BACKOFF_SEQUENCE) - 1)]
        if self._backoff_index < len(BACKOFF_SEQUENCE) - 1:
            self._backoff_index += 1
        return delay
    
    def _reset_backoff(self) -> None:
        """Reset backoff index after successful connection."""
        self._backoff_index = 0
    
    def _mark_connected(self, connected: bool) -> None:
        """Update metrics connection status."""
        if self._metrics:
            self._metrics.mark_connected(self._market, connected)
    
    def _inc_reconnect(self) -> None:
        """Increment metrics reconnect counter."""
        if self._metrics:
            self._metrics.inc_reconnect(self._market)
    
    async def _connect(self) -> bool:
        """
        Establish WebSocket connection.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self._ws = await websockets.connect(
                self._url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
                max_size=10 * 1024 * 1024,  # 10MB max message size
            )
            
            logger.info(
                "ws_connected",
                extra={
                    "ws_name": self.name,
                    "market": self._market,
                    "url": self._url,
                    "symbol": self.symbol,
                },
            )
            
            self._reset_backoff()
            self._mark_connected(True)
            return True
            
        except Exception as e:
            logger.warning(
                "ws_connection_failed",
                extra={
                    "ws_name": self.name,
                    "market": self._market,
                    "url": self._url,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return False
    
    async def _receive_loop(self) -> None:
        """
        Main receive loop - reads messages and puts them in queue.
        """
        while self._running and self._ws:
            try:
                raw_message = await self._ws.recv()
                recv_ts_ms = now_ms()
                
                # Parse JSON
                try:
                    parsed = orjson.loads(raw_message)
                except orjson.JSONDecodeError as e:
                    logger.warning(
                        "ws_invalid_json",
                        extra={
                            "ws_name": self.name,
                            "error": str(e),
                            "raw_preview": raw_message[:200] if isinstance(raw_message, str) else str(raw_message)[:200],
                        },
                    )
                    continue
                
                # Create envelope
                envelope = {
                    "ws_name": self.name,
                    "recv_ts_ms": recv_ts_ms,
                    "msg": parsed,
                }
                
                # Put in queue with timeout to detect backpressure
                try:
                    await asyncio.wait_for(
                        self.out_queue.put(envelope),
                        timeout=1.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "ws_queue_backpressure",
                        extra={
                            "ws_name": self.name,
                            "queue_size": self.out_queue.qsize(),
                        },
                    )
                    # Still put it, but blocking
                    await self.out_queue.put(envelope)
                    
            except ConnectionClosedOK:
                logger.info(
                    "ws_disconnected",
                    extra={
                        "ws_name": self.name,
                        "market": self._market,
                        "reason": "connection_closed_ok",
                    },
                )
                self._mark_connected(False)
                break
                
            except ConnectionClosedError as e:
                logger.warning(
                    "ws_disconnected",
                    extra={
                        "ws_name": self.name,
                        "market": self._market,
                        "reason": "connection_closed_error",
                        "code": e.code if hasattr(e, 'code') else None,
                        "error": str(e),
                    },
                )
                self._mark_connected(False)
                break
                
            except ConnectionClosed as e:
                logger.warning(
                    "ws_disconnected",
                    extra={
                        "ws_name": self.name,
                        "market": self._market,
                        "reason": "connection_closed",
                        "error": str(e),
                    },
                )
                self._mark_connected(False)
                break
                
            except Exception as e:
                logger.exception(
                    "ws_receive_error",
                    extra={
                        "ws_name": self.name,
                        "error": str(e),
                        "error_type": type(e).__name__,
                    },
                )
                self._mark_connected(False)
                break
    
    async def run_forever(self) -> None:
        """
        Main entry point - connects and reconnects forever.
        
        Handles reconnection with exponential backoff.
        Runs until stop() is called or task is cancelled.
        """
        self._running = True
        
        logger.info(
            "ws_client_starting",
            extra={
                "ws_name": self.name,
                "market": self._market,
                "url": self._url,
                "streams": self.streams,
            },
        )
        
        while self._running:
            # Attempt connection
            connected = await self._connect()
            
            if connected:
                # Run receive loop until disconnection
                await self._receive_loop()
                
                # Clean up connection
                if self._ws:
                    try:
                        await self._ws.close()
                    except Exception:
                        pass
                    self._ws = None
            
            # Check if we should continue
            if not self._running:
                break
            
            # Increment reconnect counter before attempting reconnection
            self._inc_reconnect()
            
            # Wait before reconnection
            delay = self._get_backoff_delay()
            logger.info(
                "ws_reconnecting",
                extra={
                    "ws_name": self.name,
                    "market": self._market,
                    "delay_seconds": delay,
                },
            )
            
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break
        
        logger.info(
            "ws_client_stopped",
            extra={"ws_name": self.name, "market": self._market},
        )
    
    async def stop(self) -> None:
        """
        Stop the WebSocket client gracefully.
        """
        self._running = False
        self._mark_connected(False)
        
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
