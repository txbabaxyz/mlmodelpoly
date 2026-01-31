"""
Polymarket WebSocket Client
===========================

Async WebSocket client for Polymarket CLOB orderbook data.

The client:
- Connects to Polymarket WS and subscribes to market tokens
- Parses "book" events and updates PolymarketBookStore
- Handles automatic reconnection on market rotation (every 15 min)
- Does NOT perform any trading logic

Architecture:
    PolymarketWSClient → on_book_event → PolymarketBookStore.update_yes/no()

Usage:
    store = PolymarketBookStore()
    resolver = MarketResolver()
    client = PolymarketWSClient(store, resolver)
    
    # Run in asyncio event loop
    await client.run_forever(shutdown_event)
"""

import asyncio
import json
import logging
from typing import Optional

import websockets
from websockets.exceptions import ConnectionClosed

from collector.polymarket.book_store import PolymarketBookStore
from collector.polymarket.market_resolver import MarketResolver
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


# Polymarket WebSocket endpoint
PM_WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"

# Reconnect backoff sequence (seconds)
RECONNECT_BACKOFF = [1, 2, 5, 10, 30]


class PolymarketWSClient:
    """
    WebSocket client for Polymarket CLOB orderbook.
    
    Connects to Polymarket WS, subscribes to current BTC UP/DOWN tokens,
    and feeds orderbook updates to PolymarketBookStore.
    
    Handles:
    - Automatic market rotation every 15 minutes
    - Reconnection with exponential backoff
    - Graceful shutdown
    
    Args:
        store: PolymarketBookStore to update with orderbook data
        resolver: MarketResolver for fetching token IDs
        
    Thread Safety:
        Runs in asyncio event loop. Store methods are thread-safe.
    """
    
    def __init__(
        self,
        store: PolymarketBookStore,
        resolver: MarketResolver,
    ) -> None:
        self.store = store
        self.resolver = resolver
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._current_tokens: Optional[dict] = None
        self._reconnect_count = 0
        self._stop_requested = False
        
        logger.info("polymarket_ws_client_initialized")
    
    async def run_forever(self, shutdown_event: asyncio.Event) -> None:
        """
        Main run loop. Connects, subscribes, and handles market rotation.
        
        Args:
            shutdown_event: Event to signal shutdown
        """
        logger.info("polymarket_ws_starting")
        
        while not shutdown_event.is_set() and not self._stop_requested:
            try:
                # Resolve current market tokens
                tokens = await self.resolver.get_current_btc_tokens()
                
                if not tokens or not tokens.get("up") or not tokens.get("down"):
                    logger.warning(
                        "polymarket_tokens_unavailable",
                        extra={"retrying_in": 5},
                    )
                    await asyncio.sleep(5)
                    continue
                
                self._current_tokens = tokens
                
                # Update store with new market
                self.store.set_market(
                    market_id=tokens["market_id"],
                    yes_token_id=tokens["up"],
                    no_token_id=tokens["down"],
                )
                
                # Calculate time until market ends
                _, seconds_left = self.resolver.time_to_market_end()
                
                logger.info(
                    "polymarket_connecting",
                    extra={
                        "market_id": tokens["market_id"],
                        "seconds_until_rotation": seconds_left,
                    },
                )
                
                # Connect and run until market rotation or error
                await self._connect_and_run(shutdown_event, seconds_left)
                
                # If we exited cleanly for market rotation, reset reconnect count
                self._reconnect_count = 0
                
            except asyncio.CancelledError:
                logger.info("polymarket_ws_cancelled")
                break
            except Exception as e:
                logger.exception(
                    "polymarket_ws_error",
                    extra={"error": str(e), "reconnect_count": self._reconnect_count},
                )
                
                # Backoff before retry
                backoff = RECONNECT_BACKOFF[min(self._reconnect_count, len(RECONNECT_BACKOFF) - 1)]
                self._reconnect_count += 1
                
                logger.info(
                    "polymarket_ws_reconnecting",
                    extra={"backoff_sec": backoff, "attempt": self._reconnect_count},
                )
                
                # Mark disconnected
                self.store.set_connected(False)
                
                # Wait before retry (check shutdown periodically)
                for _ in range(backoff):
                    if shutdown_event.is_set():
                        break
                    await asyncio.sleep(1)
        
        # Final cleanup
        self.store.set_connected(False)
        logger.info("polymarket_ws_stopped")
    
    async def _connect_and_run(
        self,
        shutdown_event: asyncio.Event,
        max_duration_sec: int,
    ) -> None:
        """
        Connect to WS, subscribe, and receive messages.
        
        Args:
            shutdown_event: Event to signal shutdown
            max_duration_sec: Maximum time to run before market rotation
        """
        if not self._current_tokens:
            return
        
        tokens = self._current_tokens
        
        try:
            async with websockets.connect(
                PM_WS_URL,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=5,
            ) as ws:
                self._ws = ws
                
                # Send subscription message
                sub_msg = {
                    "auth": {},
                    "type": "MARKET",
                    "assets_ids": [tokens["up"], tokens["down"]],
                }
                await ws.send(json.dumps(sub_msg))
                
                logger.info(
                    "polymarket_ws_connected",
                    extra={"market_id": tokens["market_id"]},
                )
                
                # Mark connected
                self.store.set_connected(True)
                self._reconnect_count = 0
                
                # Create timeout task for market rotation
                async def rotation_timer():
                    # Add 2 seconds buffer after market end
                    await asyncio.sleep(max_duration_sec + 2)
                
                rotation_task = asyncio.create_task(rotation_timer())
                receive_task = asyncio.create_task(self._receive_loop(ws, shutdown_event))
                
                # Wait for either: shutdown, rotation timeout, or receive error
                done, pending = await asyncio.wait(
                    [rotation_task, receive_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                
                # Cancel pending tasks
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                
                # Check if rotation triggered
                if rotation_task in done:
                    logger.info(
                        "polymarket_market_rotation",
                        extra={"old_market": tokens["market_id"]},
                    )
                
        except ConnectionClosed as e:
            logger.warning(
                "polymarket_ws_closed",
                extra={"code": e.code, "reason": e.reason},
            )
        finally:
            self._ws = None
            self.store.set_connected(False)
    
    async def _receive_loop(
        self,
        ws: websockets.WebSocketClientProtocol,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Receive and process WS messages.
        """
        while not shutdown_event.is_set():
            try:
                # Receive with timeout to check shutdown
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                except asyncio.TimeoutError:
                    continue
                
                # Parse and process
                self._on_message(message)
                
            except ConnectionClosed:
                raise
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(
                    "polymarket_ws_message_error",
                    extra={"error": str(e)},
                )
    
    def _on_message(self, message: str) -> None:
        """
        Process incoming WS message.
        
        Expected format:
        {
            "event_type": "book",
            "asset_id": "0x...",
            "bids": [{"price": "0.52", "size": "1000"}, ...],
            "asks": [{"price": "0.54", "size": "800"}, ...]
        }
        """
        try:
            data = json.loads(message)
            
            event_type = data.get("event_type")
            if event_type != "book":
                return
            
            asset_id = data.get("asset_id", "")
            bids_raw = data.get("bids", [])
            asks_raw = data.get("asks", [])
            
            # Parse orderbook
            bids = self._parse_orders(bids_raw)
            asks = self._parse_orders(asks_raw)
            
            ts = now_ms()
            
            # Update store by token ID
            matched = self.store.update_by_token_id(asset_id, bids, asks, ts)
            
            if not matched:
                # Token doesn't match current market (might be stale)
                pass
                
        except json.JSONDecodeError as e:
            logger.warning(
                "polymarket_ws_json_error",
                extra={"error": str(e)},
            )
        except Exception as e:
            logger.warning(
                "polymarket_ws_parse_error",
                extra={"error": str(e)},
            )
    
    @staticmethod
    def _parse_orders(orders_raw: list) -> list[tuple[float, float]]:
        """
        Parse bids or asks into (price, size) tuples.
        
        Handles both dict and list formats:
        - [{"price": "0.52", "size": "1000"}, ...]
        - [[0.52, 1000], ...]
        """
        orders = []
        for order in orders_raw or []:
            try:
                if isinstance(order, dict):
                    price = float(order.get("price", 0))
                    size = float(order.get("size", 0))
                else:
                    price = float(order[0])
                    size = float(order[1]) if len(order) > 1 else 0.0
                
                if price > 0 and size > 0:
                    orders.append((price, size))
            except (ValueError, IndexError, TypeError):
                continue
        
        return orders
    
    async def stop(self) -> None:
        """Request graceful stop."""
        self._stop_requested = True
        if self._ws:
            await self._ws.close()
