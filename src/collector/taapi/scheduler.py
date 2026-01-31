"""
TAAPI Scheduler
===============

Periodic updates of TAAPI indicators during the trading session.

Schedule:
    - 1m:  every 15s (micro stage inside window)
    - 5m:  every 30s (local bias)
    - 15m: every 60s (regime doesn't change fast)
    - 1h:  every 180s (background)

Calls are staggered to avoid >1 bulk per 15 seconds.

Usage:
    scheduler = TaapiScheduler(client, store, symbol="BTCUSDT")
    task = asyncio.create_task(scheduler.run_forever(shutdown_event))
"""

import asyncio
import logging
from typing import Optional

from collector.taapi.client import TaapiAsyncClient, parse_bulk_response, TaapiError
from collector.taapi.store import TaapiContextStore
from collector.taapi.indicator_sets import (
    INDICATOR_SETS,
    format_symbol_for_taapi,
)
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


# Update intervals per timeframe (seconds)
# Staggered to avoid multiple calls in the same 15-second window
TAAPI_UPDATE_INTERVALS = {
    "1m": 15,    # Every 15s - micro stage
    "5m": 30,    # Every 30s - local bias
    "15m": 60,   # Every 60s - regime
    "1h": 180,   # Every 180s (3min) - background
}

# Initial delays to stagger calls (seconds)
# This ensures calls don't overlap
TAAPI_INITIAL_DELAYS = {
    "1m": 0,     # Start immediately
    "5m": 5,     # Start after 5s
    "15m": 10,   # Start after 10s
    "1h": 15,    # Start after 15s (in next window)
}


class TaapiScheduler:
    """
    Scheduler for periodic TAAPI indicator updates.
    
    Runs separate update loops per timeframe with staggered intervals
    to stay within API rate limits.
    
    Args:
        client: TaapiAsyncClient instance
        store: TaapiContextStore for storing results
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframes: List of timeframes to update (default: all)
    """
    
    def __init__(
        self,
        client: TaapiAsyncClient,
        store: TaapiContextStore,
        symbol: str,
        timeframes: Optional[list[str]] = None,
    ) -> None:
        self._client = client
        self._store = store
        self._symbol = symbol
        self._taapi_symbol = format_symbol_for_taapi(symbol)
        self._timeframes = timeframes or list(TAAPI_UPDATE_INTERVALS.keys())
        
        # Metrics
        self._update_counts: dict[str, int] = {tf: 0 for tf in self._timeframes}
        self._error_counts: dict[str, int] = {tf: 0 for tf in self._timeframes}
        self._last_update_ts: dict[str, int] = {}
        
        # State logging interval
        self._state_log_interval_sec = 60
        self._last_state_log_ts = 0
        
        logger.info(
            "taapi_scheduler_initialized",
            extra={
                "symbol": symbol,
                "taapi_symbol": self._taapi_symbol,
                "timeframes": self._timeframes,
                "intervals": {tf: TAAPI_UPDATE_INTERVALS.get(tf) for tf in self._timeframes},
            },
        )
    
    async def run_forever(self, shutdown_event: asyncio.Event) -> None:
        """
        Run scheduler until shutdown.
        
        Spawns separate update tasks for each timeframe with
        staggered initial delays.
        
        Args:
            shutdown_event: Event to signal shutdown
        """
        logger.info("taapi_scheduler_starting")
        
        tasks = []
        
        for tf in self._timeframes:
            task = asyncio.create_task(
                self._tf_update_loop(tf, shutdown_event),
                name=f"taapi_updater_{tf}",
            )
            tasks.append(task)
        
        # State logger task
        state_task = asyncio.create_task(
            self._state_logger_loop(shutdown_event),
            name="taapi_scheduler_state_logger",
        )
        tasks.append(state_task)
        
        logger.info(
            "taapi_scheduler_started",
            extra={
                "timeframes": self._timeframes,
                "tasks_count": len(tasks),
            },
        )
        
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            logger.info("taapi_scheduler_cancelled")
        finally:
            logger.info("taapi_scheduler_stopped")
    
    async def _tf_update_loop(self, tf: str, shutdown_event: asyncio.Event) -> None:
        """
        Update loop for a single timeframe.
        
        Args:
            tf: Timeframe to update
            shutdown_event: Event to signal shutdown
        """
        interval_sec = TAAPI_UPDATE_INTERVALS.get(tf, 60)
        initial_delay = TAAPI_INITIAL_DELAYS.get(tf, 0)
        
        logger.info(
            "taapi_tf_updater_starting",
            extra={
                "tf": tf,
                "interval_sec": interval_sec,
                "initial_delay_sec": initial_delay,
            },
        )
        
        # Initial delay to stagger calls
        if initial_delay > 0:
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=initial_delay,
                )
                return  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Continue to update
        
        while not shutdown_event.is_set():
            try:
                await self._update_tf(tf)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._error_counts[tf] += 1
                logger.error(
                    "taapi_update_error",
                    extra={
                        "tf": tf,
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "error_count": self._error_counts[tf],
                    },
                )
            
            # Wait for next interval or shutdown
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=interval_sec,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Continue to next update
        
        logger.info(
            "taapi_tf_updater_stopped",
            extra={
                "tf": tf,
                "total_updates": self._update_counts[tf],
                "total_errors": self._error_counts[tf],
            },
        )
    
    async def _update_tf(self, tf: str) -> None:
        """
        Fetch and store indicators for a timeframe.
        
        Args:
            tf: Timeframe to update
        """
        indicators = INDICATOR_SETS.get(tf, [])
        
        if not indicators:
            return
        
        start_ms = now_ms()
        
        # Fetch via bulk request
        response = await self._client.bulk_indicators(
            symbol=self._taapi_symbol,
            interval=tf,
            indicators=indicators,
        )
        
        # Parse response
        parsed = parse_bulk_response(response)
        
        # Store results
        self._store.update(tf, parsed)
        
        # Update metrics
        self._update_counts[tf] += 1
        self._last_update_ts[tf] = now_ms()
        
        elapsed_ms = now_ms() - start_ms
        
        logger.debug(
            "taapi_scheduled_update",
            extra={
                "tf": tf,
                "indicators_count": len(parsed),
                "elapsed_ms": elapsed_ms,
                "update_count": self._update_counts[tf],
            },
        )
    
    async def _state_logger_loop(self, shutdown_event: asyncio.Event) -> None:
        """
        Periodically log scheduler state.
        
        Args:
            shutdown_event: Event to signal shutdown
        """
        while not shutdown_event.is_set():
            try:
                await asyncio.wait_for(
                    shutdown_event.wait(),
                    timeout=self._state_log_interval_sec,
                )
                break  # Shutdown requested
            except asyncio.TimeoutError:
                pass  # Log state
            
            self._log_state()
    
    def _log_state(self) -> None:
        """Log current scheduler state."""
        ts = now_ms()
        
        state = {
            tf: {
                "updates": self._update_counts[tf],
                "errors": self._error_counts[tf],
                "age_sec": round((ts - self._last_update_ts.get(tf, 0)) / 1000, 1) if tf in self._last_update_ts else None,
            }
            for tf in self._timeframes
        }
        
        total_updates = sum(self._update_counts.values())
        total_errors = sum(self._error_counts.values())
        
        logger.info(
            "taapi_scheduler_state",
            extra={
                "total_updates": total_updates,
                "total_errors": total_errors,
                "timeframes": state,
            },
        )
    
    def get_stats(self) -> dict:
        """
        Get scheduler statistics.
        
        Returns:
            Dict with update counts and errors per TF
        """
        ts = now_ms()
        
        return {
            "timeframes": {
                tf: {
                    "updates": self._update_counts[tf],
                    "errors": self._error_counts[tf],
                    "interval_sec": TAAPI_UPDATE_INTERVALS.get(tf),
                    "last_update_age_sec": round((ts - self._last_update_ts.get(tf, 0)) / 1000, 1) if tf in self._last_update_ts else None,
                }
                for tf in self._timeframes
            },
            "total_updates": sum(self._update_counts.values()),
            "total_errors": sum(self._error_counts.values()),
        }
