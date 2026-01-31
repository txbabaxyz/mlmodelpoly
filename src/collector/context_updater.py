"""
Context Updater Module
======================

Incremental updates for HTF klines context after bootstrap.

Features:
- Minimal API usage (limit=2 per request)
- Different update intervals per timeframe
- Metrics tracking (updates count, last update time)
- Periodic state logging

Update Schedule:
    1m  - every 10s
    5m  - every 30s
    15m - every 60s
    1h  - every 300s (5 min)

Usage:
    updater = ContextUpdater(
        futures_client=futures_rest,
        spot_client=spot_rest,
        store=klines_store,
        symbol="BTCUSDT",
        timeframes=["1m", "5m", "15m", "1h"],
    )
    
    # Run as background task
    task = asyncio.create_task(updater.run_forever(shutdown_event))
"""

import asyncio
import logging
from typing import Optional

from collector.context_klines_store import KlinesStore
from collector.rest_client import RestClient
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# Update intervals per timeframe (in seconds)
UPDATE_INTERVALS: dict[str, int] = {
    "1m": 10,      # Every 10 seconds
    "5m": 30,      # Every 30 seconds
    "15m": 60,     # Every 60 seconds
    "1h": 300,     # Every 5 minutes
    "4h": 600,     # Every 10 minutes (if used)
    "1d": 3600,    # Every hour (if used)
}

# Default interval for unknown timeframes
DEFAULT_UPDATE_INTERVAL = 60

# Fetch limit - minimal to save API weight
FETCH_LIMIT = 2

# State logging interval
STATE_LOG_INTERVAL_SEC = 60


class ContextUpdater:
    """
    Incremental updater for HTF klines context.
    
    After bootstrap, keeps context up-to-date with minimal API calls.
    Uses limit=2 to fetch only the latest klines.
    
    Args:
        futures_client: RestClient for futures API
        spot_client: RestClient for spot API
        store: KlinesStore to update
        symbol: Trading pair symbol
        timeframes: List of timeframes to update
    """
    
    def __init__(
        self,
        futures_client: RestClient,
        spot_client: RestClient,
        store: KlinesStore,
        symbol: str,
        timeframes: list[str],
    ) -> None:
        self._futures_client = futures_client
        self._spot_client = spot_client
        self._store = store
        self._symbol = symbol
        self._timeframes = timeframes
        
        # Metrics per (market, tf)
        self._updates_total: dict[tuple[str, str], int] = {}
        self._last_update_ms: dict[tuple[str, str], int] = {}
        self._errors_total: dict[tuple[str, str], int] = {}
        
        # Last state log time
        self._last_state_log_ms: int = 0
        
        # Initialize metrics
        for tf in timeframes:
            for market in ["futures", "spot"]:
                key = (market, tf)
                self._updates_total[key] = 0
                self._last_update_ms[key] = 0
                self._errors_total[key] = 0
        
        logger.info(
            "context_updater_initialized",
            extra={
                "symbol": symbol,
                "timeframes": timeframes,
                "intervals": {tf: UPDATE_INTERVALS.get(tf, DEFAULT_UPDATE_INTERVAL) for tf in timeframes},
                "fetch_limit": FETCH_LIMIT,
            },
        )
    
    def _get_interval(self, tf: str) -> int:
        """Get update interval for timeframe."""
        return UPDATE_INTERVALS.get(tf, DEFAULT_UPDATE_INTERVAL)
    
    async def _update_klines(
        self,
        client: RestClient,
        market: str,
        tf: str,
    ) -> bool:
        """
        Fetch and update klines for a single (market, tf).
        
        Args:
            client: RestClient to use
            market: Market name
            tf: Timeframe
            
        Returns:
            True if update succeeded
        """
        key = (market, tf)
        
        # Determine API path
        if market == "futures":
            path = "/fapi/v1/klines"
        else:
            path = "/api/v3/klines"
        
        params = {
            "symbol": self._symbol,
            "interval": tf,
            "limit": FETCH_LIMIT,
        }
        
        try:
            data, headers = await client.get_json(path, params)
            
            if data is None:
                self._errors_total[key] = self._errors_total.get(key, 0) + 1
                return False
            
            if not isinstance(data, list):
                self._errors_total[key] = self._errors_total.get(key, 0) + 1
                return False
            
            # Upsert into store
            updated = self._store.upsert_klines(market, tf, data)
            
            # Update metrics
            self._updates_total[key] = self._updates_total.get(key, 0) + 1
            self._last_update_ms[key] = now_ms()
            
            logger.debug(
                "context_update_done",
                extra={
                    "market": market,
                    "tf": tf,
                    "fetched": len(data),
                    "updated": updated,
                },
            )
            
            return True
            
        except Exception as e:
            self._errors_total[key] = self._errors_total.get(key, 0) + 1
            logger.warning(
                "context_update_error",
                extra={
                    "market": market,
                    "tf": tf,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return False
    
    async def _run_tf_updater(
        self,
        tf: str,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Run updater loop for a single timeframe.
        
        Updates both futures and spot at the specified interval.
        
        Args:
            tf: Timeframe to update
            shutdown_event: Event to signal shutdown
        """
        interval = self._get_interval(tf)
        
        logger.info(
            "context_tf_updater_started",
            extra={"tf": tf, "interval_sec": interval},
        )
        
        while not shutdown_event.is_set():
            try:
                # Wait for interval or shutdown
                try:
                    await asyncio.wait_for(
                        shutdown_event.wait(),
                        timeout=interval,
                    )
                    # Shutdown signaled
                    break
                except asyncio.TimeoutError:
                    # Normal timeout, proceed with update
                    pass
                
                # Update futures
                await self._update_klines(self._futures_client, "futures", tf)
                
                # Small delay between requests
                await asyncio.sleep(0.1)
                
                # Update spot
                await self._update_klines(self._spot_client, "spot", tf)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(
                    "context_tf_updater_error",
                    extra={"tf": tf, "error": str(e)},
                )
                # Wait a bit before retrying
                await asyncio.sleep(5)
        
        logger.info(
            "context_tf_updater_stopped",
            extra={"tf": tf},
        )
    
    async def _state_logger(
        self,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Periodically log updater state.
        
        Args:
            shutdown_event: Event to signal shutdown
        """
        while not shutdown_event.is_set():
            try:
                await asyncio.sleep(STATE_LOG_INTERVAL_SEC)
                
                if shutdown_event.is_set():
                    break
                
                self._log_state()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(
                    "context_state_logger_error",
                    extra={"error": str(e)},
                )
    
    def _log_state(self) -> None:
        """Log current updater state."""
        current_ms = now_ms()
        
        # Build state summary
        updates_by_tf: dict[str, dict[str, int]] = {}
        last_update_by_tf: dict[str, dict[str, int]] = {}
        errors_by_tf: dict[str, dict[str, int]] = {}
        
        for (market, tf), count in self._updates_total.items():
            if tf not in updates_by_tf:
                updates_by_tf[tf] = {}
            updates_by_tf[tf][market] = count
        
        for (market, tf), ts in self._last_update_ms.items():
            if tf not in last_update_by_tf:
                last_update_by_tf[tf] = {}
            # Convert to seconds ago
            last_update_by_tf[tf][market] = (current_ms - ts) // 1000 if ts > 0 else -1
        
        for (market, tf), count in self._errors_total.items():
            if count > 0:
                if tf not in errors_by_tf:
                    errors_by_tf[tf] = {}
                errors_by_tf[tf][market] = count
        
        # Total updates
        total_updates = sum(self._updates_total.values())
        total_errors = sum(self._errors_total.values())
        
        logger.info(
            "context_updater_state",
            extra={
                "total_updates": total_updates,
                "total_errors": total_errors,
                "updates_by_tf": updates_by_tf,
                "last_update_sec_ago": last_update_by_tf,
                "store_counts": self._store.get_counts(),
            },
        )
    
    async def run_forever(
        self,
        shutdown_event: asyncio.Event,
    ) -> None:
        """
        Run all updater loops forever.
        
        Creates a separate task for each timeframe plus state logger.
        
        Args:
            shutdown_event: Event to signal shutdown
        """
        logger.info("context_updater_starting")
        
        # Create tasks for each timeframe
        tasks = []
        
        for tf in self._timeframes:
            task = asyncio.create_task(
                self._run_tf_updater(tf, shutdown_event),
                name=f"context_updater_{tf}",
            )
            tasks.append(task)
        
        # Add state logger task
        state_logger_task = asyncio.create_task(
            self._state_logger(shutdown_event),
            name="context_state_logger",
        )
        tasks.append(state_logger_task)
        
        logger.info(
            "context_updater_started",
            extra={
                "timeframes": self._timeframes,
                "tasks_count": len(tasks),
            },
        )
        
        # Wait for all tasks
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            pass
        
        # Log final state
        self._log_state()
        
        logger.info("context_updater_stopped")
    
    def get_metrics(self) -> dict:
        """
        Get updater metrics.
        
        Returns:
            Dict with updates_total, last_update_ms, errors_total
        """
        return {
            "updates_total": {
                f"{market}_{tf}": count
                for (market, tf), count in self._updates_total.items()
            },
            "last_update_ms": {
                f"{market}_{tf}": ts
                for (market, tf), ts in self._last_update_ms.items()
            },
            "errors_total": {
                f"{market}_{tf}": count
                for (market, tf), count in self._errors_total.items()
                if count > 0
            },
        }
