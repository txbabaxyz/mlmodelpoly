"""
Context Bootstrapper Module
===========================

Bootstrap HTF klines from Binance REST API on service startup.

Features:
- Fetches klines for multiple timeframes (1m, 5m, 15m, 1h)
- Fetches both futures and spot in parallel
- Uses RestClient with rate limiting and retry
- Tracks readiness status

Usage:
    bootstrapper = ContextBootstrapper(
        futures_client=futures_rest,
        spot_client=spot_rest,
        store=klines_store,
        symbol="BTCUSDT",
        timeframes=["1m", "5m", "15m", "1h"],
        limit=500,
        min_ready_bars=200,
    )
    
    await bootstrapper.bootstrap()
    
    if bootstrapper.is_ready():
        print("Context ready!")
"""

import asyncio
import logging
from typing import Optional

from collector.context_klines_store import KlinesStore
from collector.rest_client import RestClient
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


class ContextBootstrapper:
    """
    Bootstrap HTF klines from Binance REST API.
    
    Fetches historical klines for specified timeframes from both
    futures and spot markets on service startup.
    
    Args:
        futures_client: RestClient for futures API
        spot_client: RestClient for spot API
        store: KlinesStore to store fetched klines
        symbol: Trading pair symbol (e.g., "BTCUSDT")
        timeframes: List of timeframes to fetch (e.g., ["1m", "5m", "15m", "1h"])
        limit: Number of klines to fetch per request (default: 500)
        min_ready_bars: Minimum bars required for ready status (default: 200)
    """
    
    def __init__(
        self,
        futures_client: RestClient,
        spot_client: RestClient,
        store: KlinesStore,
        symbol: str,
        timeframes: list[str],
        limit: int = 500,
        min_ready_bars: int = 200,
    ) -> None:
        self._futures_client = futures_client
        self._spot_client = spot_client
        self._store = store
        self._symbol = symbol
        self._timeframes = timeframes
        self._limit = limit
        self._min_ready_bars = min_ready_bars
        
        self._bootstrap_done = False
        self._bootstrap_time_ms: Optional[int] = None
        self._errors: list[str] = []
        
        logger.info(
            "context_bootstrapper_initialized",
            extra={
                "symbol": symbol,
                "timeframes": timeframes,
                "limit": limit,
                "min_ready_bars": min_ready_bars,
            },
        )
    
    async def _fetch_klines(
        self,
        client: RestClient,
        market: str,
        tf: str,
    ) -> int:
        """
        Fetch klines for a single (market, tf) combination.
        
        Args:
            client: RestClient to use
            market: Market name ("futures" or "spot")
            tf: Timeframe
            
        Returns:
            Number of klines fetched
        """
        # Determine API path based on market
        if market == "futures":
            path = "/fapi/v1/klines"
        else:
            path = "/api/v3/klines"
        
        params = {
            "symbol": self._symbol,
            "interval": tf,
            "limit": self._limit,
        }
        
        start_ts = now_ms()
        data, headers = await client.get_json(path, params)
        elapsed_ms = now_ms() - start_ts
        
        if data is None:
            error_msg = f"Failed to fetch {market} {tf} klines"
            self._errors.append(error_msg)
            logger.warning(
                "context_fetch_failed",
                extra={
                    "market": market,
                    "tf": tf,
                    "elapsed_ms": elapsed_ms,
                },
            )
            return 0
        
        if not isinstance(data, list):
            error_msg = f"Invalid response for {market} {tf}: expected list"
            self._errors.append(error_msg)
            logger.warning(
                "context_fetch_invalid",
                extra={
                    "market": market,
                    "tf": tf,
                    "data_type": type(data).__name__,
                },
            )
            return 0
        
        # Upsert into store
        count = self._store.upsert_klines(market, tf, data)
        
        logger.info(
            "context_fetch_done",
            extra={
                "market": market,
                "tf": tf,
                "fetched": len(data),
                "stored": count,
                "elapsed_ms": elapsed_ms,
            },
        )
        
        return count
    
    async def bootstrap(self) -> bool:
        """
        Bootstrap all klines for all (market, tf) combinations.
        
        Fetches futures and spot klines for all configured timeframes
        in parallel (respecting RestClient's concurrency limits).
        
        Returns:
            True if all fetches succeeded, False if any failed
        """
        start_ts = now_ms()
        
        logger.info(
            "context_bootstrap_starting",
            extra={
                "symbol": self._symbol,
                "timeframes": self._timeframes,
                "limit": self._limit,
            },
        )
        
        # Create tasks for all (market, tf) combinations
        tasks = []
        task_info = []
        
        for tf in self._timeframes:
            # Futures task
            tasks.append(
                self._fetch_klines(self._futures_client, "futures", tf)
            )
            task_info.append(("futures", tf))
            
            # Spot task
            tasks.append(
                self._fetch_klines(self._spot_client, "spot", tf)
            )
            task_info.append(("spot", tf))
        
        # Execute all tasks in parallel
        # RestClient's Semaphore handles concurrency limiting
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        counts = {}
        all_success = True
        
        for (market, tf), result in zip(task_info, results):
            key = f"{market}_{tf}"
            
            if isinstance(result, Exception):
                logger.error(
                    "context_bootstrap_error",
                    extra={
                        "market": market,
                        "tf": tf,
                        "error": str(result),
                        "error_type": type(result).__name__,
                    },
                )
                counts[key] = 0
                all_success = False
            else:
                counts[key] = result
        
        elapsed_ms = now_ms() - start_ts
        self._bootstrap_done = True
        self._bootstrap_time_ms = elapsed_ms
        
        # Get final counts from store
        store_counts = self._store.get_counts()
        
        logger.info(
            "context_bootstrap_done",
            extra={
                "elapsed_ms": elapsed_ms,
                "counts": store_counts,
                "ready": self.is_ready(),
                "errors": len(self._errors),
            },
        )
        
        return all_success
    
    def is_ready(self) -> bool:
        """
        Check if context is ready (enough bars for all markets/timeframes).
        
        Returns:
            True if all (market, tf) combinations have >= min_ready_bars
        """
        if not self._bootstrap_done:
            return False
        
        for tf in self._timeframes:
            for market in ["futures", "spot"]:
                count = self._store.count(market, tf)
                if count < self._min_ready_bars:
                    return False
        
        return True
    
    def get_status(self) -> dict:
        """
        Get bootstrapper status.
        
        Returns:
            Dict with status information
        """
        return {
            "bootstrap_done": self._bootstrap_done,
            "bootstrap_time_ms": self._bootstrap_time_ms,
            "ready": self.is_ready(),
            "counts": self._store.get_counts(),
            "min_ready_bars": self._min_ready_bars,
            "errors": self._errors,
        }
