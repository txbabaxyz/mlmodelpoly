"""
TAAPI Bootstrapper
==================

Bootstrap technical indicators from TAAPI at startup.

Fetches indicators for all timeframes (1m, 5m, 15m, 1h) to provide
immediate context without waiting for WS data to accumulate.

Usage:
    bootstrapper = TaapiBootstrapper(client, store, symbol="BTC/USDT")
    await bootstrapper.bootstrap()
"""

import asyncio
import logging
from typing import Optional

from collector.taapi.client import TaapiAsyncClient, parse_bulk_response, TaapiError
from collector.taapi.store import TaapiContextStore
from collector.taapi.indicator_sets import (
    INDICATOR_SETS,
    TAAPI_TIMEFRAMES,
    format_symbol_for_taapi,
)
from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


class TaapiBootstrapper:
    """
    Bootstrap TAAPI indicators at startup.
    
    Fetches indicators for all configured timeframes using bulk requests.
    Each timeframe is fetched with a single API call.
    
    Args:
        client: TaapiAsyncClient instance
        store: TaapiContextStore for storing results
        symbol: Trading symbol (e.g., "BTCUSDT")
        timeframes: List of timeframes to bootstrap (default: all)
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
        self._timeframes = timeframes or TAAPI_TIMEFRAMES
        
        self._bootstrap_done: bool = False
        self._bootstrap_ts_ms: Optional[int] = None
        self._errors: list[str] = []
        
        logger.info(
            "taapi_bootstrapper_initialized",
            extra={
                "symbol": symbol,
                "taapi_symbol": self._taapi_symbol,
                "timeframes": self._timeframes,
            },
        )
    
    async def bootstrap(self) -> bool:
        """
        Bootstrap indicators for all timeframes.
        
        Fetches indicators sequentially (one TF at a time) to avoid
        rate limit issues. Each TF uses a single bulk request.
        
        Returns:
            True if all timeframes succeeded, False otherwise
        """
        start_ms = now_ms()
        
        logger.info(
            "taapi_bootstrap_starting",
            extra={
                "symbol": self._symbol,
                "timeframes": self._timeframes,
            },
        )
        
        success_count = 0
        self._errors = []
        
        for tf in self._timeframes:
            try:
                await self._fetch_timeframe(tf)
                success_count += 1
                
                # Small delay between requests to be safe
                await asyncio.sleep(0.5)
                
            except TaapiError as e:
                error_msg = f"{tf}: {str(e)}"
                self._errors.append(error_msg)
                logger.error(
                    "taapi_bootstrap_tf_failed",
                    extra={"tf": tf, "error": str(e)},
                )
            except Exception as e:
                error_msg = f"{tf}: {type(e).__name__}: {str(e)}"
                self._errors.append(error_msg)
                logger.exception(
                    "taapi_bootstrap_tf_error",
                    extra={"tf": tf, "error": str(e)},
                )
        
        elapsed_ms = now_ms() - start_ms
        
        self._bootstrap_done = success_count == len(self._timeframes)
        self._bootstrap_ts_ms = now_ms()
        
        # Mark store as ready if at least some data was fetched
        if success_count > 0:
            self._store.mark_ready()
        
        logger.info(
            "taapi_bootstrap_done",
            extra={
                "success_count": success_count,
                "total_count": len(self._timeframes),
                "timeframes": self._timeframes,
                "elapsed_ms": elapsed_ms,
                "errors": self._errors if self._errors else None,
            },
        )
        
        return self._bootstrap_done
    
    async def _fetch_timeframe(self, tf: str) -> None:
        """
        Fetch indicators for a single timeframe.
        
        Args:
            tf: Timeframe (1m, 5m, 15m, 1h)
        """
        indicators = INDICATOR_SETS.get(tf, [])
        
        if not indicators:
            logger.warning(
                "taapi_bootstrap_no_indicators",
                extra={"tf": tf},
            )
            return
        
        logger.debug(
            "taapi_bootstrap_fetching_tf",
            extra={
                "tf": tf,
                "indicators_count": len(indicators),
            },
        )
        
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
        
        logger.info(
            "taapi_bootstrap_tf_done",
            extra={
                "tf": tf,
                "indicators_received": len(parsed),
                "indicators_requested": len(indicators),
            },
        )
    
    def is_ready(self) -> bool:
        """Check if bootstrap completed successfully."""
        return self._bootstrap_done
    
    def get_status(self) -> dict:
        """Get bootstrap status."""
        return {
            "done": self._bootstrap_done,
            "bootstrap_ts_ms": self._bootstrap_ts_ms,
            "timeframes": self._timeframes,
            "errors": self._errors if self._errors else None,
            "store_ready": self._store.is_ready(),
        }
