"""
TAAPI Async Client
==================

Async wrapper for TAAPI.IO API using aiohttp.

Features:
- Async bulk requests
- Rate limiting awareness
- Retry with backoff
- Error handling

Usage:
    async with TaapiAsyncClient(secret, exchange="binancefutures") as client:
        result = await client.bulk_indicators("BTC/USDT", "1h", indicators)
"""

import asyncio
import logging
from typing import Optional

import aiohttp

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# TAAPI API settings
TAAPI_BASE_URL = "https://api.taapi.io"
TAAPI_BULK_ENDPOINT = f"{TAAPI_BASE_URL}/bulk"
TAAPI_TIMEOUT = 30  # seconds
TAAPI_MAX_RETRIES = 3
TAAPI_RETRY_DELAYS = [5, 10, 15]  # seconds between retries


class TaapiError(Exception):
    """Base TAAPI error."""
    pass


class TaapiRateLimitError(TaapiError):
    """Rate limit exceeded."""
    def __init__(self, message: str, retry_after: int = 15):
        super().__init__(message)
        self.retry_after = retry_after


class TaapiAsyncClient:
    """
    Async client for TAAPI.IO API.
    
    Provides async bulk_indicators method for fetching multiple indicators
    in a single request.
    
    Args:
        secret: TAAPI API key
        exchange: Default exchange (binance, binancefutures)
        timeout: Request timeout in seconds
    """
    
    def __init__(
        self,
        secret: str,
        exchange: str = "binancefutures",
        timeout: int = TAAPI_TIMEOUT,
    ) -> None:
        self._secret = secret
        self._exchange = exchange
        self._timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting
        self._last_request_ts_ms: int = 0
        self._min_interval_ms: int = 1000  # 1 second minimum between requests
        
        logger.info(
            "taapi_async_client_initialized",
            extra={"exchange": exchange},
        )
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._timeout),
                headers={"Content-Type": "application/json"},
            )
        return self._session
    
    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info("taapi_async_client_closed")
    
    async def __aenter__(self) -> "TaapiAsyncClient":
        await self._ensure_session()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
    
    async def bulk_indicators(
        self,
        symbol: str,
        interval: str,
        indicators: list[dict],
        exchange: Optional[str] = None,
    ) -> dict:
        """
        Fetch multiple indicators in a single bulk request.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT")
            interval: Timeframe (1m, 5m, 15m, 1h, etc.)
            indicators: List of indicator dicts with 'indicator' key and optional params
            exchange: Override default exchange
            
        Returns:
            Dict with 'data' key containing list of results
            
        Raises:
            TaapiError: On API errors
            TaapiRateLimitError: On rate limit (429)
        """
        exchange = exchange or self._exchange
        
        # Rate limiting
        await self._rate_limit()
        
        payload = {
            "secret": self._secret,
            "construct": {
                "exchange": exchange,
                "symbol": symbol,
                "interval": interval,
                "indicators": indicators,
            },
        }
        
        start_ms = now_ms()
        
        for attempt in range(TAAPI_MAX_RETRIES):
            try:
                session = await self._ensure_session()
                
                async with session.post(TAAPI_BULK_ENDPOINT, json=payload) as response:
                    elapsed_ms = now_ms() - start_ms
                    
                    # Log request
                    logger.info(
                        "taapi_request",
                        extra={
                            "symbol": symbol,
                            "interval": interval,
                            "indicators_count": len(indicators),
                            "status": response.status,
                            "elapsed_ms": elapsed_ms,
                            "attempt": attempt + 1,
                        },
                    )
                    
                    # Handle rate limit
                    if response.status == 429:
                        if attempt < TAAPI_MAX_RETRIES - 1:
                            delay = TAAPI_RETRY_DELAYS[min(attempt, len(TAAPI_RETRY_DELAYS) - 1)]
                            logger.warning(
                                "taapi_rate_limit",
                                extra={"retry_after": delay, "attempt": attempt + 1},
                            )
                            await asyncio.sleep(delay)
                            continue
                        raise TaapiRateLimitError("Rate limit exceeded", retry_after=15)
                    
                    # Handle other errors
                    if response.status != 200:
                        text = await response.text()
                        raise TaapiError(f"TAAPI error {response.status}: {text}")
                    
                    # Parse response
                    data = await response.json()
                    
                    self._last_request_ts_ms = now_ms()
                    
                    return data
                    
            except aiohttp.ClientError as e:
                if attempt < TAAPI_MAX_RETRIES - 1:
                    delay = TAAPI_RETRY_DELAYS[min(attempt, len(TAAPI_RETRY_DELAYS) - 1)]
                    logger.warning(
                        "taapi_request_error",
                        extra={"error": str(e), "retry_after": delay, "attempt": attempt + 1},
                    )
                    await asyncio.sleep(delay)
                    continue
                raise TaapiError(f"Request failed: {e}")
        
        raise TaapiError("Max retries exceeded")
    
    async def _rate_limit(self) -> None:
        """Apply rate limiting between requests."""
        if self._last_request_ts_ms > 0:
            elapsed = now_ms() - self._last_request_ts_ms
            if elapsed < self._min_interval_ms:
                wait_ms = self._min_interval_ms - elapsed
                await asyncio.sleep(wait_ms / 1000.0)


def parse_bulk_response(response: dict) -> dict[str, any]:
    """
    Parse TAAPI bulk response into indicator dict.
    
    Args:
        response: Raw TAAPI response with 'data' key
        
    Returns:
        Dict of indicator_id -> value/dict
    """
    result = {}
    
    data = response.get("data", [])
    
    for item in data:
        indicator_id = item.get("id", "unknown")
        indicator_result = item.get("result", {})
        
        # Check for errors
        errors = item.get("errors", [])
        if errors:
            logger.warning(
                "taapi_indicator_error",
                extra={"id": indicator_id, "errors": errors},
            )
            continue
        
        # Extract value(s)
        if "value" in indicator_result:
            # Simple value indicator (RSI, EMA, ATR, etc.)
            result[indicator_id] = indicator_result["value"]
        else:
            # Multi-value indicator (MACD, Supertrend, etc.)
            result[indicator_id] = indicator_result
    
    return result
