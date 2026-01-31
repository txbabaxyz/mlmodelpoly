"""
REST Client Module
==================

Reliable REST client for Binance API with:
- Concurrency limiting (Semaphore)
- Minimum interval between requests (rate limiting)
- Retry with exponential backoff (429, 418, 5xx)
- Timeout handling
- Used weight header logging

Usage:
    client = RestClient(
        base_url="https://fapi.binance.com",
        name="futures",
        max_concurrency=4,
        min_interval_ms=150,
    )
    
    data, headers = await client.get_json("/fapi/v1/time")
"""

import asyncio
import logging
from typing import Any

import aiohttp

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)

# Backoff sequence in seconds
BACKOFF_SEQUENCE = [1, 2, 5, 10, 10]  # max 10s, 5 attempts
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT_SEC = 10

# HTTP codes that trigger retry
RETRY_STATUS_CODES = {429, 418, 500, 502, 503, 504}


class RestClient:
    """
    Reliable async REST client with rate limiting and retry logic.
    
    Features:
        - Concurrency limit via Semaphore
        - Minimum interval between requests (global rate limit)
        - Exponential backoff retry for 429, 418, 5xx
        - Timeout handling (10s default)
        - Logging of used weight headers
    
    Args:
        base_url: Base URL for requests (e.g., "https://fapi.binance.com")
        name: Client name for logging (e.g., "futures", "spot")
        max_concurrency: Maximum concurrent requests (default: 4)
        min_interval_ms: Minimum milliseconds between requests (default: 150)
    """
    
    def __init__(
        self,
        base_url: str,
        name: str,
        max_concurrency: int = 4,
        min_interval_ms: int = 150,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.name = name
        self.max_concurrency = max_concurrency
        self.min_interval_ms = min_interval_ms
        
        # Concurrency control
        self._semaphore = asyncio.Semaphore(max_concurrency)
        
        # Rate limiting
        self._rate_lock = asyncio.Lock()
        self._last_request_ts_ms: int = 0
        
        # Session (created lazily)
        self._session: aiohttp.ClientSession | None = None
        
        logger.info(
            "rest_client_initialized",
            extra={
                "rest_name": self.name,
                "base_url": self.base_url,
                "max_concurrency": self.max_concurrency,
                "min_interval_ms": self.min_interval_ms,
            },
        )
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_SEC)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
            logger.info(
                "rest_client_closed",
                extra={"rest_name": self.name},
            )
    
    async def _wait_for_rate_limit(self) -> None:
        """Wait if minimum interval hasn't passed since last request."""
        async with self._rate_lock:
            current_ts = now_ms()
            elapsed = current_ts - self._last_request_ts_ms
            
            if elapsed < self.min_interval_ms:
                wait_ms = self.min_interval_ms - elapsed
                await asyncio.sleep(wait_ms / 1000.0)
            
            self._last_request_ts_ms = now_ms()
    
    @staticmethod
    def _extract_used_weight(headers: dict[str, str]) -> dict[str, str]:
        """
        Extract all headers containing 'USED-WEIGHT' (case-insensitive).
        
        Binance returns headers like:
            X-MBX-USED-WEIGHT-1M
            X-SAPI-USED-IP-WEIGHT-1M
        """
        used_weight = {}
        for key, value in headers.items():
            if "USED-WEIGHT" in key.upper():
                used_weight[key] = value
        return used_weight
    
    async def get_json(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> tuple[Any | None, dict[str, str]]:
        """
        Make GET request and return JSON response.
        
        Args:
            path: API path (e.g., "/fapi/v1/time")
            params: Optional query parameters
        
        Returns:
            Tuple of (data, headers):
                - data: Parsed JSON (dict or list), or None on error
                - headers: Response headers as dict
        
        Retries on:
            - HTTP 429 (rate limit)
            - HTTP 418 (IP ban warning)
            - HTTP 5xx (server errors)
            - Connection/timeout errors
        
        No retry on:
            - HTTP 4xx (except 429, 418)
        """
        url = f"{self.base_url}{path}"
        headers_result: dict[str, str] = {}
        
        async with self._semaphore:
            for attempt in range(1, MAX_ATTEMPTS + 1):
                start_ts = now_ms()
                status = 0
                error_msg = None
                
                try:
                    # Wait for rate limit
                    await self._wait_for_rate_limit()
                    
                    # Make request
                    session = await self._get_session()
                    async with session.get(url, params=params) as response:
                        status = response.status
                        headers_result = dict(response.headers)
                        elapsed_ms = now_ms() - start_ts
                        
                        # Extract used weight
                        used_weight = self._extract_used_weight(headers_result)
                        
                        # Log request
                        logger.info(
                            "rest_request",
                            extra={
                                "rest_name": self.name,
                                "path": path,
                                "status": status,
                                "elapsed_ms": elapsed_ms,
                                "attempts": attempt,
                                "used_weight": used_weight if used_weight else None,
                            },
                        )
                        
                        # Success
                        if 200 <= status < 300:
                            data = await response.json()
                            return data, headers_result
                        
                        # Retry on specific status codes
                        if status in RETRY_STATUS_CODES:
                            error_msg = f"HTTP {status}"
                            # Continue to retry logic below
                        else:
                            # 4xx (except 429, 418) - no retry
                            logger.warning(
                                "rest_request_failed",
                                extra={
                                    "rest_name": self.name,
                                    "path": path,
                                    "status": status,
                                    "attempts": attempt,
                                    "reason": "client_error_no_retry",
                                },
                            )
                            return None, headers_result
                
                except asyncio.TimeoutError:
                    elapsed_ms = now_ms() - start_ts
                    error_msg = "timeout"
                    logger.warning(
                        "rest_request_timeout",
                        extra={
                            "rest_name": self.name,
                            "path": path,
                            "elapsed_ms": elapsed_ms,
                            "attempts": attempt,
                        },
                    )
                
                except aiohttp.ClientError as e:
                    elapsed_ms = now_ms() - start_ts
                    error_msg = str(e)
                    logger.warning(
                        "rest_request_error",
                        extra={
                            "rest_name": self.name,
                            "path": path,
                            "error": error_msg,
                            "error_type": type(e).__name__,
                            "elapsed_ms": elapsed_ms,
                            "attempts": attempt,
                        },
                    )
                
                except Exception as e:
                    elapsed_ms = now_ms() - start_ts
                    error_msg = str(e)
                    logger.exception(
                        "rest_request_unexpected_error",
                        extra={
                            "rest_name": self.name,
                            "path": path,
                            "error": error_msg,
                            "error_type": type(e).__name__,
                            "elapsed_ms": elapsed_ms,
                            "attempts": attempt,
                        },
                    )
                
                # Retry logic with backoff
                if attempt < MAX_ATTEMPTS:
                    backoff_idx = min(attempt - 1, len(BACKOFF_SEQUENCE) - 1)
                    backoff_sec = BACKOFF_SEQUENCE[backoff_idx]
                    
                    logger.info(
                        "rest_request_retry",
                        extra={
                            "rest_name": self.name,
                            "path": path,
                            "attempt": attempt,
                            "next_attempt": attempt + 1,
                            "backoff_sec": backoff_sec,
                            "error": error_msg,
                        },
                    )
                    
                    await asyncio.sleep(backoff_sec)
                else:
                    # Max attempts reached
                    logger.error(
                        "rest_request_max_retries",
                        extra={
                            "rest_name": self.name,
                            "path": path,
                            "attempts": attempt,
                            "error": error_msg,
                        },
                    )
        
        return None, headers_result
    
    async def __aenter__(self) -> "RestClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
