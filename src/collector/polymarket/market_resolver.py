"""
Polymarket Market Resolver
==========================

Resolves current market and token IDs via Polymarket Gamma API.

The BTC 15-minute market rotates every 15 minutes:
- Market slug: btc-updown-15m-{slot} where slot = unix_timestamp // 900
- Each market has UP and DOWN tokens with unique CLOB token IDs

Usage:
    resolver = MarketResolver()
    
    # Get current market tokens
    tokens = await resolver.get_current_btc_tokens()
    # {"market_id": "btc-updown-15m-1234567890", "up": "0x...", "down": "0x..."}
    
    # Calculate time to market end
    time_left, seconds = resolver.time_to_market_end()
"""

import json
import logging
from typing import Optional

import aiohttp

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


# Polymarket API endpoints
PM_GAMMA_API = "https://gamma-api.polymarket.com"


class MarketResolver:
    """
    Resolves Polymarket market IDs and token IDs.
    
    The Gamma API provides market metadata including CLOB token IDs
    for each outcome (UP/DOWN for BTC markets).
    
    Args:
        session: Optional aiohttp session (will create one if not provided)
        
    Thread Safety:
        Methods are async and should be called from event loop.
    """
    
    def __init__(self, session: Optional[aiohttp.ClientSession] = None) -> None:
        self._session = session
        self._owns_session = session is None
        
        logger.info("polymarket_market_resolver_initialized")
    
    async def _ensure_session(self) -> aiohttp.ClientSession:
        """Ensure we have an active session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10),
            )
            self._owns_session = True
        return self._session
    
    async def close(self) -> None:
        """Close session if we own it."""
        if self._owns_session and self._session and not self._session.closed:
            await self._session.close()
    
    @staticmethod
    def current_btc_slot() -> int:
        """
        Get current 15-minute slot timestamp.
        
        Returns:
            Unix timestamp aligned to 15-minute boundary
        """
        ts = now_ms() // 1000
        return (ts // 900) * 900
    
    @staticmethod
    def current_btc_slug() -> str:
        """
        Get current BTC market slug.
        
        Returns:
            Market slug like "btc-updown-15m-1234567890"
        """
        slot = MarketResolver.current_btc_slot()
        return f"btc-updown-15m-{slot}"
    
    @staticmethod
    def time_to_market_end() -> tuple[str, int]:
        """
        Calculate time remaining until current 15m market ends.
        
        Returns:
            (formatted_time "MM:SS", total_seconds)
        """
        ts = now_ms() // 1000
        slot_start = (ts // 900) * 900
        slot_end = slot_start + 900
        seconds_left = slot_end - ts
        
        minutes = seconds_left // 60
        seconds = seconds_left % 60
        return f"{minutes}:{seconds:02d}", seconds_left
    
    @staticmethod
    def next_market_slot() -> int:
        """Get next market slot timestamp."""
        return MarketResolver.current_btc_slot() + 900
    
    async def get_current_btc_tokens(self) -> Optional[dict]:
        """
        Get token IDs for current BTC UP/DOWN market.
        
        Returns:
            Dictionary with market_id, up (token_id), down (token_id)
            or None if API fails
            
        Example response:
            {
                "market_id": "btc-updown-15m-1234567890",
                "slug": "btc-updown-15m-1234567890",
                "up": "0x1234...abcd",
                "down": "0x5678...efgh",
                "slot": 1234567890,
                "ends_at_sec": 1234567890 + 900
            }
        """
        slug = self.current_btc_slug()
        return await self.get_tokens_by_slug(slug)
    
    async def get_tokens_by_slug(self, slug: str) -> Optional[dict]:
        """
        Get token IDs for a specific market slug.
        
        Args:
            slug: Market slug (e.g., "btc-updown-15m-1234567890")
            
        Returns:
            Dictionary with market_id, up, down tokens or None
        """
        session = await self._ensure_session()
        url = f"{PM_GAMMA_API}/events?slug={slug}"
        
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    logger.warning(
                        "polymarket_api_error",
                        extra={"status": resp.status, "slug": slug},
                    )
                    return None
                
                events = await resp.json()
                
                if not events:
                    logger.warning(
                        "polymarket_no_events",
                        extra={"slug": slug},
                    )
                    return None
                
                # Parse market data
                market = events[0].get("markets", [{}])[0]
                clob_token_ids = market.get("clobTokenIds", [])
                outcomes = market.get("outcomes", [])
                
                # Handle string-encoded JSON
                if isinstance(clob_token_ids, str):
                    clob_token_ids = json.loads(clob_token_ids)
                if isinstance(outcomes, str):
                    outcomes = json.loads(outcomes)
                
                # Find UP and DOWN indices
                up_idx = outcomes.index("Up") if "Up" in outcomes else 0
                down_idx = outcomes.index("Down") if "Down" in outcomes else 1
                
                # Extract slot from slug
                slot = 0
                try:
                    slot = int(slug.split("-")[-1])
                except (ValueError, IndexError):
                    pass
                
                result = {
                    "market_id": slug,
                    "slug": slug,
                    "up": clob_token_ids[up_idx] if up_idx < len(clob_token_ids) else None,
                    "down": clob_token_ids[down_idx] if down_idx < len(clob_token_ids) else None,
                    "slot": slot,
                    "ends_at_sec": slot + 900 if slot else None,
                }
                
                logger.info(
                    "polymarket_tokens_resolved",
                    extra={
                        "slug": slug,
                        "up_token": result["up"][:16] + "..." if result["up"] else None,
                        "down_token": result["down"][:16] + "..." if result["down"] else None,
                    },
                )
                
                return result
                
        except aiohttp.ClientError as e:
            logger.error(
                "polymarket_api_client_error",
                extra={"error": str(e), "slug": slug},
            )
            return None
        except json.JSONDecodeError as e:
            logger.error(
                "polymarket_api_json_error",
                extra={"error": str(e), "slug": slug},
            )
            return None
        except Exception as e:
            logger.exception(
                "polymarket_api_unexpected_error",
                extra={"error": str(e), "slug": slug},
            )
            return None
    
    async def get_market_info(self, slug: str) -> Optional[dict]:
        """
        Get full market info from Gamma API.
        
        Returns raw API response for debugging/analysis.
        """
        session = await self._ensure_session()
        url = f"{PM_GAMMA_API}/events?slug={slug}"
        
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    return await resp.json()
                return None
        except Exception as e:
            logger.error(
                "polymarket_market_info_error",
                extra={"error": str(e), "slug": slug},
            )
            return None
