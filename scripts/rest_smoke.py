#!/usr/bin/env python3
"""
REST Client Smoke Test
======================

Tests REST client against Binance /time endpoints.

Usage:
    python scripts/rest_smoke.py
    
    # Or from project root:
    python -m scripts.rest_smoke
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from collector.logging_setup import setup_logging
from collector.rest_client import RestClient


async def main() -> None:
    """Run smoke tests for REST client."""
    setup_logging("INFO")
    
    print("=" * 60)
    print("REST Client Smoke Test")
    print("=" * 60)
    print()
    
    # Test Futures API
    print("--- Test 1: Futures /fapi/v1/time ---")
    async with RestClient(
        base_url="https://fapi.binance.com",
        name="futures",
        max_concurrency=4,
        min_interval_ms=150,
    ) as futures_client:
        data, headers = await futures_client.get_json("/fapi/v1/time")
        
        if data:
            print(f"✓ Futures time: {data}")
            print(f"  Server time: {data.get('serverTime')}")
        else:
            print("✗ Futures request failed")
    
    print()
    
    # Test Spot API
    print("--- Test 2: Spot /api/v3/time ---")
    async with RestClient(
        base_url="https://api.binance.com",
        name="spot",
        max_concurrency=4,
        min_interval_ms=150,
    ) as spot_client:
        data, headers = await spot_client.get_json("/api/v3/time")
        
        if data:
            print(f"✓ Spot time: {data}")
            print(f"  Server time: {data.get('serverTime')}")
        else:
            print("✗ Spot request failed")
    
    print()
    
    # Test with params
    print("--- Test 3: Futures exchangeInfo (with symbol param) ---")
    async with RestClient(
        base_url="https://fapi.binance.com",
        name="futures",
        max_concurrency=4,
        min_interval_ms=150,
    ) as futures_client:
        data, headers = await futures_client.get_json(
            "/fapi/v1/exchangeInfo",
            params={"symbol": "BTCUSDT"},
        )
        
        if data:
            symbols = data.get("symbols", [])
            if symbols:
                symbol_info = symbols[0]
                print(f"✓ Exchange info for BTCUSDT:")
                print(f"  Status: {symbol_info.get('status')}")
                print(f"  Base asset: {symbol_info.get('baseAsset')}")
                print(f"  Quote asset: {symbol_info.get('quoteAsset')}")
            else:
                print("✓ Exchange info received (no symbols)")
        else:
            print("✗ Exchange info request failed")
    
    print()
    
    # Test multiple concurrent requests
    print("--- Test 4: Concurrent requests ---")
    async with RestClient(
        base_url="https://fapi.binance.com",
        name="futures",
        max_concurrency=4,
        min_interval_ms=100,
    ) as client:
        # Make 5 concurrent requests
        tasks = [
            client.get_json("/fapi/v1/time")
            for _ in range(5)
        ]
        
        results = await asyncio.gather(*tasks)
        
        success_count = sum(1 for data, _ in results if data is not None)
        print(f"✓ Concurrent requests: {success_count}/5 succeeded")
    
    print()
    print("=" * 60)
    print("Smoke tests completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
