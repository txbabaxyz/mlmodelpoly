"""
TAAPI Integration Module
========================

Provides technical indicators from TAAPI.IO service.

Components:
    - TaapiAsyncClient: Async wrapper for TAAPI API
    - TaapiContextStore: Storage for indicator values
    - TaapiBootstrapper: Bootstrap indicators at startup
    - TaapiScheduler: Periodic indicator updates
    - TaapiContextEngine: Interpret indicators into context
    - INDICATOR_SETS: Predefined indicator sets per timeframe
"""

from collector.taapi.client import TaapiAsyncClient
from collector.taapi.store import TaapiContextStore
from collector.taapi.bootstrap import TaapiBootstrapper
from collector.taapi.scheduler import TaapiScheduler
from collector.taapi.context_engine import TaapiContextEngine
from collector.taapi.indicator_sets import INDICATOR_SETS

__all__ = [
    "TaapiAsyncClient",
    "TaapiContextStore",
    "TaapiBootstrapper",
    "TaapiScheduler",
    "TaapiContextEngine",
    "INDICATOR_SETS",
]
