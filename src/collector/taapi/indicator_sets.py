"""
TAAPI Indicator Sets
====================

Predefined indicator sets for each timeframe.

Each set is designed to fit within TAAPI bulk limits (max 20 indicators per request).
We keep each set <= 12 indicators for safety margin.

Usage:
    from collector.taapi.indicator_sets import INDICATOR_SETS
    
    indicators_1m = INDICATOR_SETS["1m"]
"""

# =============================================================================
# 1m - Inside window micro signals
# =============================================================================

INDICATORS_1M = [
    {"indicator": "ema", "period": 20, "id": "ema_20"},
    {"indicator": "ema", "period": 50, "id": "ema_50"},
    {"indicator": "rsi", "period": 14, "id": "rsi_14"},
    {"indicator": "macd", "optInFastPeriod": 12, "optInSlowPeriod": 26, "optInSignalPeriod": 9, "id": "macd"},
    {"indicator": "roc", "period": 9, "id": "roc_9"},
]

# =============================================================================
# 5m - Local structure
# =============================================================================

INDICATORS_5M = [
    {"indicator": "ema", "period": 20, "id": "ema_20"},
    {"indicator": "ema", "period": 50, "id": "ema_50"},
    {"indicator": "adx", "period": 14, "id": "adx_14"},
    {"indicator": "atr", "period": 14, "id": "atr_14"},
    {"indicator": "supertrend", "period": 10, "multiplier": 3, "id": "supertrend"},
]

# =============================================================================
# 15m - Main window timeframe
# =============================================================================

INDICATORS_15M = [
    {"indicator": "ema", "period": 20, "id": "ema_20"},
    {"indicator": "ema", "period": 50, "id": "ema_50"},
    {"indicator": "adx", "period": 14, "id": "adx_14"},
    {"indicator": "atr", "period": 14, "id": "atr_14"},
    {"indicator": "rsi", "period": 14, "id": "rsi_14"},
    {"indicator": "macd", "optInFastPeriod": 12, "optInSlowPeriod": 26, "optInSignalPeriod": 9, "id": "macd"},
    {"indicator": "cmf", "period": 20, "id": "cmf_20"},
]

# =============================================================================
# 1h - Background context
# =============================================================================

INDICATORS_1H = [
    {"indicator": "ema", "period": 50, "id": "ema_50"},
    {"indicator": "ema", "period": 200, "id": "ema_200"},
    {"indicator": "adx", "period": 14, "id": "adx_14"},
    {"indicator": "atr", "period": 14, "id": "atr_14"},
    {"indicator": "macd", "optInFastPeriod": 12, "optInSlowPeriod": 26, "optInSignalPeriod": 9, "id": "macd"},
]

# =============================================================================
# Combined indicator sets
# =============================================================================

INDICATOR_SETS: dict[str, list[dict]] = {
    "1m": INDICATORS_1M,
    "5m": INDICATORS_5M,
    "15m": INDICATORS_15M,
    "1h": INDICATORS_1H,
}

# Timeframes to bootstrap/update
TAAPI_TIMEFRAMES = ["1m", "5m", "15m", "1h"]

# Default symbol format for TAAPI
def format_symbol_for_taapi(symbol: str) -> str:
    """
    Format symbol for TAAPI (requires slash).
    
    Args:
        symbol: Symbol like "BTCUSDT"
        
    Returns:
        Formatted symbol like "BTC/USDT"
    """
    # Handle common quote assets
    for quote in ["USDT", "BUSD", "USDC", "BTC", "ETH"]:
        if symbol.endswith(quote):
            base = symbol[:-len(quote)]
            return f"{base}/{quote}"
    
    # Default: assume last 4 chars are quote
    return f"{symbol[:-4]}/{symbol[-4:]}"


def get_indicator_count() -> dict[str, int]:
    """Get count of indicators per timeframe."""
    return {tf: len(indicators) for tf, indicators in INDICATOR_SETS.items()}
