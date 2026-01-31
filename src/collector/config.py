"""
Configuration Module
====================

Application configuration using pydantic-settings.
All settings can be overridden via environment variables.

Environment variables:
    SYMBOL          - Trading pair (default: BTCUSDT)
    FUTURES_WS      - Futures WebSocket URL
    SPOT_WS         - Spot WebSocket URL
    DEPTH_ENABLED   - Enable depth stream (default: true)
    DEPTH_SPEED     - Depth update speed: 100ms, 500ms, none
    TOPN            - Number of order book levels (default: 10)
    LOG_LEVEL       - Logging level (default: INFO)
    HTTP_HOST       - HTTP server host (default: 0.0.0.0)
    HTTP_PORT       - HTTP server port (default: 8000)

Production notes:
    - DEPTH_SPEED=100ms generates high traffic (~10 msg/sec), consider 500ms for lower load
    - SYMBOL must be valid Binance trading pair (e.g., BTCUSDT, ETHUSDT)
    - Ensure NTP is synchronized on the server for accurate lag measurements
"""

import logging
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings.
    
    All fields can be configured via environment variables.
    Example: SYMBOL=ETHUSDT python -m collector
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Trading pair
    SYMBOL: str = Field(
        default="BTCUSDT",
        description="Trading pair symbol (uppercase)",
    )
    
    # WebSocket endpoints
    FUTURES_WS: str = Field(
        default="wss://fstream.binance.com",
        description="Binance Futures WebSocket base URL",
    )
    SPOT_WS: str = Field(
        default="wss://stream.binance.com:9443",
        description="Binance Spot WebSocket base URL",
    )
    
    # Depth stream settings
    DEPTH_ENABLED: bool = Field(
        default=True,
        description="Enable order book depth stream",
    )
    DEPTH_SPEED: Literal["100ms", "500ms", "none"] = Field(
        default="100ms",
        description="Depth stream update speed (100ms or 500ms for futures)",
    )
    TOPN: int = Field(
        default=10,
        ge=1,
        le=20,
        description="Number of order book levels to track",
    )
    
    # Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level",
    )
    
    # HTTP server
    HTTP_HOST: str = Field(
        default="0.0.0.0",
        description="HTTP server bind host",
    )
    HTTP_PORT: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="HTTP server bind port",
    )
    
    # REST API endpoints
    FUTURES_REST_BASE: str = Field(
        default="https://fapi.binance.com",
        description="Binance Futures REST API base URL",
    )
    SPOT_REST_BASE: str = Field(
        default="https://api.binance.com",
        description="Binance Spot REST API base URL",
    )
    
    # Context / HTF Klines Bootstrap
    CONTEXT_ENABLED: bool = Field(
        default=True,
        description="Enable HTF klines bootstrap on startup",
    )
    CONTEXT_TFS: list[str] = Field(
        default=["1m", "5m", "15m", "1h"],
        description="Timeframes to bootstrap (1m, 5m, 15m, 1h, etc.)",
    )
    CONTEXT_BOOTSTRAP_LIMIT: int = Field(
        default=500,
        ge=1,
        le=1500,
        description="Number of klines to fetch per timeframe on bootstrap",
    )
    CONTEXT_MIN_READY_BARS: int = Field(
        default=200,
        ge=1,
        description="Minimum bars required to consider context ready",
    )
    
    # TAAPI Integration
    TAAPI_ENABLED: bool = Field(
        default=True,
        description="Enable TAAPI technical indicators",
    )
    TAAPI_SECRET: str = Field(
        default="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjkzNmVmODc4MDZmZjE2NTFlMWEwYmY5IiwiaWF0IjoxNzY5NTExNzQ4LCJleHAiOjMzMjczOTc1NzQ4fQ.AQdQfO_3XonkFUU6-QioU_O7QHJn6k80kzWNkSesyRw",
        description="TAAPI.IO API secret key",
    )
    TAAPI_EXCHANGE: str = Field(
        default="binancefutures",
        description="TAAPI exchange (binance, binancefutures)",
    )
    TAAPI_TFS: list[str] = Field(
        default=["1m", "5m", "15m", "1h"],
        description="Timeframes to fetch from TAAPI",
    )
    
    # Polymarket Integration
    POLYMARKET_ENABLED: bool = Field(
        default=True,
        description="Enable Polymarket prediction market integration",
    )
    POLYMARKET_WS_URL: str = Field(
        default="wss://ws-subscriptions-clob.polymarket.com/ws/market",
        description="Polymarket CLOB WebSocket URL",
    )
    POLYMARKET_GAMMA_API: str = Field(
        default="https://gamma-api.polymarket.com",
        description="Polymarket Gamma API URL for market resolution",
    )
    POLYMARKET_STALE_THRESHOLD_SEC: float = Field(
        default=5.0,
        description="Threshold (seconds) after which Polymarket data is considered stale",
    )
    POLYMARKET_MIN_DEPTH: float = Field(
        default=200.0,
        description="Minimum depth (top3) required for trade execution",
    )
    POLYMARKET_MAX_SPREAD_BPS: float = Field(
        default=500.0,
        description="Maximum spread (bps) before veto applies",
    )
    
    # Polymarket UP/DOWN Mapping
    POLY_UP_IS_YES: bool = Field(
        default=True,
        description="If True, UP market direction = YES token (typical for 'will price be higher' questions)",
    )
    
    # Fair Value Model
    FAIR_ENABLED: bool = Field(
        default=False,
        description="Enable fair value computation (set False until properly calibrated)",
    )
    
    # Event Recording for Backtesting
    RECORD_ENABLED: bool = Field(
        default=False,
        description="Enable high-frequency event recording for backtesting",
    )
    RECORD_DIR: str = Field(
        default="data/recordings",
        description="Directory to store recording files",
    )
    RECORD_BUFFER_SIZE: int = Field(
        default=100,
        ge=10,
        le=10000,
        description="Number of events to buffer before writing",
    )
    RECORD_FLUSH_INTERVAL_MS: int = Field(
        default=1000,
        ge=100,
        le=10000,
        description="Maximum milliseconds between flushes",
    )
    
    # Price Smoothing (S1)
    PRICE_EMA_SEC: float = Field(
        default=20.0,
        ge=1.0,
        le=120.0,
        description="EMA time constant for price smoothing (seconds)",
    )
    
    # Volatility Estimation (S2)
    VOL_FAST_MINUTES: int = Field(
        default=60,
        ge=10,
        le=120,
        description="Fast volatility window (minutes of 1m bars)",
    )
    VOL_SLOW_MINUTES: int = Field(
        default=360,
        ge=60,
        le=720,
        description="Slow volatility window (minutes, ~6 hours)",
    )
    VOL_MIN_BARS: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Minimum bars for valid sigma calculation (warmup)",
    )
    SIGMA_BLEND_W_RVOL_K: float = Field(
        default=0.5,
        ge=0.1,
        le=2.0,
        description="Sensitivity of blend weight to rvol",
    )
    SIGMA_BLEND_W_CLAMP_MIN: float = Field(
        default=0.2,
        ge=0.0,
        le=0.5,
        description="Minimum blend weight for fast sigma",
    )
    SIGMA_BLEND_W_CLAMP_MAX: float = Field(
        default=0.9,
        ge=0.5,
        le=1.0,
        description="Maximum blend weight for fast sigma",
    )
    
    # Bias Model (S4)
    BIAS_SLOPE_LOOKBACK: int = Field(
        default=20,
        ge=5,
        le=50,
        description="Bars for slope calculation",
    )
    BIAS_EMA_FAST: int = Field(
        default=12,
        ge=5,
        le=30,
        description="Fast EMA period for bias",
    )
    BIAS_EMA_SLOW: int = Field(
        default=26,
        ge=15,
        le=60,
        description="Slow EMA period for bias",
    )
    BIAS_UP_THRESHOLD: float = Field(
        default=0.55,
        ge=0.5,
        le=0.7,
        description="Threshold for UP bias direction",
    )
    BIAS_DOWN_THRESHOLD: float = Field(
        default=0.45,
        ge=0.3,
        le=0.5,
        description="Threshold for DOWN bias direction",
    )
    
    # Confidence Model (S5)
    CONF_WEIGHT_NET_EDGE: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Weight for net edge in confidence",
    )
    CONF_WEIGHT_AGREEMENT: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
        description="Weight for fair/bias agreement",
    )
    CONF_WEIGHT_EVENTS: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Weight for events (spike/dip)",
    )
    CONF_WEIGHT_QUALITY: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Weight for data quality",
    )
    CONF_LOW_THRESHOLD: float = Field(
        default=0.3,
        ge=0.0,
        le=0.5,
        description="Threshold for LOW confidence level",
    )
    CONF_HIGH_THRESHOLD: float = Field(
        default=0.7,
        ge=0.5,
        le=1.0,
        description="Threshold for HIGH confidence level",
    )
    
    # ========================================================================
    # Slice Execution Settings (BLOCK 0)
    # ========================================================================
    # These settings control the paper/real trading execution logic.
    # A "slice" is a small fixed-size order (e.g., $20) that we place.
    
    # Minimum slice size in USD - below this, not worth the execution overhead
    MIN_SLICE_USD: float = Field(
        default=20.0,
        ge=1.0,
        description="Minimum slice size in USD for execution",
    )
    
    # Minimum depth at top-of-book (best bid/ask) to consider executable
    MIN_TOP1_USD: float = Field(
        default=20.0,
        ge=1.0,
        description="Minimum depth at top1 (best price) in USD to execute one slice",
    )
    
    # Minimum depth at top 3 levels to consider for multiple slices
    MIN_TOP3_USD: float = Field(
        default=60.0,
        ge=1.0,
        description="Minimum depth at top3 levels in USD for comfortable execution",
    )
    
    # ========================================================================
    # Spread Cost Model (BLOCK 0)
    # ========================================================================
    # Edge must exceed spread cost + buffer to be profitable.
    # required_edge = (spread_bps / 2) + EDGE_BUFFER_BPS
    
    EDGE_BUFFER_BPS: float = Field(
        default=25.0,
        ge=0.0,
        description="Additional buffer over half-spread required for edge (in bps)",
    )
    
    # Spread thresholds for quality degradation
    MAX_SPREAD_BPS_DEGRADED: float = Field(
        default=400.0,
        ge=0.0,
        description="Spread above this = DEGRADED mode (higher cost, proceed with caution)",
    )
    
    MAX_SPREAD_BPS_BAD: float = Field(
        default=800.0,
        ge=0.0,
        description="Spread above this = BAD mode (too expensive, hard veto)",
    )
    
    # ========================================================================
    # Decision Cadence (BLOCK 0)
    # ========================================================================
    
    SLOW_LOOP_SEC: float = Field(
        default=1.0,
        ge=0.1,
        description="Interval between edge decisions in slow loop (seconds)",
    )
    
    COOLDOWN_SEC: float = Field(
        default=2.0,
        ge=0.0,
        description="Cooldown after executing a slice before next execution (seconds)",
    )
    
    # ========================================================================
    # Budgeting Per Window (BLOCK 0)
    # ========================================================================
    # Paper trading budget constraints per 15-minute window.
    # Prevents over-trading even with many signals.
    
    MAX_SLICES_PER_WINDOW: int = Field(
        default=30,
        ge=1,
        description="Maximum number of slices per 15-min window",
    )
    
    MAX_USD_PER_WINDOW: float = Field(
        default=300.0,
        ge=1.0,
        description="Maximum total USD exposure per 15-min window",
    )
    
    SLICE_USD: float = Field(
        default=20.0,
        ge=1.0,
        description="Standard slice size in USD for paper/real trading",
    )
    
    @field_validator("SYMBOL")
    @classmethod
    def symbol_uppercase(cls, v: str) -> str:
        """Ensure symbol is uppercase and valid format."""
        v = v.upper()
        # Basic validation: must end with USDT, BUSD, BTC, ETH, etc.
        valid_quotes = ("USDT", "BUSD", "BTC", "ETH", "BNB", "USDC")
        if not any(v.endswith(q) for q in valid_quotes):
            # Log warning but don't fail - Binance may have new quote assets
            logger = logging.getLogger(__name__)
            logger.warning(
                f"config_symbol_unusual: {v} doesn't end with common quote asset"
            )
        return v
    
    @model_validator(mode="after")
    def validate_and_warn(self) -> "Settings":
        """Validate configuration and log warnings for high-load settings."""
        logger = logging.getLogger(__name__)
        
        # Warn about high-load depth settings
        if self.DEPTH_ENABLED and self.DEPTH_SPEED == "100ms":
            logger.warning(
                "config_depth_high_load: DEPTH_SPEED=100ms generates ~10 msg/sec. "
                "Consider DEPTH_SPEED=500ms for lower CPU/network load."
            )
        
        return self
    
    def dump(self) -> dict:
        """
        Dump current configuration as dictionary.
        Useful for logging configuration at startup.
        
        Returns:
            Dictionary with all configuration values.
        """
        return {
            "symbol": self.SYMBOL,
            "futures_ws": self.FUTURES_WS,
            "spot_ws": self.SPOT_WS,
            "depth_enabled": self.DEPTH_ENABLED,
            "depth_speed": self.DEPTH_SPEED,
            "topn": self.TOPN,
            "log_level": self.LOG_LEVEL,
            "http_host": self.HTTP_HOST,
            "http_port": self.HTTP_PORT,
            "futures_rest_base": self.FUTURES_REST_BASE,
            "spot_rest_base": self.SPOT_REST_BASE,
            "context_enabled": self.CONTEXT_ENABLED,
            "context_tfs": self.CONTEXT_TFS,
            "context_bootstrap_limit": self.CONTEXT_BOOTSTRAP_LIMIT,
            "context_min_ready_bars": self.CONTEXT_MIN_READY_BARS,
            "taapi_enabled": self.TAAPI_ENABLED,
            "taapi_exchange": self.TAAPI_EXCHANGE,
            "taapi_tfs": self.TAAPI_TFS,
            "polymarket_enabled": self.POLYMARKET_ENABLED,
            "polymarket_stale_threshold_sec": self.POLYMARKET_STALE_THRESHOLD_SEC,
            "polymarket_min_depth": self.POLYMARKET_MIN_DEPTH,
            "polymarket_max_spread_bps": self.POLYMARKET_MAX_SPREAD_BPS,
            # Event recording
            "record_enabled": self.RECORD_ENABLED,
            "record_dir": self.RECORD_DIR,
            # BLOCK 0: Slice execution
            "slice_usd": self.SLICE_USD,
            "min_slice_usd": self.MIN_SLICE_USD,
            "min_top1_usd": self.MIN_TOP1_USD,
            "min_top3_usd": self.MIN_TOP3_USD,
            # BLOCK 0: Spread cost model
            "edge_buffer_bps": self.EDGE_BUFFER_BPS,
            "max_spread_bps_degraded": self.MAX_SPREAD_BPS_DEGRADED,
            "max_spread_bps_bad": self.MAX_SPREAD_BPS_BAD,
            # BLOCK 0: Decision cadence
            "slow_loop_sec": self.SLOW_LOOP_SEC,
            "cooldown_sec": self.COOLDOWN_SEC,
            # BLOCK 0: Budgeting
            "max_slices_per_window": self.MAX_SLICES_PER_WINDOW,
            "max_usd_per_window": self.MAX_USD_PER_WINDOW,
        }
    
    def futures_streams(self) -> list[str]:
        """
        Build list of Futures WebSocket streams to subscribe.
        
        Always includes:
            - {symbol}@aggTrade
            - {symbol}@bookTicker
            - {symbol}@markPrice
            - {symbol}@forceOrder
        
        Optionally includes (if DEPTH_ENABLED and DEPTH_SPEED != "none"):
            - {symbol}@depth@{DEPTH_SPEED}
        
        Returns:
            List of stream names (lowercase).
        """
        symbol_lower = self.SYMBOL.lower()
        
        streams = [
            f"{symbol_lower}@aggTrade",
            f"{symbol_lower}@bookTicker",
            f"{symbol_lower}@markPrice",
            f"{symbol_lower}@forceOrder",
        ]
        
        # Add depth stream if enabled
        if self.DEPTH_ENABLED and self.DEPTH_SPEED != "none":
            streams.append(f"{symbol_lower}@depth@{self.DEPTH_SPEED}")
        
        return streams
    
    def spot_streams(self) -> list[str]:
        """
        Build list of Spot WebSocket streams to subscribe.
        
        Includes:
            - {symbol}@aggTrade
            - {symbol}@bookTicker
        
        Returns:
            List of stream names (lowercase).
        """
        symbol_lower = self.SYMBOL.lower()
        
        return [
            f"{symbol_lower}@aggTrade",
            f"{symbol_lower}@bookTicker",
        ]


# Global settings instance
settings = Settings()
