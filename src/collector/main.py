"""
Main Entry Point
================

Entry point for the Binance Collector service.

Usage:
    python -m collector

The service will:
1. Load configuration from environment
2. Setup JSON logging
3. Start HTTP API server (/health, /state, /latest/features, /control/anchor/*)
4. Connect to Binance Futures WebSocket (aggTrade, bookTicker, markPrice, forceOrder, depth)
5. Connect to Binance Spot WebSocket (aggTrade, bookTicker)
6. Process events through normalization pipeline
7. Aggregate aggTrade into OHLCV+delta bars (5s, 15s, 1m for both markets)
8. Compute trading features (CVD, RVOL, Impulse, Basis, Microprice, Anchored VWAP)
9. Collect quality metrics (lag p50/p95, RPS, reconnects, bars)
10. Log throughput, metrics, features, and anchor state periodically
11. Log sanity check every 60 seconds
"""

import asyncio
import logging
import signal
import sys
from collections import defaultdict
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from collector.polymarket import PolymarketBookStore
    from collector.event_recorder import EventRecorder

import uvicorn

from collector import __schema_version__, __version__
from collector.bars import BarsManager, LastBarsStore
from collector.config import settings
from collector.context_bootstrapper import ContextBootstrapper
from collector.context_klines_store import KlinesStore
from collector.context_updater import ContextUpdater
from collector.decision_logger import DecisionLogger
from collector.edge_engine import EdgeEngine
from collector.features import FeatureEngine
from collector.paper_trading import PaperEngine
from collector.storage_writer import StorageWriter
from collector.market_context import MarketContextEngine
from collector.taapi import (
    TaapiAsyncClient,
    TaapiContextStore,
    TaapiBootstrapper,
    TaapiScheduler,
    TaapiContextEngine,
)
from collector.polymarket import (
    PolymarketBookStore,
    PolymarketWSClient,
    MarketResolver,
    SpikeProbe,
)
from collector.http_api import create_app
from collector.logging_setup import setup_logging
from collector.metrics import Metrics
from collector.pipeline import EventPipeline, SamplingLogger
from collector.rest_client import RestClient
from collector.types import NormalizedEvent
from collector.utils_time import now_ms
from collector.ws_client import WSClient

# Logger for this module
logger = logging.getLogger(__name__)

# Constants
QUEUE_MAX_SIZE = 20_000  # Shared queue for both futures and spot
BARS_QUEUE_MAX_SIZE = 5_000  # Queue for closed bars
THROUGHPUT_INTERVAL_MS = 5_000  # Log throughput every 5 seconds
METRICS_LOG_INTERVAL_MS = 5_000  # Log metrics snapshot every 5 seconds
FEATURES_LOG_INTERVAL_MS = 5_000  # Log features snapshot every 5 seconds
SANITY_LOG_INTERVAL_SEC = 60  # Log sanity check every 60 seconds


async def consumer_loop(
    queue: asyncio.Queue,
    pipeline: EventPipeline,
    sampler: SamplingLogger,
    metrics: Metrics,
    bars_manager: BarsManager,
    bars_out_queue: asyncio.Queue,
    feature_engine: FeatureEngine,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Consumer loop - processes messages from the shared queue.
    
    Args:
        queue: Shared queue with raw WebSocket messages from all sources
        pipeline: Event processing pipeline
        sampler: Sampling logger for event samples
        metrics: Metrics collector
        bars_manager: Manager for all bar aggregators
        bars_out_queue: Queue for emitting closed bars
        feature_engine: Feature computation engine
        shutdown_event: Event to signal shutdown
    """
    logger.info("consumer_started")
    
    # Per-market, per-type counters for throughput
    event_counts: dict[str, dict[str, int]] = {
        "futures": defaultdict(int),
        "spot": defaultdict(int),
    }
    last_throughput_time = now_ms()
    
    while not shutdown_event.is_set():
        try:
            # Get message with timeout to check shutdown periodically
            try:
                envelope = await asyncio.wait_for(
                    queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            
            # Process through pipeline
            normalized: Optional[NormalizedEvent] = pipeline.handle_ws_message(envelope)
            
            if normalized:
                market = normalized.market
                event_type = normalized.type
                lag_ms = normalized.data.get("lag_ms", 0)
                
                # Update counter
                if market in event_counts:
                    event_counts[market][event_type] += 1
                
                # Record metrics
                metrics.observe_event(market, event_type, lag_ms)
                
                # Feed feature engine with tick data
                if event_type == "bookTicker":
                    feature_engine.on_bookticker(normalized)
                elif event_type == "markPrice":
                    feature_engine.on_markprice(normalized)
                elif event_type == "depth":
                    feature_engine.on_depth(normalized)
                elif event_type == "liquidation":
                    feature_engine.on_liquidation(normalized)
                elif event_type == "aggTrade":
                    # Update aggtrade timestamp for gap detection
                    feature_engine.update_aggtrade_timestamp(market, normalized.ts_recv_ms)
                
                # Process through bars manager
                closed_bars = bars_manager.handle_event(normalized)
                for bar in closed_bars:
                    metrics.inc_bar(bar["market"], bar["tf"])
                    # Put bar in output queue (non-blocking)
                    try:
                        bars_out_queue.put_nowait(bar)
                    except asyncio.QueueFull:
                        logger.warning(
                            "bars_queue_full",
                            extra={"queue_size": bars_out_queue.qsize()},
                        )
                
                # Sample logging based on market and event type
                if sampler.should_log(market, event_type):
                    _log_sample_event(normalized)
            
            # Throughput logging
            current_time = now_ms()
            elapsed = current_time - last_throughput_time
            
            if elapsed >= THROUGHPUT_INTERVAL_MS:
                _log_throughput(event_counts, elapsed, queue.qsize(), pipeline, bars_manager)
                
                # Reset counters
                for market in event_counts:
                    event_counts[market].clear()
                last_throughput_time = current_time
            
            # Mark task done
            queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("consumer_cancelled")
            break
        except Exception as e:
            logger.exception(
                "consumer_error",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
    
    # Flush all bars on shutdown
    flushed_bars = bars_manager.flush_all()
    if flushed_bars:
        logger.info(
            "bars_flushed",
            extra={"count": len(flushed_bars)},
        )
    
    logger.info("consumer_stopped")


async def bars_consumer_loop(
    bars_queue: asyncio.Queue,
    last_bars_store: LastBarsStore,
    feature_engine: FeatureEngine,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Bars consumer loop - processes closed bars from the queue.
    
    Logs bars with different frequencies:
        - futures 5s/15s/1m: log every bar
        - spot 5s: log 1/5 (reduce spam)
        - spot 15s/1m: log every bar
    
    Args:
        bars_queue: Queue with closed bars
        last_bars_store: Store for last bars (used by HTTP API)
        feature_engine: Feature computation engine
        shutdown_event: Event to signal shutdown
    """
    logger.info("bars_consumer_started")
    
    # Sampling counters for spot 5s bars
    spot_5s_counter = 0
    
    while not shutdown_event.is_set():
        try:
            # Get bar with timeout to check shutdown periodically
            try:
                bar = await asyncio.wait_for(
                    bars_queue.get(),
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue
            
            market = bar["market"]
            tf = bar["tf"]
            
            # Update last bars store
            last_bars_store.update(bar)
            
            # Feed feature engine with bar data
            feature_engine.on_bar(bar)
            
            # Determine if we should log this bar
            should_log = True
            
            if market == "spot" and tf == "5s":
                # Log only 1/5 spot 5s bars
                spot_5s_counter += 1
                should_log = (spot_5s_counter % 5 == 0)
            
            if should_log:
                _log_bar_closed(bar)
            
            bars_queue.task_done()
            
        except asyncio.CancelledError:
            logger.info("bars_consumer_cancelled")
            break
        except Exception as e:
            logger.exception(
                "bars_consumer_error",
                extra={"error": str(e)},
            )
    
    logger.info("bars_consumer_stopped")


async def metrics_logger_loop(
    metrics: Metrics,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Periodically log metrics snapshot.
    """
    logger.info("metrics_logger_started")
    
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(METRICS_LOG_INTERVAL_MS / 1000.0)
            
            if shutdown_event.is_set():
                break
            
            summary = metrics.get_short_summary()
            
            logger.info(
                "metrics_snapshot",
                extra=summary,
            )
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(
                "metrics_logger_error",
                extra={"error": str(e)},
            )
    
    logger.info("metrics_logger_stopped")


async def features_logger_loop(
    feature_engine: FeatureEngine,
    metrics: Metrics,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Periodically log features, anchor state, depth, and final features.
    """
    logger.info("features_logger_started")
    
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(FEATURES_LOG_INTERVAL_MS / 1000.0)
            
            if shutdown_event.is_set():
                break
            
            # Get metrics snapshot for quality computation
            metrics_snap = metrics.snapshot()
            
            # Log features summary
            summary = feature_engine.get_short_summary()
            logger.info(
                "features_snapshot",
                extra=summary,
            )
            
            # Log anchor state separately
            anchor_summary = feature_engine.get_anchor_summary()
            logger.info(
                "anchor_state",
                extra=anchor_summary,
            )
            
            # Log depth state
            depth_summary = feature_engine.get_depth_summary()
            logger.info(
                "depth_state",
                extra=depth_summary,
            )
            
            # Log micro 1m features
            micro_1m_summary = feature_engine.get_micro_1m_summary()
            logger.info(
                "micro_1m",
                extra=micro_1m_summary,
            )
            
            # Log final features (liquidations, absorption, quality mode, gaps)
            final_summary = feature_engine.get_final_summary(metrics_snap)
            logger.info(
                "final_features",
                extra=final_summary,
            )
            
            # Log polymarket spike stats if available
            pm_snap = feature_engine.snapshot(metrics_snap).get("polymarket", {})
            spike_stats = pm_snap.get("spike_stats") if pm_snap else None
            if spike_stats:
                yes_stats = spike_stats.get("yes", {})
                no_stats = spike_stats.get("no", {})
                logger.info(
                    "spike_stats",
                    extra={
                        "yes_count": yes_stats.get("count", 0),
                        "yes_median_ms": yes_stats.get("median_duration_ms"),
                        "yes_max_ticks": yes_stats.get("max_improve_ticks"),
                        "no_count": no_stats.get("count", 0),
                        "no_median_ms": no_stats.get("median_duration_ms"),
                        "no_max_ticks": no_stats.get("max_improve_ticks"),
                    },
                )
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(
                "features_logger_error",
                extra={"error": str(e)},
            )
    
    logger.info("features_logger_stopped")


async def sanity_logger_loop(
    feature_engine: FeatureEngine,
    metrics: Metrics,
    shutdown_event: asyncio.Event,
) -> None:
    """
    Periodically log sanity check every 60 seconds.
    
    This provides a high-level health summary for monitoring:
    - uptime_sec
    - ws_futures / ws_spot connectivity
    - bars_total
    - features_ok
    - quality_mode
    """
    logger.info("sanity_logger_started")
    
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(SANITY_LOG_INTERVAL_SEC)
            
            if shutdown_event.is_set():
                break
            
            # Get metrics and features snapshot
            metrics_snap = metrics.snapshot()
            features_snap = feature_engine.snapshot(metrics_snap)
            
            # Determine features_ok: all core features present
            features_ok = all([
                features_snap.get("futures", {}).get("cvd") is not None,
                features_snap.get("spot", {}).get("cvd") is not None,
                features_snap.get("basis", {}).get("basis_bps") is not None,
                features_snap.get("anchor", {}).get("avwap_15m") is not None,
            ])
            
            # Get quality mode
            quality = features_snap.get("quality", {})
            quality_mode = quality.get("mode", "UNKNOWN")
            ws_futures = quality.get("ws_futures", False)
            ws_spot = quality.get("ws_spot", False)
            
            logger.info(
                "sanity",
                extra={
                    "uptime_sec": round(metrics_snap.get("uptime_ms", 0) / 1000, 1),
                    "ws_futures": ws_futures,
                    "ws_spot": ws_spot,
                    "bars_total": metrics_snap.get("bars_total", {}),
                    "features_ok": features_ok,
                    "quality_mode": quality_mode,
                },
            )
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(
                "sanity_logger_error",
                extra={"error": str(e)},
            )
    
    logger.info("sanity_logger_stopped")


# Market context update interval (seconds)
MARKET_CONTEXT_UPDATE_INTERVAL_SEC = 5

# Edge decision interval (seconds) - 1 second for decision logging
EDGE_DECISION_INTERVAL_SEC = 1


async def edge_decision_loop(
    edge_engine: EdgeEngine,
    feature_engine: FeatureEngine,
    metrics: Metrics,
    decision_logger: DecisionLogger,
    paper_engine: PaperEngine,
    storage_writer: StorageWriter,
    shutdown_event: asyncio.Event,
    polymarket_store: Optional["PolymarketBookStore"] = None,
    accumulate_engine: Optional["AccumulateEngine"] = None,
    event_recorder: Optional["EventRecorder"] = None,
) -> None:
    """
    Unified slow loop: Edge → Accumulate → Paper → Logger → Storage.
    
    Runs every 1 second:
    1. Get features snapshot
    2. Make edge decision
    3. Make accumulate decision (BLOCK 7)
    4. Apply to paper trading
    5. Log to DecisionLogger
    6. Write to storage
    
    Args:
        edge_engine: EdgeEngine instance
        feature_engine: FeatureEngine instance
        metrics: Metrics instance for quality computation
        decision_logger: DecisionLogger for structured recording
        paper_engine: PaperEngine for virtual execution
        storage_writer: StorageWriter for persistence
        shutdown_event: Event to signal shutdown
        polymarket_store: Optional PolymarketBookStore for snapshot
        accumulate_engine: Optional AccumulateEngine for BLOCK 7
        event_recorder: Optional EventRecorder for high-frequency logging
    """
    from collector.utils_time import now_ms
    
    logger.info(
        "edge_decision_loop_starting",
        extra={"interval_sec": EDGE_DECISION_INTERVAL_SEC},
    )
    
    # Track market window for event recorder
    last_market_id: Optional[str] = None
    
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(EDGE_DECISION_INTERVAL_SEC)
            
            if shutdown_event.is_set():
                break
            
            ts = now_ms()
            
            # Update market ID for event recorder on window change
            if event_recorder:
                current_market_id = f"market_{ts // 900_000}"  # 15-min window ID
                if current_market_id != last_market_id:
                    event_recorder.set_current_market(current_market_id)
                    logger.info("event_recorder_new_window", extra={"market_id": current_market_id})
                    last_market_id = current_market_id
            
            # 1. Get features snapshot with metrics
            metrics_snap = metrics.snapshot()
            features = feature_engine.snapshot(metrics_snap)
            
            # 2. Make edge decision
            decision = edge_engine.decide(features)
            
            # 3. Get polymarket snapshot
            pm_snap = polymarket_store.snapshot() if polymarket_store else None
            
            # 3.5 Make accumulate decision (BLOCK 7)
            accum_decision = None
            if accumulate_engine:
                accum_decision = accumulate_engine.decide(features)
            
            # 4. Apply to paper trading (uses features["polymarket"])
            paper_engine.apply(decision, pm_snap)
            paper_snap = paper_engine.snapshot()
            
            # 5. Log to DecisionLogger (structured record with BLOCK 8 data)
            record = decision_logger.log(
                ts_ms=ts,
                features=features,
                edge=decision,
                polymarket=pm_snap,
                paper=paper_snap,
                accumulate=accum_decision,  # BLOCK 8: Add accumulate decision
            )
            
            # 6. Write to storage
            storage_writer.write(record)
            
            # Log summary for quick monitoring (less verbose)
            accum_action = accum_decision.action if accum_decision else "N/A"
            logger.info(
                "edge_decision",
                extra={
                    "direction": decision.direction,
                    "edge_score": round(decision.edge_score, 1),
                    "confidence": round(decision.confidence, 3),
                    "veto": decision.veto,
                    "veto_reasons": decision.veto_reasons[:2] if decision.veto_reasons else None,
                    "accumulate": accum_action,
                    "paper_pnl": paper_snap.get("total_pnl"),
                    "paper_trades": paper_snap.get("total_trades"),
                },
            )
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(
                "edge_decision_error",
                extra={"error": str(e)},
            )
    
    # Flush storage on shutdown
    storage_writer.flush()
    logger.info("edge_decision_loop_stopped")


async def market_context_update_loop(
    market_context_engine: "MarketContextEngine",
    shutdown_event: asyncio.Event,
) -> None:
    """
    Periodically update market context engine.
    
    Updates trends, regime, volatility every few seconds.
    
    Args:
        market_context_engine: MarketContextEngine instance
        shutdown_event: Event to signal shutdown
    """
    logger.info(
        "market_context_update_loop_starting",
        extra={"interval_sec": MARKET_CONTEXT_UPDATE_INTERVAL_SEC},
    )
    
    while not shutdown_event.is_set():
        try:
            await asyncio.sleep(MARKET_CONTEXT_UPDATE_INTERVAL_SEC)
            
            if shutdown_event.is_set():
                break
            
            # Update market context
            market_context_engine.update()
            
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.exception(
                "market_context_update_error",
                extra={"error": str(e)},
            )
    
    logger.info("market_context_update_loop_stopped")


def _log_sample_event(event: NormalizedEvent) -> None:
    """Log a sample event based on its type and market."""
    data = event.data
    event_type = event.type
    market = event.market
    
    if event_type == "aggTrade":
        logger.info(
            "sample_aggTrade",
            extra={
                "market": market,
                "symbol": event.symbol,
                "price": data.get("price"),
                "qty": data.get("qty"),
                "side": data.get("side_aggressor"),
                "lag_ms": data.get("lag_ms"),
            },
        )
    
    elif event_type == "bookTicker":
        logger.info(
            "sample_bookTicker",
            extra={
                "market": market,
                "symbol": event.symbol,
                "bid": data.get("bid_px"),
                "ask": data.get("ask_px"),
                "mid": round(data.get("mid_px", 0), 2),
                "spread_bps": data.get("spread_bps"),
                "lag_ms": data.get("lag_ms"),
            },
        )
    
    elif event_type == "markPrice":
        logger.info(
            "sample_markPrice",
            extra={
                "market": market,
                "symbol": event.symbol,
                "mark_px": data.get("mark_px"),
                "index_px": data.get("index_px"),
                "funding_rate": data.get("funding_rate"),
                "lag_ms": data.get("lag_ms"),
            },
        )
    
    elif event_type == "liquidation":
        logger.info(
            "sample_liquidation",
            extra={
                "market": market,
                "symbol": event.symbol,
                "side": data.get("side"),
                "price": data.get("price"),
                "qty": data.get("qty"),
                "value_usd": round(data.get("price", 0) * data.get("qty", 0), 2),
                "lag_ms": data.get("lag_ms"),
            },
        )
    
    elif event_type == "depth":
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        logger.info(
            "sample_depth",
            extra={
                "market": market,
                "symbol": event.symbol,
                "bid_levels": len(bids),
                "ask_levels": len(asks),
                "best_bid": bids[0]["px"] if bids else None,
                "best_ask": asks[0]["px"] if asks else None,
                "lag_ms": data.get("lag_ms"),
            },
        )


def _log_bar_closed(bar: dict) -> None:
    """Log a closed bar."""
    logger.info(
        "bar_closed",
        extra={
            "market": bar["market"],
            "symbol": bar["symbol"],
            "tf": bar["tf"],
            "t_open_ms": bar["t_open_ms"],
            "open": bar["open"],
            "high": bar["high"],
            "low": bar["low"],
            "close": bar["close"],
            "volume": bar["volume_total"],
            "trades": bar["trades_count"],
            "delta": bar["delta_vol"],
        },
    )


def _log_throughput(
    event_counts: dict[str, dict[str, int]],
    elapsed_ms: int,
    queue_size: int,
    pipeline: EventPipeline,
    bars_manager: Optional[BarsManager] = None,
) -> None:
    """Log throughput statistics by market and type."""
    
    total_events = sum(
        sum(types.values())
        for types in event_counts.values()
    )
    events_per_sec = total_events / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
    
    market_stats = {}
    for market, types in event_counts.items():
        type_stats = {}
        market_total = 0
        for event_type, count in types.items():
            eps = count / (elapsed_ms / 1000.0) if elapsed_ms > 0 else 0
            type_stats[event_type] = {
                "count": count,
                "eps": round(eps, 2),
            }
            market_total += count
        
        market_stats[market] = {
            "total": market_total,
            "types": type_stats,
        }
    
    # Add bars manager stats
    bars_stats = None
    if bars_manager:
        bars_stats = bars_manager.get_stats()
    
    logger.info(
        "throughput",
        extra={
            "total_events": total_events,
            "interval_ms": elapsed_ms,
            "events_per_sec": round(events_per_sec, 2),
            "queue_size": queue_size,
            "futures": market_stats.get("futures", {}),
            "spot": market_stats.get("spot", {}),
            "processed_total": pipeline.processed_count,
            "errors_total": pipeline.error_count,
            "bars": bars_stats,
        },
    )


async def run_http_server(
    app,
    host: str,
    port: int,
    shutdown_event: asyncio.Event,
) -> None:
    """Run uvicorn HTTP server."""
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    
    logger.info(
        "http_server_starting",
        extra={"host": host, "port": port},
    )
    
    try:
        await server.serve()
    except asyncio.CancelledError:
        logger.info("http_server_cancelled")
    
    logger.info("http_server_stopped")


async def main() -> None:
    """Main async entry point."""
    # Log startup
    logger.info(
        "collector_starting",
        extra={
            "version": __version__,
            "schema_version": __schema_version__,
        },
    )
    
    logger.info(
        "config_loaded",
        extra={"config": settings.dump()},
    )
    
    symbol_upper = settings.SYMBOL.upper()
    futures_streams = settings.futures_streams()
    spot_streams = settings.spot_streams()
    
    logger.info(
        "streams_configured",
        extra={
            "symbol": symbol_upper,
            "futures_streams": futures_streams,
            "spot_streams": spot_streams,
            "depth_enabled": settings.DEPTH_ENABLED,
            "depth_speed": settings.DEPTH_SPEED,
        },
    )
    
    # Create shared components
    queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
    bars_out_queue: asyncio.Queue = asyncio.Queue(maxsize=BARS_QUEUE_MAX_SIZE)
    pipeline = EventPipeline(symbol=symbol_upper, topn=settings.TOPN)
    sampler = SamplingLogger()
    metrics = Metrics()
    shutdown_event = asyncio.Event()
    
    # Create bars manager (handles all markets and timeframes)
    bars_manager = BarsManager(symbol=symbol_upper)
    
    # Create last bars store for HTTP API
    last_bars_store = LastBarsStore()
    
    # Create feature engine
    feature_engine = FeatureEngine(symbol=symbol_upper, topn=settings.TOPN)
    
    # Create edge engine
    edge_engine = EdgeEngine(symbol=symbol_upper)
    
    # Create accumulate engine (BLOCK 7)
    from collector.accumulate_engine import AccumulateEngine
    accumulate_engine = AccumulateEngine(
        cooldown_sec=settings.COOLDOWN_SEC,
        max_slices_per_window=settings.MAX_SLICES_PER_WINDOW,
        max_usd_per_window=settings.MAX_USD_PER_WINDOW,
        slice_usd=settings.SLICE_USD,
        window_len_sec=900,
    )
    
    # Create decision logger
    decision_logger = DecisionLogger(buffer_size=1000)
    
    # Create paper trading engine
    paper_engine = PaperEngine(trade_qty=1.0, max_spread_bps=500.0)
    
    # Create event recorder for backtesting (if enabled)
    from collector.event_recorder import EventRecorder
    event_recorder: Optional[EventRecorder] = None
    
    if settings.RECORD_ENABLED:
        event_recorder = EventRecorder(
            output_dir=settings.RECORD_DIR,
            buffer_size=settings.RECORD_BUFFER_SIZE,
            flush_interval_ms=settings.RECORD_FLUSH_INTERVAL_MS,
            enabled=True,
        )
        # Connect to feature_engine and accumulate_engine
        feature_engine.set_event_recorder(event_recorder)
        accumulate_engine.set_event_recorder(event_recorder)
        
        logger.info(
            "event_recorder_configured",
            extra={
                "output_dir": settings.RECORD_DIR,
                "buffer_size": settings.RECORD_BUFFER_SIZE,
            },
        )
    
    # Create storage writer
    storage_writer = StorageWriter(
        output_dir="./data/decisions",
        batch_size=100,
        flush_interval_sec=5.0,
    )
    
    # =========================================
    # CONTEXT BOOTSTRAP (HTF Klines)
    # =========================================
    klines_store: Optional[KlinesStore] = None
    context_bootstrapper: Optional[ContextBootstrapper] = None
    futures_rest_client: Optional[RestClient] = None
    spot_rest_client: Optional[RestClient] = None
    
    if settings.CONTEXT_ENABLED:
        logger.info(
            "context_bootstrap_enabled",
            extra={
                "timeframes": settings.CONTEXT_TFS,
                "limit": settings.CONTEXT_BOOTSTRAP_LIMIT,
                "min_ready_bars": settings.CONTEXT_MIN_READY_BARS,
            },
        )
        
        # Create REST clients
        futures_rest_client = RestClient(
            base_url=settings.FUTURES_REST_BASE,
            name="futures",
            max_concurrency=4,
            min_interval_ms=150,
        )
        spot_rest_client = RestClient(
            base_url=settings.SPOT_REST_BASE,
            name="spot",
            max_concurrency=4,
            min_interval_ms=150,
        )
        
        # Create klines store
        klines_store = KlinesStore()
        
        # Create bootstrapper
        context_bootstrapper = ContextBootstrapper(
            futures_client=futures_rest_client,
            spot_client=spot_rest_client,
            store=klines_store,
            symbol=symbol_upper,
            timeframes=settings.CONTEXT_TFS,
            limit=settings.CONTEXT_BOOTSTRAP_LIMIT,
            min_ready_bars=settings.CONTEXT_MIN_READY_BARS,
        )
        
        # Bootstrap klines BEFORE starting WS clients
        await context_bootstrapper.bootstrap()
        
        logger.info(
            "context_ready_status",
            extra={
                "ready": context_bootstrapper.is_ready(),
                "counts": klines_store.get_counts(),
            },
        )
        
        # =========================================
        # FIX-FAIR-001: Warmup VolEstimator from bootstrap klines
        # =========================================
        # Extract 1m close prices from klines_store and warmup vol + bias
        try:
            futures_1m_klines = klines_store.get("futures", "1m")
            if futures_1m_klines:
                # Kline objects have .close attribute
                closes = [k.close for k in futures_1m_klines if k.close and k.close > 0]
                if closes:
                    from collector.utils_time import now_ms
                    ts = now_ms()
                    
                    # Warmup VolEstimator
                    returns_added = feature_engine._vol_estimator.warmup_from_closes(closes, ts)
                    logger.info(
                        "vol_estimator_warmup_from_bootstrap",
                        extra={
                            "klines_count": len(futures_1m_klines),
                            "closes_count": len(closes),
                            "returns_added": returns_added,
                            "vol_ready": feature_engine._vol_estimator.is_ready(),
                        },
                    )
                    
                    # S4: Warmup BiasModel with 1m closes
                    bias_count = feature_engine._bias_model.warmup_tf("1m", closes, ts)
                    logger.info(
                        "bias_model_warmup_from_bootstrap",
                        extra={
                            "tf": "1m",
                            "closes_count": len(closes),
                            "processed": bias_count,
                        },
                    )
                else:
                    logger.warning(
                        "vol_estimator_warmup_no_closes",
                        extra={"klines_count": len(futures_1m_klines)},
                    )
            else:
                logger.warning("vol_estimator_warmup_no_klines")
            
            # =========================================
            # FIX-BIAS-001: Warmup BiasModel for ALL TFs from bootstrap
            # =========================================
            # Now warmup 5m, 15m, 1h from their respective klines
            for tf in ["5m", "15m", "1h"]:
                try:
                    tf_klines = klines_store.get("futures", tf)
                    if tf_klines:
                        tf_closes = [k.close for k in tf_klines if k.close and k.close > 0]
                        if tf_closes:
                            from collector.utils_time import now_ms
                            ts = now_ms()
                            bias_count = feature_engine._bias_model.warmup_tf(tf, tf_closes, ts)
                            logger.info(
                                "bias_model_warmup_from_bootstrap",
                                extra={
                                    "tf": tf,
                                    "closes_count": len(tf_closes),
                                    "processed": bias_count,
                                },
                            )
                        else:
                            logger.warning(f"bias_model_warmup_no_closes_{tf}")
                    else:
                        logger.warning(f"bias_model_warmup_no_klines_{tf}")
                except Exception as e:
                    logger.warning(
                        f"bias_model_warmup_failed_{tf}",
                        extra={"error": str(e)},
                    )
        except Exception as e:
            logger.warning(
                "vol_estimator_warmup_failed",
                extra={"error": str(e)},
            )
    
    # =========================================
    # TAAPI BOOTSTRAP (Technical Indicators)
    # =========================================
    taapi_store: Optional[TaapiContextStore] = None
    taapi_client: Optional[TaapiAsyncClient] = None
    
    if settings.TAAPI_ENABLED:
        logger.info("taapi_bootstrap_starting")
        
        taapi_store = TaapiContextStore()
        taapi_client = TaapiAsyncClient(
            secret=settings.TAAPI_SECRET,
            exchange=settings.TAAPI_EXCHANGE,
        )
        
        taapi_bootstrapper = TaapiBootstrapper(
            client=taapi_client,
            store=taapi_store,
            symbol=symbol_upper,
            timeframes=settings.TAAPI_TFS,
        )
        
        # Bootstrap TAAPI indicators
        await taapi_bootstrapper.bootstrap()
        
        logger.info(
            "taapi_ready_status",
            extra={
                "ready": taapi_store.is_ready(),
                "summary": taapi_store.get_summary(),
            },
        )
    
    # Create TAAPI context engine and scheduler
    taapi_context_engine: Optional[TaapiContextEngine] = None
    taapi_scheduler: Optional[TaapiScheduler] = None
    
    if settings.TAAPI_ENABLED and taapi_store:
        taapi_context_engine = TaapiContextEngine(taapi_store)
        taapi_scheduler = TaapiScheduler(
            client=taapi_client,
            store=taapi_store,
            symbol=symbol_upper,
            timeframes=settings.TAAPI_TFS,
        )
        
        # Integrate into feature engine
        feature_engine.set_taapi_context_engine(taapi_context_engine)
        
        logger.info(
            "taapi_context_engine_ready",
            extra={
                "summary": taapi_context_engine.get_summary(),
            },
        )
    
    # =========================================
    # POLYMARKET INTEGRATION
    # =========================================
    polymarket_store: Optional[PolymarketBookStore] = None
    polymarket_resolver: Optional[MarketResolver] = None
    polymarket_ws_client: Optional[PolymarketWSClient] = None
    
    if settings.POLYMARKET_ENABLED:
        logger.info("polymarket_integration_starting")
        
        # Create book store (single source of truth)
        polymarket_store = PolymarketBookStore()
        
        # Create spike probe for microspike detection
        spike_probe = SpikeProbe(tick=0.01, min_log_ms=50)
        polymarket_store.set_spike_probe(spike_probe)
        
        # Connect event recorder if enabled
        if event_recorder:
            polymarket_store.set_event_recorder(event_recorder)
        
        # Create market resolver for token IDs
        polymarket_resolver = MarketResolver()
        
        # Create WS client
        polymarket_ws_client = PolymarketWSClient(
            store=polymarket_store,
            resolver=polymarket_resolver,
        )
        
        # Integrate into feature engine
        feature_engine.set_polymarket_book_store(polymarket_store)
        
        logger.info(
            "polymarket_integration_ready",
            extra={
                "ws_url": settings.POLYMARKET_WS_URL,
                "stale_threshold_sec": settings.POLYMARKET_STALE_THRESHOLD_SEC,
            },
        )
    
    # Create context updater for incremental updates
    context_updater: Optional[ContextUpdater] = None
    if settings.CONTEXT_ENABLED and klines_store:
        context_updater = ContextUpdater(
            futures_client=futures_rest_client,
            spot_client=spot_rest_client,
            store=klines_store,
            symbol=symbol_upper,
            timeframes=settings.CONTEXT_TFS,
        )
    
    # Create market context engine for HTF trends/regime
    market_context_engine: Optional[MarketContextEngine] = None
    if settings.CONTEXT_ENABLED and klines_store:
        market_context_engine = MarketContextEngine(
            store=klines_store,
            symbol=symbol_upper,
            market="futures",  # Use futures for context by default
        )
        # Initial update
        market_context_engine.update()
        
        # Integrate context into feature engine
        feature_engine.set_klines_store(klines_store)
        feature_engine.set_market_context_engine(market_context_engine)
    
    # Create HTTP API app (pass klines_store and market_context_engine for API access)
    http_app = create_app(
        metrics,
        last_bars_store,
        feature_engine,
        klines_store,
        market_context_engine,
        edge_engine,
        polymarket_store,
        decision_logger,
        paper_engine,
        storage_writer,
        accumulate_engine,
    )
    
    # Create WebSocket clients
    ws_futures = WSClient(
        name="futures",
        base_url=settings.FUTURES_WS,
        streams=futures_streams,
        out_queue=queue,
        symbol=symbol_upper,
        metrics=metrics,
        market="futures",
    )
    
    ws_spot = WSClient(
        name="spot",
        base_url=settings.SPOT_WS,
        streams=spot_streams,
        out_queue=queue,
        symbol=symbol_upper,
        metrics=metrics,
        market="spot",
    )
    
    # Setup signal handlers
    loop = asyncio.get_running_loop()
    
    def signal_handler(sig: signal.Signals) -> None:
        logger.info("shutdown_signal", extra={"signal": sig.name})
        shutdown_event.set()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    # Create tasks
    ws_futures_task = asyncio.create_task(ws_futures.run_forever(), name="ws_futures")
    ws_spot_task = asyncio.create_task(ws_spot.run_forever(), name="ws_spot")
    consumer_task = asyncio.create_task(
        consumer_loop(
            queue, pipeline, sampler, metrics, bars_manager,
            bars_out_queue, feature_engine, shutdown_event
        ),
        name="consumer",
    )
    bars_consumer_task = asyncio.create_task(
        bars_consumer_loop(bars_out_queue, last_bars_store, feature_engine, shutdown_event),
        name="bars_consumer",
    )
    metrics_logger_task = asyncio.create_task(
        metrics_logger_loop(metrics, shutdown_event),
        name="metrics_logger",
    )
    features_logger_task = asyncio.create_task(
        features_logger_loop(feature_engine, metrics, shutdown_event),
        name="features_logger",
    )
    sanity_logger_task = asyncio.create_task(
        sanity_logger_loop(feature_engine, metrics, shutdown_event),
        name="sanity_logger",
    )
    http_task = asyncio.create_task(
        run_http_server(http_app, settings.HTTP_HOST, settings.HTTP_PORT, shutdown_event),
        name="http_server",
    )
    
    # Start context updater task if enabled
    context_updater_task: Optional[asyncio.Task] = None
    if context_updater:
        context_updater_task = asyncio.create_task(
            context_updater.run_forever(shutdown_event),
            name="context_updater",
        )
    
    # Start market context update task if enabled
    market_context_task: Optional[asyncio.Task] = None
    if market_context_engine:
        market_context_task = asyncio.create_task(
            market_context_update_loop(market_context_engine, shutdown_event),
            name="market_context_updater",
        )
    
    # Start TAAPI scheduler task if enabled
    taapi_scheduler_task: Optional[asyncio.Task] = None
    if taapi_scheduler:
        taapi_scheduler_task = asyncio.create_task(
            taapi_scheduler.run_forever(shutdown_event),
            name="taapi_scheduler",
        )
    
    # Start event recorder if enabled
    if event_recorder:
        await event_recorder.start()
        # Set current market ID based on timestamp
        market_id = f"market_{int(now_ms() // 900_000)}"  # 15-min window ID
        event_recorder.set_current_market(market_id)
        logger.info("event_recorder_started", extra={"market_id": market_id})
    
    # Start edge decision task (unified slow loop)
    edge_decision_task = asyncio.create_task(
        edge_decision_loop(
            edge_engine,
            feature_engine,
            metrics,
            decision_logger,
            paper_engine,
            storage_writer,
            shutdown_event,
            polymarket_store=polymarket_store,
            accumulate_engine=accumulate_engine,
            event_recorder=event_recorder,  # Pass for market ID updates
        ),
        name="edge_decision",
    )
    
    # Start Polymarket WS task if enabled
    polymarket_ws_task: Optional[asyncio.Task] = None
    if polymarket_ws_client:
        polymarket_ws_task = asyncio.create_task(
            polymarket_ws_client.run_forever(shutdown_event),
            name="polymarket_ws",
        )
    
    # Determine context ready status
    context_ready = context_bootstrapper.is_ready() if context_bootstrapper else False
    context_counts = klines_store.get_counts() if klines_store else {}
    
    # Determine TAAPI ready status
    taapi_context_ready = taapi_store.is_ready() if taapi_store else False
    
    logger.info(
        "collector_ready",
        extra={
            "symbol": symbol_upper,
            "futures_streams_count": len(futures_streams),
            "spot_streams_count": len(spot_streams),
            "http_endpoint": f"http://{settings.HTTP_HOST}:{settings.HTTP_PORT}",
            "bar_timeframes": ["5s", "15s", "1m"],
            "bar_markets": ["futures", "spot"],
            "context_enabled": settings.CONTEXT_ENABLED,
            "context_ready": context_ready,
            "context_counts": context_counts,
            "taapi_enabled": settings.TAAPI_ENABLED,
            "taapi_ready": taapi_context_ready,
            "polymarket_enabled": settings.POLYMARKET_ENABLED,
            "features": [
                "CVD", "RVOL", "Impulse", "Microprice", "Basis",
                "AVWAP", "Depth", "Liquidations", "Absorption", "Quality",
                "TAAPI Context", "Edge Engine", "Polymarket"
            ],
        },
    )
    
    # Wait for shutdown signal
    await shutdown_event.wait()
    
    # =========================================
    # GRACEFUL SHUTDOWN (VERY IMPORTANT)
    # =========================================
    logger.info("shutdown_start")
    
    # 1. Stop WebSocket clients (closes connections gracefully)
    logger.info("shutdown_stopping_ws_clients")
    await ws_futures.stop()
    await ws_spot.stop()
    
    # 2. Cancel all asyncio tasks
    logger.info("shutdown_cancelling_tasks")
    all_tasks = [
        ws_futures_task,
        ws_spot_task,
        consumer_task,
        bars_consumer_task,
        metrics_logger_task,
        features_logger_task,
        sanity_logger_task,
        http_task,
    ]
    
    # Add context updater task if it exists
    if context_updater_task:
        all_tasks.append(context_updater_task)
    
    # Add market context task if it exists
    if market_context_task:
        all_tasks.append(market_context_task)
    
    # Add TAAPI scheduler task if it exists
    if taapi_scheduler_task:
        all_tasks.append(taapi_scheduler_task)
    
    # Add edge decision task
    all_tasks.append(edge_decision_task)
    
    # Add Polymarket WS task if it exists
    if polymarket_ws_task:
        all_tasks.append(polymarket_ws_task)
    
    for task in all_tasks:
        task.cancel()
    
    # 3. Wait for all tasks to complete with return_exceptions=True
    logger.info("shutdown_waiting_tasks")
    results = await asyncio.gather(*all_tasks, return_exceptions=True)
    
    # Log any unexpected errors during shutdown
    for task, result in zip(all_tasks, results):
        if isinstance(result, Exception) and not isinstance(result, asyncio.CancelledError):
            logger.warning(
                "shutdown_task_error",
                extra={"task": task.get_name(), "error": str(result)},
            )
    
    # 4. Stop event recorder (flush remaining events)
    if event_recorder:
        logger.info("shutdown_stopping_event_recorder")
        await event_recorder.stop()
        stats = event_recorder.get_stats()
        logger.info(
            "event_recorder_final_stats",
            extra={
                "total_events": stats.get("total_events"),
                "total_written": stats.get("total_written"),
                "markets": stats.get("markets"),
            },
        )
    
    # 4. Close REST clients if they were created
    if futures_rest_client:
        await futures_rest_client.close()
    if spot_rest_client:
        await spot_rest_client.close()
    
    # 5. Close TAAPI client if created
    if taapi_client:
        await taapi_client.close()
    
    # 6. Close Polymarket resolver if created
    if polymarket_resolver:
        await polymarket_resolver.close()
    
    # 7. Close storage writer
    storage_writer.close()
    
    # 8. Log final state
    final_metrics = metrics.snapshot()
    final_features = feature_engine.snapshot()
    
    # Include context info if available
    context_counts = klines_store.get_counts() if klines_store else {}
    
    logger.info(
        "shutdown_complete",
        extra={
            "total_processed": pipeline.processed_count,
            "total_errors": pipeline.error_count,
            "by_market": pipeline.get_counts_by_market(),
            "bars_total": final_metrics.get("bars_total"),
            "context_klines": context_counts,
            "uptime_sec": round(final_metrics.get("uptime_ms", 0) / 1000, 1),
            "final_cvd_futures": final_features.get("futures", {}).get("cvd"),
            "final_cvd_spot": final_features.get("spot", {}).get("cvd"),
        },
    )


def run() -> None:
    """Synchronous entry point."""
    setup_logging(settings.LOG_LEVEL)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("collector_interrupted")
    except asyncio.CancelledError:
        logger.info("collector_cancelled")
    except Exception as e:
        logger.exception(
            "collector_crashed",
            extra={"error": str(e)},
        )
        sys.exit(1)


if __name__ == "__main__":
    run()
