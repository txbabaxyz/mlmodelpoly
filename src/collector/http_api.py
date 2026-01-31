"""
HTTP API Module
===============

FastAPI application providing health check, metrics, features, and control endpoints.

Endpoints:
    GET  /health              - Simple health check
    GET  /state               - Full metrics snapshot including last bars
    GET  /latest/features     - Current computed trading features + anchor state
    GET  /latest/edge         - Current edge decision from EdgeEngine
    POST /control/anchor/mode - Set anchor mode (AUTO_UTC or MANUAL)
    POST /control/anchor/now  - Set anchor to current time (MANUAL mode only)
"""

import logging
from typing import TYPE_CHECKING, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from collector.metrics import Metrics
from collector.utils_time import now_ms

if TYPE_CHECKING:
    from collector.bars import LastBarsStore
    from collector.context_klines_store import KlinesStore
    from collector.decision_logger import DecisionLogger
    from collector.edge_engine import EdgeEngine
    from collector.features import FeatureEngine
    from collector.market_context import MarketContextEngine
    from collector.paper_trading import PaperEngine
    from collector.polymarket.book_store import PolymarketBookStore
    from collector.storage_writer import StorageWriter

logger = logging.getLogger(__name__)


# Request models
class AnchorModeRequest(BaseModel):
    """Request body for setting anchor mode."""
    mode: str


def create_app(
    metrics: Metrics,
    last_bars_store: Optional["LastBarsStore"] = None,
    feature_engine: Optional["FeatureEngine"] = None,
    klines_store: Optional["KlinesStore"] = None,
    market_context_engine: Optional["MarketContextEngine"] = None,
    edge_engine: Optional["EdgeEngine"] = None,
    polymarket_store: Optional["PolymarketBookStore"] = None,
    decision_logger: Optional["DecisionLogger"] = None,
    paper_engine: Optional["PaperEngine"] = None,
    storage_writer: Optional["StorageWriter"] = None,
    accumulate_engine: Optional["AccumulateEngine"] = None,
) -> FastAPI:
    """
    Create FastAPI application with metrics endpoints.
    
    Args:
        metrics: Metrics instance for collecting stats
        last_bars_store: Optional store for last closed bars
        feature_engine: Optional feature engine for computed features
        klines_store: Optional store for HTF klines context
        market_context_engine: Optional market context engine for HTF trends/regime
        edge_engine: Optional edge engine for trading decisions
        polymarket_store: Optional Polymarket book store for prediction market data
        decision_logger: Optional decision logger for edge decision recording
        paper_engine: Optional paper trading engine for virtual execution
        storage_writer: Optional storage writer for persistence
        accumulate_engine: Optional accumulate engine for BLOCK 7
    
    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="Binance Collector",
        description="Real-time Binance market data collector with feature computation and anchor control",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
    )
    
    @app.get("/health")
    async def health() -> JSONResponse:
        """
        Health check endpoint.
        
        Returns:
            {"ok": true}
        """
        return JSONResponse({"ok": True})
    
    @app.get("/state")
    async def state() -> JSONResponse:
        """
        Full metrics snapshot endpoint.
        
        Returns:
            JSON with all metrics including:
            - ws_connected: WebSocket connection status
            - reconnects_total: Reconnect counts
            - events_total: Total event counts
            - events_per_sec: Events per second by type
            - lag_p50_ms: 50th percentile latency
            - lag_p95_ms: 95th percentile latency
            - lag_samples_count: Number of lag samples
            - bars_total: Total bars closed by market/tf
            - last_bars: Last closed bars by market/tf
            - depth: orderbook imbalance and degradation status
            - server_time_ms: Current server time
            - uptime_ms: Server uptime
        """
        snapshot = metrics.snapshot()
        
        # Add last bars if store is available
        if last_bars_store:
            snapshot["last_bars"] = last_bars_store.get_all()
        else:
            snapshot["last_bars"] = {}
        
        # Add feature engine info if available
        if feature_engine:
            # Depth summary
            depth_summary = feature_engine.get_depth_summary()
            snapshot["depth"] = depth_summary
            
            # Final features summary (liquidations, absorption, quality, gaps)
            final_summary = feature_engine.get_final_summary(snapshot)
            snapshot["liquidations"] = {
                "qty_30s": final_summary.get("liq_qty_30s"),
                "count_30s": final_summary.get("liq_count_30s"),
            }
            snapshot["absorption_score_30s"] = final_summary.get("absorption_score_30s")
            snapshot["quality_mode"] = final_summary.get("quality_mode")
            snapshot["gap_detected"] = {
                "futures": final_summary.get("gap_detected_futures"),
                "spot": final_summary.get("gap_detected_spot"),
            }
        
        # Add context summary if market context engine available
        if market_context_engine:
            ctx = market_context_engine.snapshot()
            snapshot["context_summary"] = {
                "trend_5m": ctx.get("trend_5m"),
                "trend_15m": ctx.get("trend_15m"),
                "trend_1h": ctx.get("trend_1h"),
                "alignment_score": ctx.get("trend_alignment_score"),
                "regime_15m": ctx.get("regime_15m"),
                "vol_regime": ctx.get("vol_regime"),
                "data_quality": ctx.get("data_quality"),
            }
        
        return JSONResponse(snapshot)
    
    @app.get("/latest/features")
    async def latest_features() -> JSONResponse:
        """
        Current computed trading features endpoint.
        
        Returns:
            JSON with computed features including:
            - futures: CVD, RVOL, Impulse, Microprice, Mid, Mark
            - spot: CVD, Microprice, Mid
            - basis: basis_mid, basis_bps, premium_to_mark
            - anchor: mode, anchor_time_ms, avwap_15m, dev_from_avwap_bps
            - orderbook: imbalance, change_10s/30s, depth_degraded
            - liquidations: qty/count 30s/60s
            - absorption_score_30s
            - quality: mode, ws_futures/spot, gap_detected, depth_degraded
        """
        if feature_engine:
            # Get metrics snapshot for quality computation
            metrics_snap = metrics.snapshot()
            return JSONResponse(feature_engine.snapshot(metrics_snap))
        else:
            return JSONResponse({
                "error": "Feature engine not initialized",
                "features": None,
            })
    
    @app.get("/latest/edge")
    async def latest_edge() -> JSONResponse:
        """
        Current edge decision endpoint (S6 Decision v2).
        
        Returns:
            JSON with edge decision:
            - candidate_side: "UP" | "DOWN" | "NONE"
            - action: "WAIT" | "WATCH_UP" | "WATCH_DOWN" | "ACCUMULATE_UP" | "ACCUMULATE_DOWN"
            - confidence: 0..1
            - confidence_level: "LOW" | "MED" | "HIGH"
            - net_edge_fast/smooth: {up, down} in bps
            - veto, veto_reasons
            - legacy: direction, edge_score from old EdgeEngine
        """
        from collector.decision_v2 import compute_decision_v2
        from collector.config import settings
        
        # Get current features
        if not feature_engine:
            return JSONResponse({
                "error": "Feature engine not initialized",
                "candidate_side": "NONE",
                "action": "WAIT",
            })
        
        features = feature_engine.snapshot()
        pm_up_down = features.get("polymarket_up_down", {})
        
        # Compute S6 decision
        decision_v2 = compute_decision_v2(
            features=features,
            pm_up_down=pm_up_down,
            edge_buffer_bps=settings.EDGE_BUFFER_BPS,
        )
        
        # Add legacy edge engine data if available
        if edge_engine:
            old_decision = edge_engine.get_last_decision()
            if old_decision:
                decision_v2["legacy"] = {
                    "direction": old_decision.direction,
                    "edge_score": old_decision.edge_score,
                    "confidence": old_decision.confidence,
                    "veto": old_decision.veto,
                }
        
        decision_v2["ts_ms"] = now_ms()
        return JSONResponse(decision_v2)
    
    @app.get("/latest/polymarket")
    async def latest_polymarket() -> JSONResponse:
        """
        Current Polymarket orderbook state endpoint.
        
        Returns:
            JSON with YES/NO orderbook data:
            - market_id: Current market identifier
            - yes: YES token orderbook (mid, spread_bps, depth, imbalance)
            - no: NO token orderbook
            - connected: WebSocket connection status
            - ts_ms: Last update timestamp
            - age_sec: Data age in seconds
            - spike_stats: Microspike statistics (if SpikeProbe enabled)
        """
        if not polymarket_store:
            return JSONResponse({
                "error": "Polymarket store not initialized",
                "connected": False,
            })
        
        return JSONResponse(polymarket_store.snapshot())
    
    @app.get("/latest/spikes")
    async def latest_spikes() -> JSONResponse:
        """
        Current spike detection statistics endpoint.
        
        Returns:
            JSON with spike stats for YES and NO sides:
            - yes/no: {count, median_duration_ms, max_improve_ticks, ...}
            - window_ms: Statistics window (60 seconds)
        """
        if not polymarket_store:
            return JSONResponse({
                "error": "Polymarket store not initialized",
            })
        
        snap = polymarket_store.snapshot()
        spike_stats = snap.get("spike_stats")
        
        if not spike_stats:
            return JSONResponse({
                "error": "SpikeProbe not initialized",
                "spike_stats": None,
            })
        
        return JSONResponse(spike_stats)
    
    @app.get("/latest/decisions")
    async def latest_decisions() -> JSONResponse:
        """
        Recent edge decisions endpoint.
        
        Returns:
            JSON with decision logger stats and recent records:
            - stats: {total_logged, buffer_size, last_log_ts}
            - recent: Last 5 decision records
        """
        if not decision_logger:
            return JSONResponse({
                "error": "DecisionLogger not initialized",
            })
        
        return JSONResponse(decision_logger.snapshot())
    
    @app.get("/latest/paper")
    async def latest_paper() -> JSONResponse:
        """
        Current paper trading state endpoint.
        
        Returns:
            JSON with paper positions and P&L:
            - yes_qty, yes_avg_px: YES position
            - no_qty, no_avg_px: NO position
            - realized_pnl, unrealized_pnl, total_pnl
            - total_trades, skipped_trades
        """
        if not paper_engine:
            return JSONResponse({
                "error": "PaperEngine not initialized",
            })
        
        return JSONResponse({
            "position": paper_engine.snapshot(),
            "recent_trades": paper_engine.get_recent_trades(10),
        })
    
    @app.get("/latest/storage")
    async def latest_storage() -> JSONResponse:
        """
        Storage writer status endpoint.
        
        Returns:
            JSON with storage stats:
            - total_written, total_batches, total_files
            - current_file, current_file_size_kb
            - buffer_size, last_flush_ago_sec
        """
        if not storage_writer:
            return JSONResponse({
                "error": "StorageWriter not initialized",
            })
        
        return JSONResponse(storage_writer.snapshot())

    @app.get("/latest/accumulate")
    async def latest_accumulate() -> JSONResponse:
        """
        Accumulate engine decision endpoint (BLOCK 7).
        
        Returns:
            JSON with latest accumulation decision:
            - action: ACCUMULATE_UP | ACCUMULATE_DOWN | WAIT
            - fair_up, fair_down: Fair probabilities
            - market_up, market_down: Market prices
            - edge_up_bps, edge_down_bps: Raw edge
            - net_edge_up, net_edge_down: Edge after spread cost
            - triggers: spike/dip flags
            - budget status
        """
        if not accumulate_engine:
            return JSONResponse({
                "error": "AccumulateEngine not initialized",
            })
        
        decision = accumulate_engine.get_last_decision()
        if not decision:
            return JSONResponse({
                "action": "WAIT",
                "reason": "no_decision_yet",
            })
        
        return JSONResponse(decision.to_dict())

    @app.post("/control/anchor/mode")
    async def set_anchor_mode(request: AnchorModeRequest) -> JSONResponse:
        """
        Set anchor mode.
        
        Body:
            {"mode": "AUTO_UTC"} or {"mode": "MANUAL"}
        
        Returns:
            Current anchor state
        """
        if not feature_engine:
            raise HTTPException(
                status_code=500,
                detail="Feature engine not initialized",
            )
        
        valid_modes = ("AUTO_UTC", "MANUAL")
        if request.mode not in valid_modes:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid mode: {request.mode}. Must be one of {valid_modes}",
            )
        
        try:
            anchor_state = feature_engine.set_anchor_mode(request.mode)
            
            logger.info(
                "anchor_mode_set_via_api",
                extra={"mode": request.mode},
            )
            
            return JSONResponse({
                "ok": True,
                "anchor": anchor_state,
            })
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/control/anchor/now")
    async def set_anchor_now() -> JSONResponse:
        """
        Set anchor to current time (MANUAL mode only).
        
        Returns:
            Current anchor state
        
        Raises:
            409 Conflict if not in MANUAL mode
        """
        if not feature_engine:
            raise HTTPException(
                status_code=500,
                detail="Feature engine not initialized",
            )
        
        try:
            anchor_state = feature_engine.anchor_now(now_ms())
            
            logger.info(
                "anchor_set_now_via_api",
                extra={"anchor_time_ms": anchor_state.get("anchor_time_ms")},
            )
            
            return JSONResponse({
                "ok": True,
                "anchor": anchor_state,
            })
        except RuntimeError as e:
            # anchor_now() raises RuntimeError if not in MANUAL mode
            raise HTTPException(
                status_code=409,
                detail=str(e),
            )
    
    @app.get("/context/status")
    async def context_status() -> JSONResponse:
        """
        Get HTF klines context status.
        
        Returns:
            JSON with context status:
            - counts: kline counts by (market, tf)
            - last_klines: last kline info by (market, tf)
        """
        if klines_store:
            return JSONResponse(klines_store.snapshot())
        else:
            return JSONResponse({
                "error": "Klines store not initialized",
                "counts": {},
            })
    
    @app.get("/context/klines/{market}/{tf}")
    async def get_context_klines(
        market: str,
        tf: str,
        limit: int = 100,
    ) -> JSONResponse:
        """
        Get HTF klines for a specific market/timeframe.
        
        Args:
            market: "futures" or "spot"
            tf: Timeframe ("1m", "5m", "15m", "1h")
            limit: Max klines to return (default: 100)
        
        Returns:
            JSON with klines array
        """
        if not klines_store:
            return JSONResponse({
                "error": "Klines store not initialized",
                "klines": [],
            })
        
        if market not in ("futures", "spot"):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid market: {market}. Must be 'futures' or 'spot'",
            )
        
        klines = klines_store.get_last(market, tf, limit)
        
        return JSONResponse({
            "market": market,
            "tf": tf,
            "count": len(klines),
            "klines": [k.to_dict() for k in klines],
        })
    
    @app.get("/context")
    async def market_context() -> JSONResponse:
        """
        Get computed market context (trends, regime, volatility).
        
        Returns:
            JSON with market context:
            - trend_5m, trend_15m, trend_1h: UP/DOWN/FLAT
            - trend_alignment_score: 0-100
            - regime_15m: TREND/RANGE
            - adx_15m, atr_15m, atr_1h
            - vol_regime: LOW/NORMAL/HIGH
            - rsi_15m
            - ema_spread_15m_bps
            - last_update_ms
            - data_quality
        """
        if market_context_engine:
            return JSONResponse(market_context_engine.snapshot())
        else:
            return JSONResponse({
                "error": "Market context engine not initialized",
                "trend_5m": None,
                "trend_15m": None,
                "trend_1h": None,
            })
    
    return app
