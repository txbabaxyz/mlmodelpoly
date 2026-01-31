"""
DecisionLogger — Edge Decision Recording
=========================================

Records every EdgeEngine decision as a structured log entry.

Purpose:
- Build dataset for calibration and analysis
- Track all decisions (including vetoed ones)
- Enable replay and backtesting

Frequency: 1 second (not on every WS update)

Usage:
    logger = DecisionLogger()
    
    # In decision loop (every 1 second):
    logger.log(
        ts_ms=now_ms(),
        features=feature_engine.snapshot(metrics),
        edge=edge_decision,
        polymarket=polymarket_store.snapshot(),
        paper=paper_position,  # optional
    )

Output format:
    {
        "ts": 1234567890,
        "edge": {"direction": "UP", "edge_score": 72, ...},
        "context": {"bias_15m": "bullish", "regime_15m": "trending", ...},
        "micro": {"dev_from_avwap_bps": -8.2, "rvol_5s": 2.1, ...},
        "polymarket": {"yes_mid": 0.47, "yes_spread_bps": 38, ...}
    }
"""

import logging
from collections import deque
from dataclasses import asdict
from typing import Any, Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


class DecisionLogger:
    """
    Logs every edge decision as a structured record.
    
    Each log entry contains:
    - ts: Timestamp in milliseconds
    - edge: EdgeDecision fields (direction, score, confidence, veto, reasons)
    - context: TAAPI context (bias, regime, alignment)
    - micro: Micro features (AVWAP deviation, rvol, impulse)
    - polymarket: Polymarket state (mid, spread, depth)
    - paper: Paper position (if provided)
    
    Thread Safety:
        Methods should be called from a single thread (the decision loop).
        
    Args:
        buffer_size: Max records to keep in memory (default: 1000)
    """
    
    def __init__(self, buffer_size: int = 1000) -> None:
        self._buffer: deque[dict] = deque(maxlen=buffer_size)
        self._total_logged = 0
        self._last_log_ts = 0
        
        logger.info(
            "decision_logger_initialized",
            extra={"buffer_size": buffer_size},
        )
    
    def log(
        self,
        ts_ms: int,
        features: dict,
        edge: Any,  # EdgeDecision dataclass
        polymarket: Optional[dict] = None,
        paper: Optional[dict] = None,
        accumulate: Optional[Any] = None,  # AccumulateDecision dataclass
    ) -> dict:
        """
        Log a single decision record.
        
        Args:
            ts_ms: Timestamp in milliseconds
            features: Full feature snapshot from FeatureEngine
            edge: EdgeDecision dataclass instance
            polymarket: Polymarket snapshot (optional, may be in features)
            paper: Paper position (optional)
            accumulate: AccumulateDecision dataclass (BLOCK 8)
            
        Returns:
            The logged record
        """
        # Build edge section
        edge_dict = self._extract_edge(edge)
        
        # Build context section (TAAPI)
        context_dict = self._extract_context(features)
        
        # Build micro section
        micro_dict = self._extract_micro(features)
        
        # Build polymarket section
        pm_dict = self._extract_polymarket(features, polymarket)
        
        # Build quality section
        quality_dict = self._extract_quality(features)
        
        # BLOCK 8: Build accumulate section
        accum_dict = self._extract_accumulate(features, accumulate)
        
        # BLOCK 8: Build spikes/dips section
        signals_dict = self._extract_signals(features)
        
        # Build record
        record = {
            "ts": ts_ms,
            "edge": edge_dict,
            "context": context_dict,
            "micro": micro_dict,
            "polymarket": pm_dict,
            "quality": quality_dict,
            "accumulate": accum_dict,
            "signals": signals_dict,
        }
        
        # Add paper position if provided
        if paper:
            record["paper"] = paper
        
        # Store in buffer
        self._buffer.append(record)
        self._total_logged += 1
        self._last_log_ts = ts_ms
        
        # Log to structured logger
        logger.info(
            "decision_record",
            extra=record,
        )
        
        return record
    
    def _extract_edge(self, edge: Any) -> dict:
        """Extract edge decision fields."""
        if edge is None:
            return {
                "direction": "NONE",
                "edge_score": 0,
                "confidence": 0,
                "veto": True,
                "veto_reasons": ["no_edge_decision"],
                "reasons": [],
            }
        
        # Handle dataclass
        if hasattr(edge, "__dataclass_fields__"):
            return {
                "direction": edge.direction,
                "edge_score": round(edge.edge_score, 2),
                "confidence": round(edge.confidence, 3),
                "veto": edge.veto,
                "veto_reasons": edge.veto_reasons,
                "reasons": edge.reasons,
            }
        
        # Handle dict
        if isinstance(edge, dict):
            return {
                "direction": edge.get("direction", "NONE"),
                "edge_score": round(edge.get("edge_score", 0), 2),
                "confidence": round(edge.get("confidence", 0), 3),
                "veto": edge.get("veto", True),
                "veto_reasons": edge.get("veto_reasons", []),
                "reasons": edge.get("reasons", []),
            }
        
        return {"error": "unknown_edge_type"}
    
    def _extract_context(self, features: dict) -> dict:
        """Extract TAAPI context fields."""
        taapi = features.get("taapi_context") or {}
        
        return {
            "bias_1h": taapi.get("bias_1h"),
            "bias_15m": taapi.get("bias_15m"),
            "regime_15m": taapi.get("regime_15m"),
            "micro_stage": taapi.get("micro_stage"),
            "alignment_score": taapi.get("alignment_score"),
            "rsi_1m": taapi.get("rsi_1m"),
            "context_ready": taapi.get("context_ready", False),
        }
    
    def _extract_micro(self, features: dict) -> dict:
        """Extract micro features."""
        micro = features.get("micro_1m") or {}
        
        return {
            "dev_from_avwap_bps": self._round_safe(micro.get("dev_from_avwap_bps"), 2),
            "rvol_5s": self._round_safe(micro.get("rvol_5s"), 2),
            "impulse_5s": self._round_safe(micro.get("impulse_5s"), 2),
            "bar_range_bps": self._round_safe(micro.get("bar_range_bps"), 2),
            "close_vs_vwap": micro.get("close_vs_vwap"),
            "trade_count_5s": micro.get("trade_count_5s"),
            "avwap": self._round_safe(micro.get("avwap"), 2),
        }
    
    def _extract_polymarket(
        self,
        features: dict,
        polymarket: Optional[dict] = None,
    ) -> dict:
        """Extract Polymarket fields."""
        # Use provided polymarket or extract from features
        pm = polymarket or features.get("polymarket") or {}
        
        if not pm:
            return {
                "connected": False,
                "yes_mid": None,
                "no_mid": None,
            }
        
        yes = pm.get("yes") or {}
        no = pm.get("no") or {}
        
        return {
            "connected": pm.get("connected", False),
            "market_id": pm.get("market_id"),
            "age_sec": pm.get("age_sec"),
            "yes_mid": self._round_safe(yes.get("mid"), 4),
            "yes_spread_bps": self._round_safe(yes.get("spread_bps"), 1),
            "yes_depth_top1": self._round_safe(yes.get("depth_top1"), 1),
            "yes_depth_top3": self._round_safe(yes.get("depth_top3"), 1),
            "no_mid": self._round_safe(no.get("mid"), 4),
            "no_spread_bps": self._round_safe(no.get("spread_bps"), 1),
            "no_depth_top1": self._round_safe(no.get("depth_top1"), 1),
            "no_depth_top3": self._round_safe(no.get("depth_top3"), 1),
        }
    
    def _extract_quality(self, features: dict) -> dict:
        """Extract quality assessment fields."""
        quality = features.get("quality") or {}
        
        return {
            "mode": quality.get("mode", "UNKNOWN"),
            "trade_mode": quality.get("trade_mode", "UNKNOWN"),
            "trade_reasons": quality.get("trade_reasons", []),
            "binance_ws": quality.get("binance_ws"),
            "taapi_age_sec": quality.get("taapi_age_sec"),
            "polymarket_ws": quality.get("polymarket_ws"),
            "polymarket_age_sec": quality.get("polymarket_age_sec"),
        }
    
    def _extract_accumulate(self, features: dict, accumulate: Optional[Any]) -> dict:
        """Extract accumulate decision fields (BLOCK 8 + CAL1)."""
        # Get fair values from features
        fair = features.get("fair") or {}
        pm_updown = features.get("polymarket_up_down") or {}
        up = pm_updown.get("up") or {}
        down = pm_updown.get("down") or {}
        pm_dips = features.get("pm_dips") or {}
        spikes = features.get("spikes") or {}
        
        result = {
            # Fair values
            "fair_up": self._round_safe(fair.get("fair_up"), 4),
            "fair_down": self._round_safe(fair.get("fair_down"), 4),
            # Market values
            "market_up": self._round_safe(up.get("mid"), 4),
            "market_down": self._round_safe(down.get("mid"), 4),
            "ask_up": self._round_safe(up.get("ask"), 4),
            "ask_down": self._round_safe(down.get("ask"), 4),
            # Required edge from slice model
            "required_edge_up": self._round_safe(up.get("required_edge_bps"), 1),
            "required_edge_down": self._round_safe(down.get("required_edge_bps"), 1),
            # Slice OK
            "slice_ok_up": up.get("slice_ok", False),
            "slice_ok_down": down.get("slice_ok", False),
            # CAL1: Raw values for debugging
            "raw_ret_5s_bps": self._round_safe(spikes.get("ret_5s_bps"), 2),
            "raw_z_ret_5s": self._round_safe(spikes.get("z_ret_5s"), 2),
            "raw_pm_up_mid": self._round_safe(pm_dips.get("up_mid"), 4),
            "raw_pm_down_mid": self._round_safe(pm_dips.get("down_mid"), 4),
            "raw_pm_up_max_60s": self._round_safe(pm_dips.get("up_max_60s"), 4),
            "raw_pm_down_max_60s": self._round_safe(pm_dips.get("down_max_60s"), 4),
            "raw_pm_up_dip_bps": self._round_safe(pm_dips.get("up_dip_bps"), 1),
            "raw_pm_down_dip_bps": self._round_safe(pm_dips.get("down_dip_bps"), 1),
        }
        
        # If accumulate decision provided, add computed fields
        if accumulate and hasattr(accumulate, "action"):
            # Build triggers string for easy filtering
            triggers = []
            if accumulate.down_spike:
                triggers.append("down_spike")
            if accumulate.up_spike:
                triggers.append("up_spike")
            if accumulate.up_dip:
                triggers.append("up_dip")
            if accumulate.down_dip:
                triggers.append("down_dip")
            
            result.update({
                "action": accumulate.action,
                "edge_up_bps": self._round_safe(accumulate.edge_up_bps, 1),
                "edge_down_bps": self._round_safe(accumulate.edge_down_bps, 1),
                "net_edge_up": self._round_safe(accumulate.net_edge_up, 1),
                "net_edge_down": self._round_safe(accumulate.net_edge_down, 1),
                "reasons": accumulate.reasons,
                "veto_reasons": accumulate.veto_reasons,
                # CAL1: Sanity warnings
                "sanity_warnings": accumulate.sanity_warnings if accumulate.sanity_warnings else None,
                "triggers_str": ",".join(triggers) if triggers else None,
            })
        else:
            result.update({
                "action": "N/A",
                "edge_up_bps": None,
                "edge_down_bps": None,
                "net_edge_up": None,
                "net_edge_down": None,
                "reasons": [],
                "veto_reasons": [],
                "sanity_warnings": None,
                "triggers_str": None,
            })
        
        return result
    
    def _extract_signals(self, features: dict) -> dict:
        """Extract spike/dip signals (BLOCK 8)."""
        spikes = features.get("spikes") or {}
        pm_dips = features.get("pm_dips") or {}
        market_ref = features.get("market_ref") or {}
        vol = features.get("vol") or {}
        
        return {
            # Binance spikes (BLOCK 5)
            "ret_5s_bps": self._round_safe(spikes.get("ret_5s_bps"), 2),
            "z_ret_5s": self._round_safe(spikes.get("z_ret_5s"), 2),
            "down_spike_5s": spikes.get("down_spike_5s", False),
            "up_spike_5s": spikes.get("up_spike_5s", False),
            # PM dips (BLOCK 6)
            "up_dip_bps": self._round_safe(pm_dips.get("up_dip_bps"), 1),
            "up_dip": pm_dips.get("up_dip", False),
            "down_dip_bps": self._round_safe(pm_dips.get("down_dip_bps"), 1),
            "down_dip": pm_dips.get("down_dip", False),
            # Market ref (BLOCK 2)
            "ref_px": market_ref.get("ref_px"),
            "s_now": market_ref.get("S_now"),
            "tau_sec": market_ref.get("tau_sec"),
            # Vol (BLOCK 3)
            "sigma_15m": self._round_safe(vol.get("sigma_15m"), 6),
        }
    
    def _round_safe(self, value: Any, decimals: int) -> Any:
        """Round value safely, handling None."""
        if value is None:
            return None
        try:
            return round(float(value), decimals)
        except (TypeError, ValueError):
            return value
    
    def get_recent(self, n: int = 10) -> list[dict]:
        """Get N most recent records."""
        records = list(self._buffer)
        return records[-n:] if n < len(records) else records
    
    def get_stats(self) -> dict:
        """Get logging statistics."""
        return {
            "total_logged": self._total_logged,
            "buffer_size": len(self._buffer),
            "buffer_max": self._buffer.maxlen,
            "last_log_ts": self._last_log_ts,
        }
    
    def snapshot(self) -> dict:
        """Full snapshot for API."""
        recent = self.get_recent(5)
        stats = self.get_stats()
        
        return {
            "stats": stats,
            "recent": recent,
        }
