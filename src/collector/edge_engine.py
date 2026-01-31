"""
Edge Engine v1
==============

Decision engine that combines micro-features (WS) and macro-context (TAAPI)
to produce trading edge signals.

The Edge Engine:
- Does NOT execute trades
- Does NOT replace risk management
- Provides: direction, edge_score, confidence, veto status

Usage:
    engine = EdgeEngine(symbol="BTCUSDT")
    decision = engine.decide(features_snapshot)
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


@dataclass
class EdgeDecision:
    """
    Result of edge decision computation.
    
    Attributes:
        ts_ms: Timestamp of decision
        symbol: Trading symbol
        direction: "UP" | "DOWN" | "NONE"
        edge_score: 0..100 score indicating strength of edge
        confidence: 0..1 confidence level
        veto: True if trade should be blocked
        reasons: List of reasons supporting the direction
        veto_reasons: List of reasons for veto
        payload: Debug data with key features
    """
    ts_ms: int
    symbol: str
    direction: str  # "UP" | "DOWN" | "NONE"
    edge_score: float  # 0..100
    confidence: float  # 0..1
    veto: bool
    reasons: list[str] = field(default_factory=list)
    veto_reasons: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ts_ms": self.ts_ms,
            "symbol": self.symbol,
            "direction": self.direction,
            "edge_score": round(self.edge_score, 2),
            "confidence": round(self.confidence, 3),
            "veto": self.veto,
            "reasons": self.reasons,
            "veto_reasons": self.veto_reasons if self.veto_reasons else None,
            "payload": self.payload,
        }


class EdgeEngine:
    """
    Edge detection engine for 15-minute window trading.
    
    Combines:
    - TAAPI context (bias, regime, alignment)
    - Micro features (RVOL, impulse, AVWAP deviation)
    - Orderbook imbalance
    - Liquidations
    
    To produce actionable edge signals.
    
    Args:
        symbol: Trading symbol (e.g., "BTCUSDT")
    """
    
    # Thresholds (configurable)
    DEV_THRESHOLD_BPS = 6  # Minimum deviation from AVWAP for trigger
    RVOL_CONFIRM_THRESHOLD = 1.5
    IMPULSE_CONFIRM_THRESHOLD = 1.5
    ABSORPTION_CONFIRM_THRESHOLD = 60
    IMBALANCE_CONFIRM_THRESHOLD = 0.10
    IMBALANCE_CHANGE_THRESHOLD = 0.05
    EDGE_SCORE_ACTION_THRESHOLD = 60
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # Polymarket thresholds (loaded from config at runtime)
    PM_SPREAD_PENALTY_PER_100BPS = 5  # Score penalty per 100bps over threshold
    
    def __init__(self, symbol: str) -> None:
        self.symbol = symbol
        self._last_decision: Optional[EdgeDecision] = None
        
        logger.info(
            "edge_engine_initialized",
            extra={"symbol": symbol},
        )
    
    def decide(self, features: dict) -> EdgeDecision:
        """
        Make edge decision based on current features.
        
        Args:
            features: Full features snapshot from FeatureEngine
            
        Returns:
            EdgeDecision with direction, score, confidence, and veto status
        """
        ts = now_ms()
        reasons: list[str] = []
        veto_reasons: list[str] = []
        payload: dict[str, Any] = {}
        
        # ========================================
        # 1. VETO CHECKS (Safety First)
        # ========================================
        veto = False
        
        quality = features.get("quality", {})
        
        # Check quality mode
        quality_mode = quality.get("mode", "BAD")
        if quality_mode != "OK":
            veto = True
            veto_reasons.append(f"quality_mode_{quality_mode}")
        
        # Check context ready (TAAPI)
        taapi_ready = quality.get("taapi_ready", False)
        if not taapi_ready:
            veto = True
            veto_reasons.append("taapi_not_ready")
        
        # Check context ready (Klines)
        context_ready = quality.get("context_ready", False)
        if not context_ready:
            veto = True
            veto_reasons.append("context_not_ready")
        
        # Check gaps
        gap_detected = quality.get("gap_detected", {})
        if gap_detected.get("futures", False):
            veto = True
            veto_reasons.append("gap_futures")
        
        # Check TAAPI stale
        taapi_stale = quality.get("taapi_stale", False)
        if taapi_stale:
            veto = True
            veto_reasons.append("taapi_stale")
        
        # Check Polymarket connectivity
        polymarket_ws = quality.get("polymarket_ws", False)
        polymarket_stale = quality.get("polymarket_stale", False)
        
        # Polymarket data is optional but degrades quality if unavailable
        if not polymarket_ws:
            # Not a hard veto, but note it
            veto_reasons.append("polymarket_ws_disconnected")
        elif polymarket_stale:
            veto_reasons.append("polymarket_stale")
        
        # If veto, return early with NONE
        if veto:
            decision = EdgeDecision(
                ts_ms=ts,
                symbol=self.symbol,
                direction="NONE",
                edge_score=0,
                confidence=0,
                veto=True,
                reasons=[],
                veto_reasons=veto_reasons,
                payload={"quality_mode": quality_mode},
            )
            self._last_decision = decision
            return decision
        
        # ========================================
        # 2. EXTRACT FEATURES
        # ========================================
        
        # TAAPI context
        taapi_context = features.get("taapi_context", {}) or {}
        bias_1h = taapi_context.get("bias_1h", "NEUTRAL")
        bias_15m = taapi_context.get("bias_15m", "NEUTRAL")
        regime_15m = taapi_context.get("regime_15m", "RANGE")
        micro_stage = taapi_context.get("micro_stage", "NEUTRAL")
        alignment_score = taapi_context.get("alignment_score", 0) or 0
        rsi_1m = taapi_context.get("rsi_1m")
        
        payload["bias_1h"] = bias_1h
        payload["bias_15m"] = bias_15m
        payload["regime_15m"] = regime_15m
        payload["micro_stage"] = micro_stage
        payload["alignment_score"] = alignment_score
        
        # Anchor / AVWAP
        anchor = features.get("anchor", {}) or {}
        dev_from_avwap_bps = anchor.get("dev_from_avwap_bps")
        payload["dev_from_avwap_bps"] = dev_from_avwap_bps
        
        # Basis
        basis = features.get("basis", {}) or {}
        basis_bps = basis.get("basis_bps")
        payload["basis_bps"] = basis_bps
        
        # Micro 1m features
        micro_1m = features.get("micro_1m", {}) or {}
        futures_micro = micro_1m.get("futures", {}) or {}
        micro_trend_1m = futures_micro.get("trend", "FLAT")
        micro_stage_1m = futures_micro.get("stage", "NEUTRAL")
        payload["micro_trend_1m"] = micro_trend_1m
        payload["micro_stage_1m"] = micro_stage_1m
        
        # Futures features
        futures = features.get("futures", {}) or {}
        rvol_5s = futures.get("rvol_5s")
        impulse_5s = futures.get("impulse_5s")
        payload["rvol_5s"] = rvol_5s
        payload["impulse_5s"] = impulse_5s
        
        # Orderbook
        orderbook = features.get("orderbook", {}) or {}
        imbalance = orderbook.get("imbalance")
        imbalance_change_10s = orderbook.get("imbalance_change_10s")
        depth_degraded = orderbook.get("depth_degraded", False)
        payload["imbalance"] = imbalance
        payload["imbalance_change_10s"] = imbalance_change_10s
        
        # Liquidations
        liquidations = features.get("liquidations", {}) or {}
        liq_qty_30s = liquidations.get("qty_30s", 0) or 0
        liq_count_30s = liquidations.get("count_30s", 0) or 0
        payload["liq_qty_30s"] = liq_qty_30s
        payload["liq_count_30s"] = liq_count_30s
        
        # Absorption score
        absorption_score = features.get("absorption_score_30s")
        payload["absorption_score_30s"] = absorption_score
        
        # ========== POLYMARKET DATA ==========
        polymarket = features.get("polymarket", {}) or {}
        pm_yes = polymarket.get("yes", {}) or {}
        pm_no = polymarket.get("no", {}) or {}
        pm_connected = polymarket.get("connected", False)
        
        # YES token metrics
        pm_yes_mid = pm_yes.get("mid")
        pm_yes_spread_bps = pm_yes.get("spread_bps")
        pm_yes_depth_top3 = pm_yes.get("depth_top3", 0)
        
        # NO token metrics
        pm_no_mid = pm_no.get("mid")
        pm_no_spread_bps = pm_no.get("spread_bps")
        pm_no_depth_top3 = pm_no.get("depth_top3", 0)
        
        payload["pm_connected"] = pm_connected
        payload["pm_yes_mid"] = pm_yes_mid
        payload["pm_no_mid"] = pm_no_mid
        payload["pm_yes_spread_bps"] = pm_yes_spread_bps
        payload["pm_no_spread_bps"] = pm_no_spread_bps
        payload["pm_yes_depth_top3"] = pm_yes_depth_top3
        payload["pm_no_depth_top3"] = pm_no_depth_top3
        
        # ========================================
        # 3. DETERMINE OVERALL BIAS
        # ========================================
        overall_bias = self._compute_overall_bias(
            bias_1h, bias_15m, alignment_score, micro_stage
        )
        payload["overall_bias"] = overall_bias
        
        # ========================================
        # 4. CHECK TRIGGERS
        # ========================================
        up_trigger = self._check_up_trigger(
            overall_bias, micro_stage_1m, dev_from_avwap_bps,
            rvol_5s, impulse_5s, absorption_score,
            imbalance, imbalance_change_10s
        )
        
        down_trigger = self._check_down_trigger(
            overall_bias, micro_stage_1m, dev_from_avwap_bps,
            rvol_5s, impulse_5s, absorption_score,
            imbalance, imbalance_change_10s
        )
        
        payload["up_trigger"] = up_trigger
        payload["down_trigger"] = down_trigger
        
        # ========================================
        # 5. CALCULATE EDGE SCORES
        # ========================================
        up_score = 0
        down_score = 0
        up_reasons: list[str] = []
        down_reasons: list[str] = []
        
        if up_trigger:
            up_score, up_reasons = self._calculate_edge_score(
                direction="UP",
                overall_bias=overall_bias,
                regime_15m=regime_15m,
                alignment_score=alignment_score,
                dev_from_avwap_bps=dev_from_avwap_bps,
                rvol_5s=rvol_5s,
                impulse_5s=impulse_5s,
                absorption_score=absorption_score,
                basis_bps=basis_bps,
                rsi_1m=rsi_1m,
                depth_degraded=depth_degraded,
            )
        
        if down_trigger:
            down_score, down_reasons = self._calculate_edge_score(
                direction="DOWN",
                overall_bias=overall_bias,
                regime_15m=regime_15m,
                alignment_score=alignment_score,
                dev_from_avwap_bps=dev_from_avwap_bps,
                rvol_5s=rvol_5s,
                impulse_5s=impulse_5s,
                absorption_score=absorption_score,
                basis_bps=basis_bps,
                rsi_1m=rsi_1m,
                depth_degraded=depth_degraded,
            )
        
        # ========================================
        # 6. SELECT DIRECTION
        # ========================================
        direction = "NONE"
        edge_score = 0.0
        
        if up_score >= self.EDGE_SCORE_ACTION_THRESHOLD and up_score > down_score:
            direction = "UP"
            edge_score = up_score
            reasons = up_reasons
        elif down_score >= self.EDGE_SCORE_ACTION_THRESHOLD and down_score > up_score:
            direction = "DOWN"
            edge_score = down_score
            reasons = down_reasons
        elif up_score >= self.EDGE_SCORE_ACTION_THRESHOLD and down_score >= self.EDGE_SCORE_ACTION_THRESHOLD:
            # Both triggers active - choose by deviation strength
            if dev_from_avwap_bps is not None:
                if dev_from_avwap_bps < 0:  # Price below AVWAP
                    direction = "UP"
                    edge_score = up_score
                    reasons = up_reasons
                else:
                    direction = "DOWN"
                    edge_score = down_score
                    reasons = down_reasons
        
        payload["up_score"] = up_score
        payload["down_score"] = down_score
        
        # ========================================
        # 6.5. POLYMARKET ADJUSTMENTS (using UP/DOWN mapping)
        # ========================================
        from collector.config import settings
        
        pm_penalty = 0.0
        pm_veto = False
        
        # Get UP/DOWN mapped data
        pm_up_down = features.get("polymarket_up_down") or {}
        up_book = pm_up_down.get("up") or {}
        down_book = pm_up_down.get("down") or {}
        
        # Check if we have prices at all
        up_ask = up_book.get("best_ask")
        down_ask = down_book.get("best_ask")
        
        if up_ask is None or down_ask is None:
            if direction != "NONE":
                pm_veto = True
                veto_reasons.append("polymarket_no_prices")
        
        if pm_connected and direction != "NONE" and not pm_veto:
            # Select relevant side based on direction
            if direction == "UP":
                pm_book = up_book
                token_name = "UP"
            else:  # DOWN
                pm_book = down_book
                token_name = "DOWN"
            
            pm_depth = pm_book.get("depth_top3", 0)
            pm_spread = pm_book.get("spread_bps")
            pm_mid = pm_book.get("mid")
            pm_ask = pm_book.get("best_ask")
            
            payload["pm_selected_token"] = token_name
            payload["pm_selected_depth"] = pm_depth
            payload["pm_selected_spread"] = pm_spread
            payload["pm_selected_mid"] = pm_mid
            payload["pm_selected_ask"] = pm_ask
            
            # Liquidity veto: not enough depth
            if pm_depth is not None and pm_depth < settings.POLYMARKET_MIN_DEPTH:
                pm_veto = True
                veto_reasons.append("polymarket_low_depth")
            
            # Spread veto: too wide spread
            if pm_spread is not None and pm_spread > settings.POLYMARKET_MAX_SPREAD_BPS:
                pm_veto = True
                veto_reasons.append("polymarket_spread_too_wide")
            
            # Spread penalty (even if not veto, still penalize high spread)
            if pm_spread is not None and pm_spread > 100:  # Penalty starts at 100 bps
                excess_spread = pm_spread - 100
                pm_penalty = min(20, (excess_spread / 100) * self.PM_SPREAD_PENALTY_PER_100BPS)
                if pm_penalty > 0:
                    reasons.append("spread_penalty")
            
            # Apply penalty to edge_score
            edge_score = max(0, edge_score - pm_penalty)
            payload["pm_penalty"] = pm_penalty
        
        # If PM veto triggered
        if pm_veto:
            direction = "NONE"
            edge_score = 0
        
        # ========================================
        # 7. CALCULATE CONFIDENCE
        # ========================================
        confidence = edge_score / 100.0 if edge_score > 0 else 0.0
        
        # Adjust confidence
        if regime_15m == "RANGE":
            confidence *= 0.8
        if alignment_score < 40:
            confidence *= 0.7
        
        confidence = max(0.0, min(1.0, confidence))
        
        # ========================================
        # 8. BUILD DECISION
        # ========================================
        # Determine final veto status
        final_veto = pm_veto  # Currently only PM can soft-veto after initial checks
        
        decision = EdgeDecision(
            ts_ms=ts,
            symbol=self.symbol,
            direction=direction,
            edge_score=edge_score,
            confidence=confidence,
            veto=final_veto,
            reasons=reasons,
            veto_reasons=veto_reasons if veto_reasons else [],
            payload=payload,
        )
        
        self._last_decision = decision
        return decision
    
    def _compute_overall_bias(
        self,
        bias_1h: str,
        bias_15m: str,
        alignment_score: float,
        micro_stage: str,
    ) -> str:
        """
        Compute overall directional bias.
        
        Returns:
            "UP", "DOWN", or "NEUTRAL"
        """
        # Strong alignment
        if bias_1h == "UP" and bias_15m == "UP":
            return "UP"
        if bias_1h == "DOWN" and bias_15m == "DOWN":
            return "DOWN"
        
        # High alignment score with 15m bias
        if alignment_score >= 70:
            if bias_15m == "UP":
                return "UP"
            elif bias_15m == "DOWN":
                return "DOWN"
        
        # Micro stage override for strong acceleration
        if micro_stage == "ACCEL_UP" and bias_15m != "DOWN":
            return "UP"
        if micro_stage == "ACCEL_DOWN" and bias_15m != "UP":
            return "DOWN"
        
        return "NEUTRAL"
    
    def _check_up_trigger(
        self,
        overall_bias: str,
        micro_stage_1m: str,
        dev_from_avwap_bps: Optional[float],
        rvol_5s: Optional[float],
        impulse_5s: Optional[float],
        absorption_score: Optional[float],
        imbalance: Optional[float],
        imbalance_change_10s: Optional[float],
    ) -> bool:
        """Check if UP trigger conditions are met."""
        # Bias check
        bias_ok = overall_bias == "UP" or (
            overall_bias == "NEUTRAL" and micro_stage_1m == "ACCEL_UP"
        )
        if not bias_ok:
            return False
        
        # Deviation check (price below AVWAP = cheap)
        if dev_from_avwap_bps is None or dev_from_avwap_bps > -self.DEV_THRESHOLD_BPS:
            return False
        
        # At least one confirmation
        confirmations = 0
        
        if rvol_5s is not None and rvol_5s >= self.RVOL_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if impulse_5s is not None and impulse_5s >= self.IMPULSE_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if absorption_score is not None and absorption_score >= self.ABSORPTION_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if imbalance is not None and imbalance > self.IMBALANCE_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if imbalance_change_10s is not None and imbalance_change_10s > self.IMBALANCE_CHANGE_THRESHOLD:
            confirmations += 1
        
        return confirmations >= 1
    
    def _check_down_trigger(
        self,
        overall_bias: str,
        micro_stage_1m: str,
        dev_from_avwap_bps: Optional[float],
        rvol_5s: Optional[float],
        impulse_5s: Optional[float],
        absorption_score: Optional[float],
        imbalance: Optional[float],
        imbalance_change_10s: Optional[float],
    ) -> bool:
        """Check if DOWN trigger conditions are met."""
        # Bias check
        bias_ok = overall_bias == "DOWN" or (
            overall_bias == "NEUTRAL" and micro_stage_1m == "ACCEL_DOWN"
        )
        if not bias_ok:
            return False
        
        # Deviation check (price above AVWAP = expensive)
        if dev_from_avwap_bps is None or dev_from_avwap_bps < self.DEV_THRESHOLD_BPS:
            return False
        
        # At least one confirmation
        confirmations = 0
        
        if rvol_5s is not None and rvol_5s >= self.RVOL_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if impulse_5s is not None and impulse_5s >= self.IMPULSE_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if absorption_score is not None and absorption_score >= self.ABSORPTION_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if imbalance is not None and imbalance < -self.IMBALANCE_CONFIRM_THRESHOLD:
            confirmations += 1
        
        if imbalance_change_10s is not None and imbalance_change_10s < -self.IMBALANCE_CHANGE_THRESHOLD:
            confirmations += 1
        
        return confirmations >= 1
    
    def _calculate_edge_score(
        self,
        direction: str,
        overall_bias: str,
        regime_15m: str,
        alignment_score: float,
        dev_from_avwap_bps: Optional[float],
        rvol_5s: Optional[float],
        impulse_5s: Optional[float],
        absorption_score: Optional[float],
        basis_bps: Optional[float],
        rsi_1m: Optional[float],
        depth_degraded: bool,
    ) -> tuple[float, list[str]]:
        """
        Calculate edge score for a direction.
        
        Returns:
            (score, reasons) tuple
        """
        score = 0.0
        reasons: list[str] = []
        
        # ========== BASE POINTS ==========
        
        # Bias alignment (+25)
        if overall_bias == direction:
            score += 25
            reasons.append(f"BIAS_{direction}")
        
        # Trend regime with aligned bias (+15)
        if regime_15m == "TREND" and overall_bias == direction:
            score += 15
            reasons.append("TREND_REGIME")
        
        # High alignment score (+10)
        if alignment_score >= 70:
            score += 10
            reasons.append("HIGH_ALIGNMENT")
        
        # ========== MISPRICING POINTS ==========
        
        if dev_from_avwap_bps is not None:
            dev_abs = abs(dev_from_avwap_bps)
            mispricing_points = min(25, dev_abs * 2)
            score += mispricing_points
            
            if direction == "UP" and dev_from_avwap_bps < 0:
                reasons.append("CHEAP_VS_AVWAP")
            elif direction == "DOWN" and dev_from_avwap_bps > 0:
                reasons.append("EXPENSIVE_VS_AVWAP")
        
        # ========== CONFIRMATION POINTS ==========
        
        # RVOL spike (+10)
        if rvol_5s is not None and rvol_5s >= 2.0:
            score += 10
            reasons.append("RVOL_SPIKE")
        
        # Impulse (+10)
        if impulse_5s is not None and impulse_5s >= 2.0:
            score += 10
            reasons.append("IMPULSE_HIGH")
        
        # Absorption (+10)
        if absorption_score is not None and absorption_score >= 80:
            score += 10
            reasons.append("ABSORPTION")
        
        # Basis support (+5)
        if basis_bps is not None:
            if direction == "UP" and basis_bps > 0:
                score += 5
                reasons.append("BASIS_SUPPORT")
            elif direction == "DOWN" and basis_bps < 0:
                score += 5
                reasons.append("BASIS_SUPPORT")
        
        # ========== PENALTIES ==========
        
        # Overheat penalty (-15)
        if rsi_1m is not None:
            if direction == "UP" and rsi_1m > self.RSI_OVERBOUGHT:
                score -= 15
                reasons.append("RSI_OVERBOUGHT_PENALTY")
            elif direction == "DOWN" and rsi_1m < self.RSI_OVERSOLD:
                score -= 15
                reasons.append("RSI_OVERSOLD_PENALTY")
        
        # Depth degraded (-10)
        if depth_degraded:
            score -= 10
            reasons.append("DEPTH_DEGRADED_PENALTY")
        
        # Basis against direction (-10)
        if basis_bps is not None:
            if direction == "UP" and basis_bps < -5:
                score -= 10
                reasons.append("BASIS_AGAINST_PENALTY")
            elif direction == "DOWN" and basis_bps > 5:
                score -= 10
                reasons.append("BASIS_AGAINST_PENALTY")
        
        # Clamp to 0..100
        score = max(0, min(100, score))
        
        return score, reasons
    
    def get_last_decision(self) -> Optional[EdgeDecision]:
        """Get the last computed decision."""
        return self._last_decision
    
    def get_summary(self) -> dict:
        """Get short summary for logging."""
        if not self._last_decision:
            return {"ready": False}
        
        d = self._last_decision
        return {
            "direction": d.direction,
            "edge_score": round(d.edge_score, 1),
            "confidence": round(d.confidence, 2),
            "veto": d.veto,
        }
