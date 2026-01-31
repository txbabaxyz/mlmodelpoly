"""
Confidence Engine (S5) + TAAPI Integration
==========================================

Computes unified confidence score (0..1) for trading decisions.

Why this exists:
    Confidence aggregates multiple signals to determine how "sure" we are
    about a potential trade. High confidence = multiple factors align.
    
    Components:
    1. Net edge quality (how much edge after spread/buffer)
    2. Agreement between fair_fast, fair_smooth, and bias
    3. Event triggers (spike/dip supporting the trade)
    4. Data quality (OK/DEGRADED/BAD)
    5. TAAPI alignment bonus (if TFs align)

TAAPI Integration:
    - taapi_alignment (0-100): Bonus for TF alignment
    - taapi_accel_up/down: Micro stage triggers
    - taapi_overheated: Penalty for overbought/oversold

Output:
    - confidence: 0..1 score
    - level: "HIGH" | "MED" | "LOW"
    - reasons: list of positive factors
    - penalties: list of negative factors
    - components: breakdown for debugging

Configuration:
    CONF_WEIGHT_NET_EDGE: 0.35
    CONF_WEIGHT_AGREEMENT: 0.30
    CONF_WEIGHT_EVENTS: 0.20
    CONF_WEIGHT_QUALITY: 0.15
    CONF_LOW_THRESHOLD: 0.3
    CONF_HIGH_THRESHOLD: 0.7

Usage:
    conf = compute_confidence(
        candidate_side="UP",
        fair_fast={"up": 0.55, "down": 0.45},
        fair_smooth={"up": 0.52, "down": 0.48},
        bias={"dir": "UP", "strength": 0.6},
        net_edge_bps=150.0,
        triggers={"up_spike": False, "down_dip": True},
        quality_mode="OK",
        taapi_alignment=70,
    )
    # {
    #   "confidence": 0.72,
    #   "level": "HIGH",
    #   "reasons": ["fair_fast_aligned", "fair_smooth_aligned", "bias_aligned", "down_dip_trigger"],
    #   "penalties": [],
    #   "components": {...}
    # }
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Default weights
DEFAULT_WEIGHT_NET_EDGE = 0.35
DEFAULT_WEIGHT_AGREEMENT = 0.30
DEFAULT_WEIGHT_EVENTS = 0.20
DEFAULT_WEIGHT_QUALITY = 0.15

# Thresholds
DEFAULT_LOW_THRESHOLD = 0.3
DEFAULT_HIGH_THRESHOLD = 0.7

# Net edge sigmoid scale (bps)
NET_EDGE_SCALE = 200.0


def sigmoid(x: float, scale: float = 1.0) -> float:
    """Sigmoid function: 1 / (1 + exp(-x/scale))"""
    try:
        return 1.0 / (1.0 + math.exp(-x / scale))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def compute_confidence(
    candidate_side: str,  # "UP" | "DOWN" | "NONE"
    fair_fast: Optional[dict] = None,
    fair_smooth: Optional[dict] = None,
    bias: Optional[dict] = None,
    net_edge_bps: Optional[float] = None,
    triggers: Optional[dict] = None,
    quality_mode: str = "OK",
    taapi_alignment: int = 0,  # TAAPI alignment score 0-100
    # Config overrides
    w_net_edge: float = DEFAULT_WEIGHT_NET_EDGE,
    w_agreement: float = DEFAULT_WEIGHT_AGREEMENT,
    w_events: float = DEFAULT_WEIGHT_EVENTS,
    w_quality: float = DEFAULT_WEIGHT_QUALITY,
    low_threshold: float = DEFAULT_LOW_THRESHOLD,
    high_threshold: float = DEFAULT_HIGH_THRESHOLD,
) -> dict:
    """
    Compute confidence score for a trading decision.
    
    Args:
        candidate_side: "UP" (for BUY_UP), "DOWN" (for BUY_DOWN), or "NONE"
        fair_fast: {"up": float, "down": float} from fair model fast
        fair_smooth: {"up": float, "down": float} from fair model smooth
        bias: {"dir": "UP"|"DOWN"|"NEUTRAL", "strength": 0..1}
        net_edge_bps: Net edge in basis points (after spread/buffer)
        triggers: {"up_spike": bool, "down_spike": bool, "up_dip": bool, "down_dip": bool, 
                   "taapi_accel_up": bool, "taapi_accel_down": bool, "taapi_overheated": bool}
        quality_mode: "OK" | "DEGRADED" | "BAD"
        taapi_alignment: TAAPI TF alignment score (0-100), bonus for high alignment
        
    Returns:
        Dictionary with confidence score and breakdown
    """
    reasons = []
    penalties = []
    components = {}
    
    if candidate_side not in ("UP", "DOWN"):
        return {
            "confidence": 0.0,
            "level": "LOW",
            "reasons": [],
            "penalties": ["no_candidate_side"],
            "components": {},
        }
    
    # ========== 1. NET EDGE SCORE ==========
    # Higher net edge -> higher score
    if net_edge_bps is not None and net_edge_bps > 0:
        # Sigmoid: net_edge=0 -> 0.5, net_edge=200 -> ~0.73, net_edge=400 -> ~0.88
        net_edge_score = sigmoid(net_edge_bps, NET_EDGE_SCALE)
        reasons.append(f"net_edge_{int(net_edge_bps)}bps")
    elif net_edge_bps is not None and net_edge_bps < 0:
        net_edge_score = sigmoid(net_edge_bps, NET_EDGE_SCALE)
        penalties.append(f"negative_edge_{int(net_edge_bps)}bps")
    else:
        net_edge_score = 0.5
        penalties.append("no_net_edge")
    
    components["net_edge_score"] = round(net_edge_score, 3)
    
    # ========== 2. AGREEMENT SCORE ==========
    # Check if fair_fast, fair_smooth, and bias agree with candidate_side
    agreement_points = 0.0
    agreement_max = 3.0
    
    # Fair fast
    if fair_fast:
        fair_fast_up = fair_fast.get("up")
        fair_fast_down = fair_fast.get("down")
        if fair_fast_up is not None and fair_fast_down is not None:
            if candidate_side == "UP" and fair_fast_up > 0.5:
                agreement_points += 1.0
                reasons.append("fair_fast_aligned")
            elif candidate_side == "DOWN" and fair_fast_down > 0.5:
                agreement_points += 1.0
                reasons.append("fair_fast_aligned")
            elif candidate_side == "UP" and fair_fast_up < 0.5:
                penalties.append("fair_fast_against")
            elif candidate_side == "DOWN" and fair_fast_down < 0.5:
                penalties.append("fair_fast_against")
    
    # Fair smooth
    if fair_smooth:
        fair_smooth_up = fair_smooth.get("up")
        fair_smooth_down = fair_smooth.get("down")
        if fair_smooth_up is not None and fair_smooth_down is not None:
            if candidate_side == "UP" and fair_smooth_up > 0.5:
                agreement_points += 1.0
                reasons.append("fair_smooth_aligned")
            elif candidate_side == "DOWN" and fair_smooth_down > 0.5:
                agreement_points += 1.0
                reasons.append("fair_smooth_aligned")
            elif candidate_side == "UP" and fair_smooth_up < 0.5:
                penalties.append("fair_smooth_against")
            elif candidate_side == "DOWN" and fair_smooth_down < 0.5:
                penalties.append("fair_smooth_against")
    
    # Bias
    if bias:
        bias_dir = bias.get("dir", "NEUTRAL")
        bias_strength = bias.get("strength", 0.0)
        
        if candidate_side == "UP" and bias_dir == "UP":
            agreement_points += 1.0 * min(1.0, bias_strength + 0.5)
            reasons.append("bias_aligned")
        elif candidate_side == "DOWN" and bias_dir == "DOWN":
            agreement_points += 1.0 * min(1.0, bias_strength + 0.5)
            reasons.append("bias_aligned")
        elif candidate_side == "UP" and bias_dir == "DOWN":
            agreement_points -= 0.5 * bias_strength
            penalties.append("bias_against")
        elif candidate_side == "DOWN" and bias_dir == "UP":
            agreement_points -= 0.5 * bias_strength
            penalties.append("bias_against")
    
    agreement_score = max(0.0, min(1.0, agreement_points / agreement_max))
    components["agreement_score"] = round(agreement_score, 3)
    components["agreement_points"] = round(agreement_points, 2)
    
    # ========== 3. EVENTS SCORE ==========
    # BUY_UP wants: down_spike (price dropped) OR up_dip (PM price dropped)
    # BUY_DOWN wants: up_spike (price rose) OR down_dip (PM price dropped)
    events_score = 0.5  # Neutral
    
    if triggers:
        up_spike = triggers.get("up_spike", False)
        down_spike = triggers.get("down_spike", False)
        up_dip = triggers.get("up_dip", False)
        down_dip = triggers.get("down_dip", False)
        # TAAPI micro stage triggers
        taapi_accel_up = triggers.get("taapi_accel_up", False)
        taapi_accel_down = triggers.get("taapi_accel_down", False)
        taapi_overheated = triggers.get("taapi_overheated", False)
        
        if candidate_side == "UP":
            # Good triggers for BUY_UP
            if down_spike:
                events_score += 0.3
                reasons.append("down_spike_trigger")
            if up_dip:
                events_score += 0.2
                reasons.append("up_dip_trigger")
            # TAAPI acceleration in our direction
            if taapi_accel_up:
                events_score += 0.15
                reasons.append("taapi_accel_up")
            # Bad triggers (price spiked up - we're late)
            if up_spike:
                events_score -= 0.2
                penalties.append("up_spike_late")
            # TAAPI acceleration against us
            if taapi_accel_down:
                events_score -= 0.15
                penalties.append("taapi_accel_down_against")
        
        elif candidate_side == "DOWN":
            # Good triggers for BUY_DOWN
            if up_spike:
                events_score += 0.3
                reasons.append("up_spike_trigger")
            if down_dip:
                events_score += 0.2
                reasons.append("down_dip_trigger")
            # TAAPI acceleration in our direction
            if taapi_accel_down:
                events_score += 0.15
                reasons.append("taapi_accel_down")
            # Bad triggers (price spiked down - we're late)
            if down_spike:
                events_score -= 0.2
                penalties.append("down_spike_late")
            # TAAPI acceleration against us
            if taapi_accel_up:
                events_score -= 0.15
                penalties.append("taapi_accel_up_against")
        
        # TAAPI overheated (overbought/oversold) - general caution
        if taapi_overheated:
            events_score -= 0.1
            penalties.append("taapi_overheated")
        
        # S9: Countertrend triggers - strong signals
        countertrend_down = triggers.get("countertrend_down", False)
        countertrend_up = triggers.get("countertrend_up", False)
        
        if candidate_side == "DOWN" and countertrend_down:
            events_score += 0.25
            reasons.append("countertrend_down")
        elif candidate_side == "UP" and countertrend_up:
            events_score += 0.25
            reasons.append("countertrend_up")
    
    events_score = max(0.0, min(1.0, events_score))
    components["events_score"] = round(events_score, 3)
    
    # ========== 4. QUALITY SCORE ==========
    quality_scores = {
        "OK": 1.0,
        "DEGRADED": 0.6,
        "BAD": 0.2,
    }
    quality_score = quality_scores.get(quality_mode, 0.5)
    
    if quality_mode == "DEGRADED":
        penalties.append("data_degraded")
    elif quality_mode == "BAD":
        penalties.append("data_bad")
    else:
        reasons.append("data_quality_ok")
    
    components["quality_score"] = quality_score
    
    # ========== 5. TAAPI ALIGNMENT BONUS ==========
    # taapi_alignment is 0-100, high values mean TFs agree
    # Add bonus up to +0.1 for perfect alignment (100)
    taapi_bonus = 0.0
    if taapi_alignment >= 70:
        # Strong alignment: +0.05 to +0.1
        taapi_bonus = 0.05 + (taapi_alignment - 70) / 300  # Max +0.1 at 100
        reasons.append(f"taapi_aligned_{taapi_alignment}")
    elif taapi_alignment >= 50:
        # Moderate alignment: small bonus
        taapi_bonus = 0.02
        reasons.append(f"taapi_moderate_{taapi_alignment}")
    elif taapi_alignment < 30:
        # Poor alignment: penalty
        taapi_bonus = -0.05
        penalties.append(f"taapi_misaligned_{taapi_alignment}")
    
    components["taapi_bonus"] = round(taapi_bonus, 3)
    components["taapi_alignment"] = taapi_alignment
    
    # ========== COMBINED SCORE ==========
    confidence = (
        w_net_edge * net_edge_score +
        w_agreement * agreement_score +
        w_events * events_score +
        w_quality * quality_score
    )
    
    # Normalize (weights might not sum to 1)
    total_weight = w_net_edge + w_agreement + w_events + w_quality
    if total_weight > 0:
        confidence = confidence / total_weight
    
    # Apply TAAPI bonus
    confidence += taapi_bonus
    
    confidence = max(0.0, min(1.0, confidence))
    
    # Determine level
    if confidence >= high_threshold:
        level = "HIGH"
    elif confidence >= low_threshold:
        level = "MED"
    else:
        level = "LOW"
    
    return {
        "confidence": round(confidence, 3),
        "level": level,
        "reasons": reasons,
        "penalties": penalties,
        "components": components,
    }


def compute_confidence_from_features(
    candidate_side: str,
    features: dict,
    net_edge_bps: Optional[float] = None,
) -> dict:
    """
    Convenience function to compute confidence from features dict.
    
    Args:
        candidate_side: "UP" or "DOWN"
        features: Full features dict from FeatureEngine.snapshot()
        net_edge_bps: Net edge (can be passed separately or extracted from features)
        
    Returns:
        Confidence dict
    """
    # Extract fair
    fair = features.get("fair", {})
    fair_fast = fair.get("fast")
    fair_smooth = fair.get("smooth")
    
    # Extract bias
    bias = features.get("bias")
    
    # Extract triggers from signals
    signals = features.get("signals", {})
    triggers = {
        "up_spike": signals.get("up_spike_5s", False),
        "down_spike": signals.get("down_spike_5s", False),
        "up_dip": signals.get("up_dip", False),
        "down_dip": signals.get("down_dip", False),
    }
    
    # Extract quality
    quality = features.get("quality", {})
    quality_mode = quality.get("mode", "OK")
    
    # Try to load config weights
    try:
        from collector.config import settings
        w_net_edge = settings.CONF_WEIGHT_NET_EDGE
        w_agreement = settings.CONF_WEIGHT_AGREEMENT
        w_events = settings.CONF_WEIGHT_EVENTS
        w_quality = settings.CONF_WEIGHT_QUALITY
        low_threshold = settings.CONF_LOW_THRESHOLD
        high_threshold = settings.CONF_HIGH_THRESHOLD
    except Exception:
        w_net_edge = DEFAULT_WEIGHT_NET_EDGE
        w_agreement = DEFAULT_WEIGHT_AGREEMENT
        w_events = DEFAULT_WEIGHT_EVENTS
        w_quality = DEFAULT_WEIGHT_QUALITY
        low_threshold = DEFAULT_LOW_THRESHOLD
        high_threshold = DEFAULT_HIGH_THRESHOLD
    
    return compute_confidence(
        candidate_side=candidate_side,
        fair_fast=fair_fast,
        fair_smooth=fair_smooth,
        bias=bias,
        net_edge_bps=net_edge_bps,
        triggers=triggers,
        quality_mode=quality_mode,
        w_net_edge=w_net_edge,
        w_agreement=w_agreement,
        w_events=w_events,
        w_quality=w_quality,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )
