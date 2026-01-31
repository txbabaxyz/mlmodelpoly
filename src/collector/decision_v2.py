"""
Decision Engine v2 (S6) + TAAPI Integration
============================================

Two-layer recommendation engine:
1. Compute net_edge for FAST and SMOOTH fair values
2. Determine candidate_side by max(net_edge_fast)
3. Compute confidence using S5 confidence engine (with TAAPI alignment boost)
4. Final action: HIGH → ACCUMULATE, MED → WATCH, LOW → WAIT

TAAPI Integration:
- Uses alignment_score for confidence boost
- Checks taapi_ready for veto
- Applies regime (TREND/RANGE) adjustments
- Includes micro_stage signals

Output:
    - candidate_side: "UP" | "DOWN" | "NONE"
    - action: "WAIT" | "WATCH_UP" | "WATCH_DOWN" | "ACCUMULATE_UP" | "ACCUMULATE_DOWN"
    - confidence: 0..1
    - confidence_level: "LOW" | "MED" | "HIGH"
    - net_edge_fast/smooth: {up, down} in bps
    - veto and reasons
    - taapi_alignment, taapi_regime, taapi_micro_stage

Usage:
    from collector.decision_v2 import compute_decision_v2
    
    result = compute_decision_v2(features, pm_up_down)
"""

import logging
from typing import Optional

from collector.confidence import compute_confidence

logger = logging.getLogger(__name__)


def compute_decision_v2(
    features: dict,
    pm_up_down: Optional[dict] = None,
    edge_buffer_bps: float = 25.0,
) -> dict:
    """
    Compute decision v2 with fast/smooth net edge and confidence.
    
    Args:
        features: Full features snapshot
        pm_up_down: Polymarket UP/DOWN book data
        edge_buffer_bps: Buffer to add to required edge
    
    Returns:
        Dictionary with decision details
    """
    # Extract fair
    fair = features.get("fair", {})
    fair_fast = fair.get("fast", {})
    fair_smooth = fair.get("smooth", {})
    fair_status = fair.get("status", {})
    
    # Extract bias (S4 - now includes TAAPI integration)
    bias = features.get("bias", {})
    
    # Extract quality
    quality = features.get("quality", {})
    quality_mode = quality.get("mode", "OK")
    trade_mode = quality.get("trade_mode", "OK")
    taapi_ready = quality.get("taapi_ready", False)
    taapi_stale = quality.get("taapi_stale", False)
    
    # Extract TAAPI context directly for additional signals
    taapi_context = features.get("taapi_context", {}) or {}
    taapi_alignment = taapi_context.get("alignment_score", 0) or 0  # 0-100
    taapi_regime = taapi_context.get("regime_15m", "RANGE")
    taapi_micro_stage = taapi_context.get("micro_stage", "NEUTRAL")
    taapi_rsi_1m = taapi_context.get("rsi_1m")
    
    # Extract signals for triggers
    signals = features.get("signals", {})
    spikes = features.get("spikes", {})
    pm_dips = features.get("pm_dips", {})
    
    # S9: Extract ROC and countertrend signal
    roc = features.get("roc", {}) or {}
    roc_30s = roc.get("roc_30s")
    roc_direction = roc.get("direction", "UNKNOWN")
    roc_ready = roc.get("ready", False)
    
    countertrend = features.get("countertrend", {}) or {}
    countertrend_signal = countertrend.get("signal", "NONE")
    
    # Get PM data
    if pm_up_down is None:
        pm_up_down = features.get("polymarket_up_down", {})
    
    up_book = pm_up_down.get("up", {}) if pm_up_down else {}
    down_book = pm_up_down.get("down", {}) if pm_up_down else {}
    
    # Extract PM prices
    up_mid = up_book.get("mid")
    down_mid = down_book.get("mid")
    up_ask = up_book.get("best_ask")
    down_ask = down_book.get("best_ask")
    up_required_edge = up_book.get("required_edge_bps", 0) or 0
    down_required_edge = down_book.get("required_edge_bps", 0) or 0
    up_slice_ok = up_book.get("slice_ok", False)
    down_slice_ok = down_book.get("slice_ok", False)
    
    # ========================================
    # 1. COMPUTE NET EDGE FAST
    # ========================================
    fast_up = fair_fast.get("up")
    fast_down = fair_fast.get("down")
    
    net_edge_fast_up = None
    net_edge_fast_down = None
    
    if fast_up is not None and up_mid is not None:
        # edge = (fair - market) * 10000 bps
        edge_up_bps = (fast_up - up_mid) * 10000
        req_up = up_required_edge + edge_buffer_bps
        net_edge_fast_up = edge_up_bps - req_up
    
    if fast_down is not None and down_mid is not None:
        edge_down_bps = (fast_down - down_mid) * 10000
        req_down = down_required_edge + edge_buffer_bps
        net_edge_fast_down = edge_down_bps - req_down
    
    # ========================================
    # 2. COMPUTE NET EDGE SMOOTH
    # ========================================
    smooth_up = fair_smooth.get("up")
    smooth_down = fair_smooth.get("down")
    
    net_edge_smooth_up = None
    net_edge_smooth_down = None
    
    if smooth_up is not None and up_mid is not None:
        edge_up_bps = (smooth_up - up_mid) * 10000
        req_up = up_required_edge + edge_buffer_bps
        net_edge_smooth_up = edge_up_bps - req_up
    
    if smooth_down is not None and down_mid is not None:
        edge_down_bps = (smooth_down - down_mid) * 10000
        req_down = down_required_edge + edge_buffer_bps
        net_edge_smooth_down = edge_down_bps - req_down
    
    # ========================================
    # 3. DETERMINE CANDIDATE SIDE
    # ========================================
    candidate_side = "NONE"
    veto = False
    veto_reasons = []
    countertrend_used = False
    
    # Check for BAD quality -> veto
    if trade_mode == "BAD" or quality_mode == "BAD":
        veto = True
        veto_reasons.append("quality_bad")
    
    # Check fair status
    if not fair_status.get("fast_ready", False):
        veto = True
        reason = fair_status.get("fast_reason", "fair_not_ready")
        veto_reasons.append(f"fair_fast:{reason}")
    
    # No PM prices
    if up_mid is None or down_mid is None:
        veto = True
        veto_reasons.append("no_pm_prices")
    
    # TAAPI checks (soft veto - add to reasons but don't block completely)
    if not taapi_ready:
        veto_reasons.append("taapi_not_ready")
    elif taapi_stale:
        veto_reasons.append("taapi_stale")
    
    if not veto:
        # ========================================
        # S9: COUNTERTREND OVERRIDE
        # ========================================
        # Countertrend signal can override net_edge logic
        # This catches momentum reversals that fair model misses
        
        countertrend_used = False
        
        if countertrend_signal == "BUY_DOWN" and down_slice_ok:
            # Z > 0 (BTC above ref) but falling → buy DOWN
            candidate_side = "DOWN"
            countertrend_used = True
            veto_reasons.append("countertrend_down")
        
        elif countertrend_signal == "BUY_UP" and up_slice_ok:
            # Z < 0 (BTC below ref) but rising → buy UP
            candidate_side = "UP"
            countertrend_used = True
            veto_reasons.append("countertrend_up")
        
        else:
            # Standard logic: select side with max positive net_edge_fast
            up_candidate = net_edge_fast_up is not None and net_edge_fast_up > 0 and up_slice_ok
            down_candidate = net_edge_fast_down is not None and net_edge_fast_down > 0 and down_slice_ok
            
            if up_candidate and down_candidate:
                # Both positive, pick larger
                if net_edge_fast_up >= net_edge_fast_down:
                    candidate_side = "UP"
                else:
                    candidate_side = "DOWN"
            elif up_candidate:
                candidate_side = "UP"
            elif down_candidate:
                candidate_side = "DOWN"
            else:
                candidate_side = "NONE"
    
    # ========================================
    # 4. COMPUTE CONFIDENCE (with TAAPI boost)
    # ========================================
    # Build triggers dict (including TAAPI signals and countertrend)
    triggers = {
        "up_spike": spikes.get("up_spike_5s", False) or signals.get("up_spike_5s", False),
        "down_spike": spikes.get("down_spike_5s", False) or signals.get("down_spike_5s", False),
        "up_dip": pm_dips.get("up_dip", False),
        "down_dip": pm_dips.get("down_dip", False),
        # TAAPI micro stage triggers
        "taapi_accel_up": taapi_micro_stage == "ACCEL_UP",
        "taapi_accel_down": taapi_micro_stage == "ACCEL_DOWN",
        "taapi_overheated": taapi_micro_stage == "OVERHEATED",
        # S9: Countertrend triggers
        "countertrend_down": countertrend_signal == "BUY_DOWN",
        "countertrend_up": countertrend_signal == "BUY_UP",
    }
    
    # Net edge for confidence
    net_edge_bps = None
    if candidate_side == "UP" and net_edge_fast_up is not None:
        net_edge_bps = net_edge_fast_up
    elif candidate_side == "DOWN" and net_edge_fast_down is not None:
        net_edge_bps = net_edge_fast_down
    
    conf_result = compute_confidence(
        candidate_side=candidate_side if candidate_side != "NONE" else "UP",  # dummy if NONE
        fair_fast=fair_fast,
        fair_smooth=fair_smooth,
        bias=bias,
        net_edge_bps=net_edge_bps,
        triggers=triggers,
        quality_mode=quality_mode,
        taapi_alignment=taapi_alignment,  # Pass TAAPI alignment for boost
    )
    
    confidence = conf_result.get("confidence", 0) if candidate_side != "NONE" else 0
    confidence_level = conf_result.get("level", "LOW") if candidate_side != "NONE" else "LOW"
    confidence_reasons = conf_result.get("reasons", [])
    confidence_penalties = conf_result.get("penalties", [])
    
    # ========================================
    # 5. DETERMINE ACTION
    # ========================================
    if veto:
        action = "WAIT"
    elif candidate_side == "NONE":
        action = "WAIT"
    elif confidence_level == "HIGH":
        action = f"ACCUMULATE_{candidate_side}"
    elif confidence_level in ("MED", "MEDIUM"):
        action = f"WATCH_{candidate_side}"
    else:
        action = "WAIT"
    
    # ========================================
    # 6. BUILD RESULT
    # ========================================
    def safe_round(x, decimals=1):
        return round(x, decimals) if x is not None else None
    
    result = {
        "candidate_side": candidate_side,
        "action": action,
        "confidence": round(confidence, 3),
        "confidence_level": confidence_level,
        "confidence_reasons": confidence_reasons,
        "confidence_penalties": confidence_penalties,
        "confidence_components": conf_result.get("components", {}),
        
        "net_edge_fast": {
            "up": safe_round(net_edge_fast_up),
            "down": safe_round(net_edge_fast_down),
        },
        "net_edge_smooth": {
            "up": safe_round(net_edge_smooth_up),
            "down": safe_round(net_edge_smooth_down),
        },
        "required_edge_bps": {
            "up": safe_round(up_required_edge + edge_buffer_bps),
            "down": safe_round(down_required_edge + edge_buffer_bps),
        },
        
        "fair_fast": {
            "up": fair_fast.get("up"),
            "down": fair_fast.get("down"),
        },
        "fair_smooth": {
            "up": fair_smooth.get("up"),
            "down": fair_smooth.get("down"),
        },
        "pm_mid": {
            "up": up_mid,
            "down": down_mid,
        },
        "pm_ask": {
            "up": up_ask,
            "down": down_ask,
        },
        "slice_ok": {
            "up": up_slice_ok,
            "down": down_slice_ok,
        },
        
        "trade_mode": trade_mode,
        "quality_mode": quality_mode,
        "veto": veto,
        "veto_reasons": veto_reasons if veto_reasons else None,
        
        "triggers": triggers,
        "bias_dir": bias.get("dir"),
        "bias_strength": bias.get("strength"),
        "bias_taapi_integrated": bias.get("taapi_integrated", False),
        
        # TAAPI context info
        "taapi_ready": taapi_ready,
        "taapi_alignment": taapi_alignment,
        "taapi_regime": taapi_regime,
        "taapi_micro_stage": taapi_micro_stage,
        "taapi_rsi_1m": taapi_rsi_1m,
        
        # S9: ROC and countertrend info
        "roc_30s": roc_30s,
        "roc_direction": roc_direction,
        "roc_ready": roc_ready,
        "countertrend_signal": countertrend_signal,
        "countertrend_used": countertrend_used,
    }
    
    return result
