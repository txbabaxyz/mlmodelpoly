"""
Accumulate Engine
=================

Final decision engine for UP/DOWN accumulation strategy.

Why this exists:
    Combines all signals to produce actionable recommendations:
    - Fair value vs market price → edge
    - Spread/depth → execution feasibility
    - Spikes/dips → timing signals
    - Budget/cooldown → risk management

Strategy Logic:
    1. Compute edge: net_edge = (fair - market_mid) * 10000 - required_edge
    2. Check trade_mode: if BAD → WAIT
    3. Check slice_ok: if False → WAIT
    4. Check triggers:
       - For UP: need down_spike_5s OR up_dip (price dropped → buy UP)
       - For DOWN: need up_spike_5s OR down_dip (price rose → buy DOWN)
    5. Check cooldown and budget
    6. Return ACCUMULATE_UP, ACCUMULATE_DOWN, or WAIT

Usage:
    engine = AccumulateEngine(
        cooldown_sec=2.0,
        max_slices_per_window=30,
        max_usd_per_window=300.0,
        slice_usd=20.0
    )
    
    decision = engine.decide(features)
    # decision = {
    #     "action": "ACCUMULATE_UP",
    #     "net_edge_up": 150.0,
    #     "reasons": ["positive_edge", "down_spike_trigger"]
    # }

Configuration:
    MIN_NET_EDGE_BPS: Minimum net edge to consider (default 0)
    Inherits from config: COOLDOWN_SEC, MAX_SLICES_PER_WINDOW, etc.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, Any

from collector.utils_time import now_ms
from collector.fair_model import compute_fair_updown, compute_edge_bps

logger = logging.getLogger(__name__)

# Minimum net edge to consider (after spread cost)
MIN_NET_EDGE_BPS = 0.0

# Sanity check thresholds
SPIKE_SANITY_BPS = 5.0  # If ret > 5 bps and down_spike, something is wrong


def validate_signs(features: dict, decision: "AccumulateDecision") -> list[str]:
    """
    Validate signal signs and decision alignment (CAL1).
    
    Checks:
    1. Spike sign sanity (ret_5s vs spike flag)
    2. Dip sign sanity (should be <= 0 when price below max)
    3. Decision-trigger alignment
    4. Decision-edge alignment
    
    Args:
        features: Full features dict
        decision: AccumulateDecision instance
    
    Returns:
        List of warning strings (empty = all OK)
    """
    warnings = []
    
    spikes = features.get("spikes") or {}
    pm_dips = features.get("pm_dips") or {}
    
    ret_5s_bps = spikes.get("ret_5s_bps")
    down_spike = spikes.get("down_spike_5s", False)
    up_spike = spikes.get("up_spike_5s", False)
    
    up_dip_bps = pm_dips.get("up_dip_bps")
    down_dip_bps = pm_dips.get("down_dip_bps")
    up_dip = pm_dips.get("up_dip", False)
    down_dip = pm_dips.get("down_dip", False)
    
    # (1) Spike sanity: down_spike should occur when ret is NEGATIVE
    if ret_5s_bps is not None:
        if ret_5s_bps > SPIKE_SANITY_BPS and down_spike:
            warnings.append(f"spike_sign_inversion:down_spike_with_positive_ret({ret_5s_bps:.1f})")
        if ret_5s_bps < -SPIKE_SANITY_BPS and up_spike:
            warnings.append(f"spike_sign_inversion:up_spike_with_negative_ret({ret_5s_bps:.1f})")
    
    # (2) Dip sanity: dip_bps should be <= 0 when price < max_60s
    # If dip flag is True but dip_bps > 0, that's wrong
    if up_dip and up_dip_bps is not None and up_dip_bps > 0:
        warnings.append(f"dip_sign_inversion:up_dip_positive({up_dip_bps:.1f})")
    if down_dip and down_dip_bps is not None and down_dip_bps > 0:
        warnings.append(f"dip_sign_inversion:down_dip_positive({down_dip_bps:.1f})")
    
    # (3) Decision-trigger alignment
    action = decision.action
    
    if action == "ACCUMULATE_UP":
        # Expected triggers: down_spike OR up_dip
        if not (down_spike or up_dip):
            warnings.append("decision_trigger_mismatch:BUY_UP_without_down_spike_or_up_dip")
    
    if action == "ACCUMULATE_DOWN":
        # Expected triggers: up_spike OR down_dip
        if not (up_spike or down_dip):
            warnings.append("decision_trigger_mismatch:BUY_DOWN_without_up_spike_or_down_dip")
    
    # (4) Decision-edge alignment
    if action == "ACCUMULATE_UP":
        if decision.net_edge_up is None or decision.net_edge_up <= 0:
            warnings.append(f"decision_net_edge_mismatch:BUY_UP_with_net_edge={decision.net_edge_up}")
    
    if action == "ACCUMULATE_DOWN":
        if decision.net_edge_down is None or decision.net_edge_down <= 0:
            warnings.append(f"decision_net_edge_mismatch:BUY_DOWN_with_net_edge={decision.net_edge_down}")
    
    return warnings


@dataclass
class AccumulateDecision:
    """
    Result of accumulate decision.
    
    Attributes:
        ts_ms: Timestamp of decision
        action: "ACCUMULATE_UP" | "ACCUMULATE_DOWN" | "WAIT"
        net_edge_up: Net edge for UP side (fair - market - spread cost)
        net_edge_down: Net edge for DOWN side
        reasons: List of reasons for the decision
        veto_reasons: List of reasons preventing action
    """
    ts_ms: int
    action: str  # "ACCUMULATE_UP" | "ACCUMULATE_DOWN" | "WAIT"
    
    # Edge computation
    fair_up: Optional[float] = None
    fair_down: Optional[float] = None
    market_up: Optional[float] = None
    market_down: Optional[float] = None
    edge_up_bps: Optional[float] = None
    edge_down_bps: Optional[float] = None
    required_edge_up: Optional[float] = None
    required_edge_down: Optional[float] = None
    net_edge_up: Optional[float] = None
    net_edge_down: Optional[float] = None
    
    # Triggers
    down_spike: bool = False
    up_spike: bool = False
    up_dip: bool = False
    down_dip: bool = False
    
    # Status
    trade_mode: str = "UNKNOWN"
    slice_ok_up: bool = False
    slice_ok_down: bool = False
    
    # Budget/cooldown
    cooldown_active: bool = False
    budget_exhausted: bool = False
    slices_this_window: int = 0
    usd_this_window: float = 0.0
    
    reasons: list[str] = field(default_factory=list)
    veto_reasons: list[str] = field(default_factory=list)
    
    # CAL1: Sanity checks
    sanity_warnings: list[str] = field(default_factory=list)
    
    # CAL1: Raw values for debugging ("why panel")
    raw_ret_5s_bps: Optional[float] = None
    raw_z_ret_5s: Optional[float] = None
    raw_pm_up_mid: Optional[float] = None
    raw_pm_down_mid: Optional[float] = None
    raw_pm_up_max_60s: Optional[float] = None
    raw_pm_down_max_60s: Optional[float] = None
    raw_pm_up_dip_bps: Optional[float] = None
    raw_pm_down_dip_bps: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "ts_ms": self.ts_ms,
            "action": self.action,
            "fair_up": self.fair_up,
            "fair_down": self.fair_down,
            "market_up": self.market_up,
            "market_down": self.market_down,
            "edge_up_bps": round(self.edge_up_bps, 1) if self.edge_up_bps else None,
            "edge_down_bps": round(self.edge_down_bps, 1) if self.edge_down_bps else None,
            "required_edge_up": round(self.required_edge_up, 1) if self.required_edge_up else None,
            "required_edge_down": round(self.required_edge_down, 1) if self.required_edge_down else None,
            "net_edge_up": round(self.net_edge_up, 1) if self.net_edge_up else None,
            "net_edge_down": round(self.net_edge_down, 1) if self.net_edge_down else None,
            "triggers": {
                "down_spike": self.down_spike,
                "up_spike": self.up_spike,
                "up_dip": self.up_dip,
                "down_dip": self.down_dip,
            },
            "trade_mode": self.trade_mode,
            "slice_ok_up": self.slice_ok_up,
            "slice_ok_down": self.slice_ok_down,
            "cooldown_active": self.cooldown_active,
            "budget_exhausted": self.budget_exhausted,
            "slices_this_window": self.slices_this_window,
            "usd_this_window": round(self.usd_this_window, 2),
            "reasons": self.reasons,
            "veto_reasons": self.veto_reasons if self.veto_reasons else None,
            # CAL1: Sanity warnings
            "sanity_warnings": self.sanity_warnings if self.sanity_warnings else None,
            # CAL1: Raw values for debugging
            "raw": {
                "ret_5s_bps": round(self.raw_ret_5s_bps, 2) if self.raw_ret_5s_bps is not None else None,
                "z_ret_5s": round(self.raw_z_ret_5s, 2) if self.raw_z_ret_5s is not None else None,
                "pm_up_mid": round(self.raw_pm_up_mid, 4) if self.raw_pm_up_mid is not None else None,
                "pm_down_mid": round(self.raw_pm_down_mid, 4) if self.raw_pm_down_mid is not None else None,
                "pm_up_max_60s": round(self.raw_pm_up_max_60s, 4) if self.raw_pm_up_max_60s is not None else None,
                "pm_down_max_60s": round(self.raw_pm_down_max_60s, 4) if self.raw_pm_down_max_60s is not None else None,
                "pm_up_dip_bps": round(self.raw_pm_up_dip_bps, 1) if self.raw_pm_up_dip_bps is not None else None,
                "pm_down_dip_bps": round(self.raw_pm_down_dip_bps, 1) if self.raw_pm_down_dip_bps is not None else None,
            },
        }


class AccumulateEngine:
    """
    Accumulation decision engine.
    
    Combines fair value, market prices, triggers, and risk limits
    to produce BUY UP / BUY DOWN / WAIT recommendations.
    """
    
    def __init__(
        self,
        cooldown_sec: float = 2.0,
        max_slices_per_window: int = 30,
        max_usd_per_window: float = 300.0,
        slice_usd: float = 20.0,
        window_len_sec: int = 900,
    ):
        """
        Initialize accumulate engine.
        
        Args:
            cooldown_sec: Seconds to wait between executions
            max_slices_per_window: Maximum slices per 15-min window
            max_usd_per_window: Maximum USD per 15-min window
            slice_usd: Size of each slice in USD
            window_len_sec: Window length for budget reset
        """
        self.cooldown_sec = cooldown_sec
        self.max_slices_per_window = max_slices_per_window
        self.max_usd_per_window = max_usd_per_window
        self.slice_usd = slice_usd
        self.window_len_sec = window_len_sec
        
        # State
        self._last_execution_ms: Optional[int] = None
        self._current_window_id: Optional[int] = None
        self._slices_this_window: int = 0
        self._usd_this_window: float = 0.0
        
        self._last_decision: Optional[AccumulateDecision] = None
        
        # Event recorder for backtesting
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from collector.event_recorder import EventRecorder
        self._event_recorder: Optional["EventRecorder"] = None
        
        logger.info(
            "accumulate_engine_initialized",
            extra={
                "cooldown_sec": cooldown_sec,
                "max_slices_per_window": max_slices_per_window,
                "max_usd_per_window": max_usd_per_window,
                "slice_usd": slice_usd,
            },
        )
    
    def set_event_recorder(self, recorder: "EventRecorder") -> None:
        """Set the EventRecorder for high-frequency logging."""
        self._event_recorder = recorder
        logger.info("accumulate_engine_event_recorder_set")
    
    def decide(self, features: dict) -> AccumulateDecision:
        """
        Make accumulation decision based on features.
        
        Args:
            features: Full features snapshot
        
        Returns:
            AccumulateDecision with action and reasoning
        """
        ts = now_ms()
        reasons: list[str] = []
        veto_reasons: list[str] = []
        
        # Check window rollover for budget
        self._check_window_rollover(ts)
        
        # ========== EXTRACT DATA ==========
        
        quality = features.get("quality") or {}
        trade_mode = quality.get("trade_mode", "BAD")
        
        market_ref = features.get("market_ref") or {}
        ref_px = market_ref.get("ref_px")
        s_now = market_ref.get("S_now")
        tau_sec = market_ref.get("tau_sec")
        
        vol = features.get("vol") or {}
        sigma_15m = vol.get("sigma_15m")
        
        pm_updown = features.get("polymarket_up_down") or {}
        up = pm_updown.get("up") or {}
        down = pm_updown.get("down") or {}
        
        market_up = up.get("mid")
        market_down = down.get("mid")
        slice_ok_up = up.get("slice_ok", False)
        slice_ok_down = down.get("slice_ok", False)
        required_edge_up = up.get("required_edge_bps")
        required_edge_down = down.get("required_edge_bps")
        
        spikes = features.get("spikes") or {}
        down_spike = spikes.get("down_spike_5s", False)
        up_spike = spikes.get("up_spike_5s", False)
        
        pm_dips = features.get("pm_dips") or {}
        up_dip = pm_dips.get("up_dip", False)
        down_dip = pm_dips.get("down_dip", False)
        
        # ========== COMPUTE FAIR VALUE ==========
        
        fair_up = None
        fair_down = None
        
        if s_now and ref_px and sigma_15m and tau_sec:
            fair_result = compute_fair_updown(
                s_now=s_now,
                ref_px=ref_px,
                sigma_15m=sigma_15m,
                tau_sec=tau_sec,
                window_sec=900.0,
            )
            fair_up = fair_result.get("fair_up")
            fair_down = fair_result.get("fair_down")
        
        # ========== COMPUTE EDGE ==========
        
        edge_up_bps = compute_edge_bps(fair_up, market_up)
        edge_down_bps = compute_edge_bps(fair_down, market_down)
        
        # Net edge = edge - required (spread cost)
        net_edge_up = None
        net_edge_down = None
        
        if edge_up_bps is not None and required_edge_up is not None:
            net_edge_up = edge_up_bps - required_edge_up
        
        if edge_down_bps is not None and required_edge_down is not None:
            net_edge_down = edge_down_bps - required_edge_down
        
        # ========== CHECK VETO CONDITIONS ==========
        
        # Trade mode
        if trade_mode == "BAD":
            veto_reasons.append("trade_mode_bad")
        
        # Cooldown
        cooldown_active = False
        if self._last_execution_ms:
            elapsed_sec = (ts - self._last_execution_ms) / 1000.0
            if elapsed_sec < self.cooldown_sec:
                cooldown_active = True
                veto_reasons.append("cooldown_active")
        
        # Budget
        budget_exhausted = False
        if self._slices_this_window >= self.max_slices_per_window:
            budget_exhausted = True
            veto_reasons.append("max_slices_reached")
        if self._usd_this_window >= self.max_usd_per_window:
            budget_exhausted = True
            veto_reasons.append("max_usd_reached")
        
        # ========== DETERMINE ACTION ==========
        
        action = "WAIT"
        
        # Skip if vetoed
        if veto_reasons:
            pass
        else:
            # Check UP accumulation
            up_trigger = down_spike or up_dip  # BTC dropped OR UP token dipped
            if (net_edge_up is not None and 
                net_edge_up > MIN_NET_EDGE_BPS and 
                slice_ok_up and 
                up_trigger):
                action = "ACCUMULATE_UP"
                reasons.append("positive_net_edge_up")
                if down_spike:
                    reasons.append("down_spike_trigger")
                if up_dip:
                    reasons.append("up_dip_trigger")
            
            # Check DOWN accumulation (only if UP not selected)
            elif (net_edge_down is not None and
                  net_edge_down > MIN_NET_EDGE_BPS and
                  slice_ok_down):
                down_trigger = up_spike or down_dip  # BTC rose OR DOWN token dipped
                if down_trigger:
                    action = "ACCUMULATE_DOWN"
                    reasons.append("positive_net_edge_down")
                    if up_spike:
                        reasons.append("up_spike_trigger")
                    if down_dip:
                        reasons.append("down_dip_trigger")
        
        # If no action, add reason with detail
        fair_status = features.get("fair_status") or {}
        fair_status_reason = fair_status.get("reason", "unknown")
        
        if action == "WAIT" and not veto_reasons:
            if net_edge_up is None or net_edge_down is None:
                # Add detailed reason from fair_status
                veto_reasons.append(f"fair_not_ready:{fair_status_reason}")
            elif net_edge_up <= MIN_NET_EDGE_BPS and net_edge_down <= MIN_NET_EDGE_BPS:
                veto_reasons.append("no_positive_edge")
            elif not (down_spike or up_dip or up_spike or down_dip):
                veto_reasons.append("no_trigger")
            elif not slice_ok_up and not slice_ok_down:
                veto_reasons.append("no_slice_ok")
        
        # ========== BUILD DECISION ==========
        
        # CAL1: Extract raw values for debugging
        raw_ret_5s_bps = spikes.get("ret_5s_bps")
        raw_z_ret_5s = spikes.get("z_ret_5s")
        raw_pm_up_mid = pm_dips.get("up_mid")
        raw_pm_down_mid = pm_dips.get("down_mid")
        raw_pm_up_max_60s = pm_dips.get("up_max_60s")
        raw_pm_down_max_60s = pm_dips.get("down_max_60s")
        raw_pm_up_dip_bps = pm_dips.get("up_dip_bps")
        raw_pm_down_dip_bps = pm_dips.get("down_dip_bps")
        
        decision = AccumulateDecision(
            ts_ms=ts,
            action=action,
            fair_up=fair_up,
            fair_down=fair_down,
            market_up=market_up,
            market_down=market_down,
            edge_up_bps=edge_up_bps,
            edge_down_bps=edge_down_bps,
            required_edge_up=required_edge_up,
            required_edge_down=required_edge_down,
            net_edge_up=net_edge_up,
            net_edge_down=net_edge_down,
            down_spike=down_spike,
            up_spike=up_spike,
            up_dip=up_dip,
            down_dip=down_dip,
            trade_mode=trade_mode,
            slice_ok_up=slice_ok_up,
            slice_ok_down=slice_ok_down,
            cooldown_active=cooldown_active,
            budget_exhausted=budget_exhausted,
            slices_this_window=self._slices_this_window,
            usd_this_window=self._usd_this_window,
            reasons=reasons,
            veto_reasons=veto_reasons,
            # CAL1: Raw values for debugging
            raw_ret_5s_bps=raw_ret_5s_bps,
            raw_z_ret_5s=raw_z_ret_5s,
            raw_pm_up_mid=raw_pm_up_mid,
            raw_pm_down_mid=raw_pm_down_mid,
            raw_pm_up_max_60s=raw_pm_up_max_60s,
            raw_pm_down_max_60s=raw_pm_down_max_60s,
            raw_pm_up_dip_bps=raw_pm_up_dip_bps,
            raw_pm_down_dip_bps=raw_pm_down_dip_bps,
        )
        
        # CAL1: Run sanity checks
        sanity_warnings = validate_signs(features, decision)
        decision.sanity_warnings = sanity_warnings
        
        # Log warnings if any
        if sanity_warnings:
            logger.warning(
                "accumulate_sanity_warnings",
                extra={
                    "action": action,
                    "warnings": sanity_warnings,
                    "ret_5s_bps": raw_ret_5s_bps,
                    "z_ret_5s": raw_z_ret_5s,
                    "up_dip_bps": raw_pm_up_dip_bps,
                    "down_dip_bps": raw_pm_down_dip_bps,
                },
            )
        
        self._last_decision = decision
        
        # Record decision for backtesting
        if self._event_recorder:
            self._event_recorder.record_decision(decision.to_dict())
        
        return decision
    
    def record_execution(self, side: str, usd_amount: float) -> None:
        """
        Record that an execution happened.
        
        Args:
            side: "UP" or "DOWN"
            usd_amount: Amount in USD
        """
        ts = now_ms()
        self._last_execution_ms = ts
        self._slices_this_window += 1
        self._usd_this_window += usd_amount
        
        logger.info(
            "accumulate_execution_recorded",
            extra={
                "side": side,
                "usd_amount": usd_amount,
                "slices_this_window": self._slices_this_window,
                "usd_this_window": self._usd_this_window,
            },
        )
    
    def _check_window_rollover(self, ts: int) -> None:
        """Check if we need to reset budget for new window."""
        window_id = ts // (self.window_len_sec * 1000)
        
        if self._current_window_id is None:
            self._current_window_id = window_id
        elif window_id != self._current_window_id:
            # New window - reset budget
            logger.info(
                "accumulate_window_rolled",
                extra={
                    "old_window": self._current_window_id,
                    "new_window": window_id,
                    "slices_last_window": self._slices_this_window,
                    "usd_last_window": self._usd_this_window,
                },
            )
            self._current_window_id = window_id
            self._slices_this_window = 0
            self._usd_this_window = 0.0
    
    def get_last_decision(self) -> Optional[AccumulateDecision]:
        """Get the last computed decision."""
        return self._last_decision
    
    def get_budget_status(self) -> dict:
        """Get current budget status."""
        return {
            "slices_this_window": self._slices_this_window,
            "max_slices": self.max_slices_per_window,
            "usd_this_window": self._usd_this_window,
            "max_usd": self.max_usd_per_window,
            "cooldown_sec": self.cooldown_sec,
            "last_execution_ms": self._last_execution_ms,
        }
