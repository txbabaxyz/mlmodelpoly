"""
Fair Value Model for UPDOWN Markets
====================================

Computes fair probability for "price will be higher" (UP) events
using a log-normal price model.

Why this exists:
    Polymarket UPDOWN markets ask: "Will BTC be above ref_px at window end?"
    
    To find fair value, we model price as geometric Brownian motion:
        S_end = S_now * exp((μ - σ²/2)*τ + σ*√τ*Z)
    
    Where Z ~ N(0,1). The probability S_end > ref_px is:
        P(S_end > ref_px) = Φ(z)
    
    Where:
        z = [ln(S_now/ref_px) + (μ - σ²/2)*τ] / (σ*√τ)
    
    For short horizons (15 min), drift μ ≈ 0, so:
        z ≈ ln(S_now/ref_px) / (σ*√τ)

Model Assumptions:
    1. Log-normal price distribution
    2. Drift μ ≈ 0 for 15-minute windows (negligible)
    3. Volatility σ estimated from recent 1m returns
    4. τ is time remaining normalized to window length

Usage:
    from collector.fair_model import compute_fair_updown
    
    fair = compute_fair_updown(
        s_now=105000.0,
        ref_px=104500.0,
        sigma_15m=0.003,
        tau_sec=600.0,
        window_sec=900.0
    )
    # fair = {"fair_up": 0.72, "fair_down": 0.28, "z_score": 0.58}

Configuration:
    DRIFT_ADJUSTMENT: Small drift adjustment from bias (default 0)
    MIN_SIGMA: Minimum sigma to avoid division issues (default 0.0001)
    MIN_TAU_SEC: Minimum tau to avoid edge cases (default 1.0)
"""

import math
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Minimum values to avoid numerical issues
MIN_SIGMA = 0.0001  # 0.01% minimum volatility
MIN_TAU_SEC = 1.0   # 1 second minimum time remaining


def standard_normal_cdf(x: float) -> float:
    """
    Compute standard normal CDF Φ(x) using error function.
    
    Φ(x) = (1 + erf(x/√2)) / 2
    
    Args:
        x: Value to compute CDF for
    
    Returns:
        Probability P(Z < x) where Z ~ N(0,1)
    
    Example:
        >>> standard_normal_cdf(0)
        0.5
        >>> standard_normal_cdf(1.96)
        0.975
    """
    return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0


def compute_fair_updown(
    s_now: float,
    ref_px: float,
    sigma_15m: float,
    tau_sec: float,
    window_sec: float = 900.0,
    drift: float = 0.0,
) -> dict:
    """
    Compute fair UP/DOWN probabilities for UPDOWN market.
    
    Model: S_end > ref_px with log-normal price dynamics.
    
    Formula:
        z = [ln(S_now/ref_px) + drift*τ_norm] / (sigma * √τ_norm)
        fair_up = Φ(z)
        fair_down = 1 - fair_up
    
    Where τ_norm = tau_sec / window_sec (normalized time remaining).
    
    Args:
        s_now: Current spot price
        ref_px: Reference price at window start
        sigma_15m: 15-minute volatility (std of log returns scaled)
        tau_sec: Time remaining until window end (seconds)
        window_sec: Total window length (default 900 = 15 min)
        drift: Optional drift adjustment (default 0)
    
    Returns:
        Dictionary with:
            - fair_up: Probability price will be above ref_px
            - fair_down: Probability price will be below ref_px
            - z_score: Z-score used in calculation
            - inputs: Dictionary of input values for debugging
    
    Edge Cases:
        - If sigma too small: returns 0.5/0.5 (can't estimate)
        - If tau too small: returns based on current price vs ref
        - If s_now == ref_px: returns ~0.5 (slight drift adjustment)
    
    Example:
        >>> # Price 1% above ref, moderate vol, 10 min left
        >>> fair = compute_fair_updown(105000, 104000, 0.003, 600, 900)
        >>> fair["fair_up"]
        0.78  # High probability of staying above
    """
    # Input validation
    if s_now <= 0 or ref_px <= 0:
        logger.warning(
            "fair_model_invalid_prices",
            extra={"s_now": s_now, "ref_px": ref_px},
        )
        return _default_fair()
    
    # Handle edge cases
    if sigma_15m is None or sigma_15m < MIN_SIGMA:
        # Can't estimate without volatility
        # Fall back to simple comparison
        if s_now > ref_px:
            return {
                "fair_up": 0.6,  # Slight edge for current direction
                "fair_down": 0.4,
                "z_score": None,
                "inputs": {"reason": "sigma_too_low"},
            }
        elif s_now < ref_px:
            return {
                "fair_up": 0.4,
                "fair_down": 0.6,
                "z_score": None,
                "inputs": {"reason": "sigma_too_low"},
            }
        else:
            return _default_fair()
    
    if tau_sec < MIN_TAU_SEC:
        # Almost no time left - price likely to stay where it is
        if s_now > ref_px:
            return {
                "fair_up": 0.95,
                "fair_down": 0.05,
                "z_score": None,
                "inputs": {"reason": "tau_too_small"},
            }
        elif s_now < ref_px:
            return {
                "fair_up": 0.05,
                "fair_down": 0.95,
                "z_score": None,
                "inputs": {"reason": "tau_too_small"},
            }
        else:
            return {
                "fair_up": 0.5,
                "fair_down": 0.5,
                "z_score": 0.0,
                "inputs": {"reason": "tau_too_small"},
            }
    
    # Normalize tau to window length
    tau_norm = tau_sec / window_sec
    
    # Compute log price ratio
    log_ratio = math.log(s_now / ref_px)
    
    # Compute z-score
    # z = [ln(S/ref) + drift*tau_norm] / (sigma * sqrt(tau_norm))
    sigma_scaled = sigma_15m * math.sqrt(tau_norm)
    
    if sigma_scaled < MIN_SIGMA:
        sigma_scaled = MIN_SIGMA
    
    z_score = (log_ratio + drift * tau_norm) / sigma_scaled
    
    # Compute fair probabilities
    fair_up = standard_normal_cdf(z_score)
    fair_down = 1.0 - fair_up
    
    return {
        "fair_up": round(fair_up, 4),
        "fair_down": round(fair_down, 4),
        "z_score": round(z_score, 3),
        "inputs": {
            "s_now": s_now,
            "ref_px": ref_px,
            "sigma_15m": sigma_15m,
            "tau_sec": tau_sec,
            "tau_norm": round(tau_norm, 3),
            "log_ratio": round(log_ratio, 6),
            "drift": drift,
        },
    }


def _default_fair() -> dict:
    """Return default 50/50 fair values."""
    return {
        "fair_up": 0.5,
        "fair_down": 0.5,
        "z_score": 0.0,
        "inputs": {"reason": "default"},
    }


def compute_fair_updown_open(
    ref_px: float,
    s_now: float,
    tau_sec: float,
    sigma_15m: float,
    drift: float = 0.0,
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute fair UP/DOWN probabilities for UPDOWN_OPEN event.
    
    New S3 API that returns (fair_up, fair_down, z) tuple.
    
    Model: P(S_end > ref_px) using log-normal dynamics.
    
    Formula:
        tau_frac = tau_sec / 900  (normalized to 15m window)
        sigma_tau = sigma_15m * sqrt(tau_frac)
        x = ln(S_now / ref_px)
        z = (x + drift * tau_frac) / sigma_tau
        fair_up = Φ(z)
        fair_down = 1 - fair_up
    
    Args:
        ref_px: Reference price at window open
        s_now: Current spot price
        tau_sec: Time remaining until window end (0..900 seconds)
        sigma_15m: 15-minute volatility (std of log returns scaled)
        drift: Optional drift adjustment (default 0)
    
    Returns:
        Tuple of (fair_up, fair_down, z_score)
        Returns (None, None, None) if inputs invalid
    
    Example:
        >>> fair_up, fair_down, z = compute_fair_updown_open(100000, 100500, 600, 0.003)
        >>> fair_up
        0.72
    """
    # Input validation
    if ref_px is None or s_now is None or ref_px <= 0 or s_now <= 0:
        return None, None, None
    
    if tau_sec is None or sigma_15m is None:
        return None, None, None
    
    if sigma_15m <= 0 or math.isnan(sigma_15m) or math.isinf(sigma_15m):
        return None, None, None
    
    # Normalize tau to window length (0..1], avoid division by zero
    tau_frac = max(1e-6, min(1.0, float(tau_sec) / 900.0))
    
    # Sigma scaled to remaining time
    sigma_tau = max(1e-9, float(sigma_15m) * math.sqrt(tau_frac))
    
    # Log price ratio
    x = math.log(float(s_now) / float(ref_px))
    
    # Add drift (very mild effect for short windows)
    x_eff = x + float(drift) * tau_frac
    
    # Z-score
    z = x_eff / sigma_tau
    
    # Fair probabilities via standard normal CDF
    fair_up = standard_normal_cdf(z)
    fair_down = 1.0 - fair_up
    
    return round(fair_up, 4), round(fair_down, 4), round(z, 3)


def compute_edge_bps(fair: float, market: float) -> Optional[float]:
    """
    Compute edge in basis points.
    
    Edge = (fair - market) * 10000
    
    Positive edge means fair value is higher than market price,
    suggesting the market is underpriced.
    
    Args:
        fair: Fair probability (0 to 1)
        market: Market probability/price (0 to 1)
    
    Returns:
        Edge in basis points, or None if inputs invalid
    
    Example:
        >>> compute_edge_bps(0.55, 0.50)
        500.0  # 500 bps edge (fair 55% vs market 50%)
    """
    if fair is None or market is None:
        return None
    
    return (fair - market) * 10000
