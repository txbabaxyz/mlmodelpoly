"""
Polymarket UP/DOWN Normalization
================================

This module normalizes Polymarket YES/NO token data into a consistent
UP/DOWN format based on the market's directional mapping.

Why this exists:
    Polymarket markets have YES and NO tokens, but our trading signals
    are in terms of UP (price will rise) and DOWN (price will fall).
    
    The mapping depends on the market question:
    - "Will BTC be above $X?" → YES = UP, NO = DOWN (up_is_yes = True)
    - "Will BTC be below $X?" → YES = DOWN, NO = UP (up_is_yes = False)
    
    This module abstracts that mapping so the rest of the system can
    work in consistent UP/DOWN terms.

Key Functions:
    map_up_down(): Convert YES/NO to UP/DOWN based on mapping
    compute_required_edge_bps(): Calculate minimum edge needed to overcome spread
    slice_ok(): Check if there's enough liquidity for a slice order

Example:
    pm_snapshot = polymarket_store.snapshot()
    up, down = map_up_down(pm_snapshot, up_is_yes=True)
    
    # Now 'up' contains the book for UP direction
    up_ask = up.get("best_ask")  # Price to buy UP
    up_depth = up.get("depth_top1")  # Liquidity at best ask

Configuration:
    settings.POLY_UP_IS_YES: bool - Controls the YES→UP mapping
    settings.MIN_TOP1_USD: float - Minimum top1 depth for single slice
    settings.MIN_TOP3_USD: float - Minimum top3 depth for comfortable execution
    settings.EDGE_BUFFER_BPS: float - Additional buffer over half-spread
"""

from typing import Optional


def map_up_down(pm_snapshot: dict, up_is_yes: bool) -> tuple[dict, dict]:
    """
    Map YES/NO token data to UP/DOWN based on market direction.
    
    Args:
        pm_snapshot: Polymarket snapshot containing "yes" and "no" dicts
        up_is_yes: If True, YES token represents UP direction
    
    Returns:
        Tuple of (up_dict, down_dict) with orderbook data for each direction
    
    Example:
        >>> pm = {"yes": {"mid": 0.60}, "no": {"mid": 0.40}}
        >>> up, down = map_up_down(pm, up_is_yes=True)
        >>> up["mid"]
        0.60
    
    Notes:
        - Returns empty dicts if input keys are missing
        - Does not modify the original snapshot
    """
    yes = pm_snapshot.get("yes") or {}
    no = pm_snapshot.get("no") or {}
    
    if up_is_yes:
        return yes, no
    return no, yes


def compute_required_edge_bps(
    spread_bps: Optional[float],
    buffer_bps: float
) -> Optional[float]:
    """
    Calculate the minimum edge (in bps) required to be profitable.
    
    The logic:
        When you buy at the ask and sell at the bid, you pay the spread.
        For a round-trip, you pay half the spread on entry (slippage to mid).
        Adding a buffer ensures we don't execute marginal trades.
    
    Formula:
        required_edge = (spread_bps / 2) + buffer_bps
    
    Args:
        spread_bps: Current spread in basis points (ask - bid) / mid * 10000
        buffer_bps: Additional safety buffer in bps (from config.EDGE_BUFFER_BPS)
    
    Returns:
        Minimum required edge in bps, or None if spread is unknown
    
    Example:
        >>> compute_required_edge_bps(200.0, 25.0)  # 2% spread, 25bps buffer
        125.0  # Need at least 125bps edge to be profitable
    
    Usage:
        If your edge_score (converted to bps) < required_edge → don't trade
    """
    if spread_bps is None:
        return None
    return (spread_bps / 2.0) + buffer_bps


def slice_ok(
    depth_top1_usd: Optional[float],
    depth_top3_usd: Optional[float],
    min_top1: float,
    min_top3: float
) -> bool:
    """
    Check if there's enough liquidity to execute at least one slice.
    
    A "slice" is a small fixed-size order (e.g., $20). We need enough
    depth to execute without excessive slippage.
    
    Criteria (any one is sufficient):
        1. depth_top1 >= min_top1: Can fill entire slice at best price
        2. depth_top3 >= min_top3: Can fill across top 3 levels comfortably
    
    Args:
        depth_top1_usd: USD depth at best bid/ask (top 1 level)
        depth_top3_usd: USD depth across top 3 levels
        min_top1: Minimum top1 depth required (from config.MIN_TOP1_USD)
        min_top3: Minimum top3 depth required (from config.MIN_TOP3_USD)
    
    Returns:
        True if liquidity is sufficient for one slice, False otherwise
    
    Example:
        >>> slice_ok(25.0, 80.0, min_top1=20.0, min_top3=60.0)
        True  # top1 of $25 >= $20 minimum
        
        >>> slice_ok(10.0, 40.0, min_top1=20.0, min_top3=60.0)
        False  # Neither threshold met
    
    Notes:
        - Returns False if all depths are None
        - Designed to be conservative: if unsure, return False
    """
    # Check top1 first (can fill at best price)
    if depth_top1_usd is not None and depth_top1_usd >= min_top1:
        return True
    
    # Check top3 (can fill across levels)
    if depth_top3_usd is not None and depth_top3_usd >= min_top3:
        return True
    
    return False


def build_side_dict(
    side_data: dict,
    min_top1: float,
    min_top3: float,
    edge_buffer_bps: float
) -> dict:
    """
    Build normalized side dictionary with computed fields.
    
    Takes raw orderbook data for one side (UP or DOWN) and adds:
    - slice_ok: Whether we can execute a slice
    - required_edge_bps: Minimum edge needed to overcome spread
    
    Args:
        side_data: Raw orderbook data dict with bid, ask, mid, spread_bps, depth_top*
        min_top1: Minimum top1 depth for slice_ok
        min_top3: Minimum top3 depth for slice_ok
        edge_buffer_bps: Buffer to add to half-spread
    
    Returns:
        Normalized dict with original fields plus computed fields
    
    Example:
        >>> raw = {"best_ask": 0.55, "spread_bps": 200, "depth_top1": 50}
        >>> build_side_dict(raw, 20, 60, 25)
        {
            "bid": None,
            "ask": 0.55,
            "mid": None,
            "spread_bps": 200,
            "depth_top1_usd": 50,
            "depth_top3_usd": None,
            "slice_ok": True,
            "required_edge_bps": 125.0
        }
    """
    spread_bps = side_data.get("spread_bps")
    depth_top1 = side_data.get("depth_top1")
    depth_top3 = side_data.get("depth_top3")
    
    return {
        "bid": side_data.get("best_bid"),
        "ask": side_data.get("best_ask"),
        "mid": side_data.get("mid"),
        "spread_bps": spread_bps,
        "depth_top1_usd": depth_top1,
        "depth_top3_usd": depth_top3,
        "slice_ok": slice_ok(depth_top1, depth_top3, min_top1, min_top3),
        "required_edge_bps": compute_required_edge_bps(spread_bps, edge_buffer_bps),
    }


def normalize_polymarket_updown(
    pm_snapshot: dict,
    up_is_yes: bool,
    min_top1: float,
    min_top3: float,
    edge_buffer_bps: float
) -> dict:
    """
    Full normalization of Polymarket snapshot to UP/DOWN format.
    
    This is the main entry point for BLOCK 1. It:
    1. Maps YES/NO to UP/DOWN based on market direction
    2. Computes slice_ok for each side
    3. Computes required_edge_bps for each side
    4. Packages everything in a clean dict for features
    
    Args:
        pm_snapshot: Raw Polymarket snapshot from book_store.snapshot()
        up_is_yes: True if YES token = UP direction
        min_top1: Minimum top1 depth for slice execution
        min_top3: Minimum top3 depth for comfortable execution
        edge_buffer_bps: Buffer to add to half-spread for required_edge
    
    Returns:
        Dict with structure:
        {
            "up": {bid, ask, mid, spread_bps, depth_top1_usd, depth_top3_usd, slice_ok, required_edge_bps},
            "down": {...same...},
            "up_is_yes": bool,
            "ts_ms": int,
            "connected": bool,
            "age_sec": float
        }
    
    Example:
        >>> from collector.config import settings
        >>> pm = book_store.snapshot()
        >>> updown = normalize_polymarket_updown(
        ...     pm,
        ...     settings.POLY_UP_IS_YES,
        ...     settings.MIN_TOP1_USD,
        ...     settings.MIN_TOP3_USD,
        ...     settings.EDGE_BUFFER_BPS
        ... )
        >>> if updown["up"]["slice_ok"]:
        ...     print("Can execute UP slice")
    """
    up_raw, down_raw = map_up_down(pm_snapshot, up_is_yes)
    
    return {
        "up": build_side_dict(up_raw, min_top1, min_top3, edge_buffer_bps),
        "down": build_side_dict(down_raw, min_top1, min_top3, edge_buffer_bps),
        "up_is_yes": up_is_yes,
        "ts_ms": pm_snapshot.get("ts_ms"),
        "connected": pm_snapshot.get("connected", False),
        "age_sec": pm_snapshot.get("age_sec"),
    }
