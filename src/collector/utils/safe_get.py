"""
Safe Get Utility
================

Safe nested dictionary access with dot-path notation.

Why this exists:
    When working with deeply nested dicts (e.g., API responses, feature snapshots),
    chained .get() calls become verbose and error-prone:
    
        # Bad: verbose and fragile
        value = data.get("level1", {}).get("level2", {}).get("level3")
        
        # Good: clean and readable
        value = safe_get(data, "level1.level2.level3")

Usage:
    from collector.utils import safe_get
    
    features = {"polymarket": {"yes": {"mid": 0.52}}}
    
    mid = safe_get(features, "polymarket.yes.mid")  # 0.52
    bid = safe_get(features, "polymarket.yes.bid", default=0.0)  # 0.0
    
    # Handles None gracefully
    mid = safe_get(None, "any.path")  # None
    mid = safe_get(None, "any.path", default=0.0)  # 0.0

Performance:
    O(n) where n is the number of path segments.
    No regex or complex parsing - just str.split(".").
"""

from typing import Any, Optional


def safe_get(dct: Optional[dict], path: str, default: Any = None) -> Any:
    """
    Safely get a nested value from a dictionary using dot notation.
    
    Args:
        dct: The dictionary to traverse. Can be None.
        path: Dot-separated path to the value, e.g., "level1.level2.key"
        default: Value to return if path doesn't exist or any intermediate 
                 value is not a dict. Defaults to None.
    
    Returns:
        The value at the specified path, or default if not found.
    
    Examples:
        >>> data = {"a": {"b": {"c": 42}}}
        >>> safe_get(data, "a.b.c")
        42
        >>> safe_get(data, "a.b.d", default="missing")
        'missing'
        >>> safe_get(data, "a.x.y")  # x doesn't exist
        None
        >>> safe_get(None, "any.path")
        None
    
    Notes:
        - Empty path ("") returns the dict itself
        - Handles None input gracefully
        - Works with any dict-like object that supports .get()
    """
    if dct is None:
        return default
    
    if not path:
        return dct
    
    cur = dct
    for segment in path.split("."):
        if not isinstance(cur, dict):
            return default
        if segment not in cur:
            return default
        cur = cur[segment]
    
    return cur


def safe_get_float(dct: Optional[dict], path: str, default: float = 0.0) -> float:
    """
    Safe get with float conversion.
    
    Args:
        dct: Dictionary to traverse
        path: Dot-separated path
        default: Default float value if not found or not convertible
    
    Returns:
        Float value at path, or default.
    """
    val = safe_get(dct, path)
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def safe_get_int(dct: Optional[dict], path: str, default: int = 0) -> int:
    """
    Safe get with int conversion.
    
    Args:
        dct: Dictionary to traverse
        path: Dot-separated path
        default: Default int value if not found or not convertible
    
    Returns:
        Int value at path, or default.
    """
    val = safe_get(dct, path)
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def safe_get_bool(dct: Optional[dict], path: str, default: bool = False) -> bool:
    """
    Safe get with bool conversion.
    
    Args:
        dct: Dictionary to traverse
        path: Dot-separated path
        default: Default bool value if not found
    
    Returns:
        Bool value at path, or default.
    """
    val = safe_get(dct, path)
    if val is None:
        return default
    return bool(val)
