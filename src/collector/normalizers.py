"""
Event Normalizers Module
========================

Functions to normalize raw Binance WebSocket events into NormalizedEvent format.
Each normalizer handles a specific event type from a specific market.

Supported markets:
    - futures: aggTrade, bookTicker, markPrice, forceOrder, depthUpdate
    - spot: aggTrade, bookTicker
"""

import logging
from typing import Optional

from collector.types import NormalizedEvent, SCHEMA_VERSION

logger = logging.getLogger(__name__)


# =============================================================================
# FUTURES NORMALIZERS
# =============================================================================

def normalize_futures_aggtrade(
    envelope: dict,
    symbol: str,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Futures aggTrade event.
    
    Args:
        envelope: Raw message envelope from WSClient
        symbol: Symbol name (uppercase, e.g., "BTCUSDT")
    
    Returns:
        NormalizedEvent with market="futures", type="aggTrade"
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        event_type = data.get("e")
        if event_type != "aggTrade":
            return None
        
        # Validate symbol
        data_symbol = data.get("s", "").upper()
        if data_symbol and data_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": data_symbol, "market": "futures"}
            )
            return None
        
        ts_event_ms = data.get("E")
        price_str = data.get("p")
        qty_str = data.get("q")
        is_buyer_maker = data.get("m")
        agg_id = data.get("a")
        
        if any(v is None for v in [ts_event_ms, price_str, qty_str, is_buyer_maker, agg_id]):
            logger.warning("normalizer_missing_field", extra={"type": "aggTrade", "market": "futures"})
            return None
        
        try:
            price = float(price_str)
            qty = float(qty_str)
        except (ValueError, TypeError):
            return None
        
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        side_aggressor = "sell" if is_buyer_maker else "buy"
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="futures",
            symbol=symbol,
            type="aggTrade",
            data={
                "price": price,
                "qty": qty,
                "is_buyer_maker": is_buyer_maker,
                "side_aggressor": side_aggressor,
                "agg_id": agg_id,
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "aggTrade", "market": "futures", "error": str(e)})
        return None


def normalize_futures_bookticker(
    envelope: dict,
    symbol: str,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Futures bookTicker event.
    
    Returns:
        NormalizedEvent with market="futures", type="bookTicker"
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        event_type = data.get("e")
        if event_type != "bookTicker":
            return None
        
        # Validate symbol
        data_symbol = data.get("s", "").upper()
        if data_symbol and data_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": data_symbol, "market": "futures"}
            )
            return None
        
        ts_event_ms = data.get("E")
        bid_px_str = data.get("b")
        bid_qty_str = data.get("B")
        ask_px_str = data.get("a")
        ask_qty_str = data.get("A")
        
        if any(v is None for v in [ts_event_ms, bid_px_str, bid_qty_str, ask_px_str, ask_qty_str]):
            logger.warning("normalizer_missing_field", extra={"type": "bookTicker", "market": "futures"})
            return None
        
        try:
            bid_px = float(bid_px_str)
            bid_qty = float(bid_qty_str)
            ask_px = float(ask_px_str)
            ask_qty = float(ask_qty_str)
        except (ValueError, TypeError):
            return None
        
        mid_px = (bid_px + ask_px) / 2
        spread_px = ask_px - bid_px
        spread_bps = (spread_px / mid_px * 10000) if mid_px > 0 else 0.0
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="futures",
            symbol=symbol,
            type="bookTicker",
            data={
                "bid_px": bid_px,
                "bid_qty": bid_qty,
                "ask_px": ask_px,
                "ask_qty": ask_qty,
                "mid_px": mid_px,
                "spread_px": spread_px,
                "spread_bps": round(spread_bps, 2),
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "bookTicker", "market": "futures", "error": str(e)})
        return None


def normalize_futures_markprice(
    envelope: dict,
    symbol: str,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Futures markPrice event.
    
    Returns:
        NormalizedEvent with market="futures", type="markPrice"
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        event_type = data.get("e")
        if event_type != "markPriceUpdate":
            return None
        
        # Validate symbol
        data_symbol = data.get("s", "").upper()
        if data_symbol and data_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": data_symbol, "market": "futures"}
            )
            return None
        
        ts_event_ms = data.get("E")
        mark_px_str = data.get("p")
        
        if ts_event_ms is None or mark_px_str is None:
            logger.warning("normalizer_missing_field", extra={"type": "markPrice", "market": "futures"})
            return None
        
        try:
            mark_px = float(mark_px_str)
        except (ValueError, TypeError):
            return None
        
        # Optional fields
        index_px = None
        funding_rate = None
        next_funding_time_ms = None
        
        if data.get("i"):
            try:
                index_px = float(data["i"])
            except (ValueError, TypeError):
                pass
        
        if data.get("r"):
            try:
                funding_rate = float(data["r"])
            except (ValueError, TypeError):
                pass
        
        if data.get("T"):
            try:
                next_funding_time_ms = int(data["T"])
            except (ValueError, TypeError):
                pass
        
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="futures",
            symbol=symbol,
            type="markPrice",
            data={
                "mark_px": mark_px,
                "index_px": index_px,
                "funding_rate": funding_rate,
                "next_funding_time_ms": next_funding_time_ms,
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "markPrice", "market": "futures", "error": str(e)})
        return None


def normalize_futures_liquidation(
    envelope: dict,
    symbol: str,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Futures forceOrder (liquidation) event.
    
    Returns:
        NormalizedEvent with market="futures", type="liquidation"
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        event_type = data.get("e")
        if event_type != "forceOrder":
            return None
        
        ts_event_ms = data.get("E")
        order = data.get("o", {})
        
        if not order or ts_event_ms is None:
            logger.warning("normalizer_missing_field", extra={"type": "liquidation", "market": "futures"})
            return None
        
        # Validate symbol
        order_symbol = order.get("s", "").upper()
        if order_symbol and order_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": order_symbol, "market": "futures"}
            )
            return None
        
        side_raw = order.get("S")
        price_str = order.get("p")
        qty_str = order.get("q")
        
        if any(v is None for v in [side_raw, price_str, qty_str]):
            logger.warning("normalizer_missing_field", extra={"type": "liquidation", "market": "futures"})
            return None
        
        try:
            price = float(price_str)
            qty = float(qty_str)
        except (ValueError, TypeError):
            return None
        
        side = side_raw.lower()
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="futures",
            symbol=symbol,
            type="liquidation",
            data={
                "side": side,
                "price": price,
                "qty": qty,
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "liquidation", "market": "futures", "error": str(e)})
        return None


def normalize_futures_depth(
    envelope: dict,
    symbol: str,
    topn: int = 10,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Futures depthUpdate event.
    
    Returns:
        NormalizedEvent with market="futures", type="depth"
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        event_type = data.get("e")
        if event_type != "depthUpdate":
            return None
        
        # Validate symbol
        data_symbol = data.get("s", "").upper()
        if data_symbol and data_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": data_symbol, "market": "futures"}
            )
            return None
        
        ts_event_ms = data.get("E")
        u_first = data.get("U")
        u_final = data.get("u")
        bids_raw = data.get("b", [])
        asks_raw = data.get("a", [])
        
        if ts_event_ms is None or u_first is None or u_final is None:
            logger.warning("normalizer_missing_field", extra={"type": "depth", "market": "futures"})
            return None
        
        def convert_levels(levels: list, limit: int) -> list[dict]:
            result = []
            for level in levels[:limit]:
                if len(level) >= 2:
                    try:
                        result.append({
                            "px": float(level[0]),
                            "qty": float(level[1]),
                        })
                    except (ValueError, TypeError):
                        continue
            return result
        
        bids = convert_levels(bids_raw, topn)
        asks = convert_levels(asks_raw, topn)
        
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="futures",
            symbol=symbol,
            type="depth",
            data={
                "u_first": u_first,
                "u_final": u_final,
                "bids": bids,
                "asks": asks,
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "depth", "market": "futures", "error": str(e)})
        return None


# =============================================================================
# SPOT NORMALIZERS
# =============================================================================

def normalize_spot_aggtrade(
    envelope: dict,
    symbol: str,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Spot aggTrade event.
    
    Args:
        envelope: Raw message envelope from WSClient
        symbol: Symbol name (uppercase, e.g., "BTCUSDT")
    
    Returns:
        NormalizedEvent with market="spot", type="aggTrade"
    
    Spot aggTrade fields:
        e: "aggTrade"
        E: Event time (ms)
        s: Symbol
        a: Aggregate trade ID
        p: Price (string)
        q: Quantity (string)
        f: First trade ID
        l: Last trade ID
        T: Trade time (ms)
        m: Is buyer maker
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        event_type = data.get("e")
        if event_type != "aggTrade":
            return None
        
        # Validate symbol
        data_symbol = data.get("s", "").upper()
        if data_symbol and data_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": data_symbol, "market": "spot"}
            )
            return None
        
        ts_event_ms = data.get("E")
        price_str = data.get("p")
        qty_str = data.get("q")
        is_buyer_maker = data.get("m")
        agg_id = data.get("a")
        
        if any(v is None for v in [ts_event_ms, price_str, qty_str, is_buyer_maker, agg_id]):
            logger.warning("normalizer_missing_field", extra={"type": "aggTrade", "market": "spot"})
            return None
        
        try:
            price = float(price_str)
            qty = float(qty_str)
        except (ValueError, TypeError):
            return None
        
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        side_aggressor = "sell" if is_buyer_maker else "buy"
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="spot",
            symbol=symbol,
            type="aggTrade",
            data={
                "price": price,
                "qty": qty,
                "is_buyer_maker": is_buyer_maker,
                "side_aggressor": side_aggressor,
                "agg_id": agg_id,
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "aggTrade", "market": "spot", "error": str(e)})
        return None


def normalize_spot_bookticker(
    envelope: dict,
    symbol: str,
) -> Optional[NormalizedEvent]:
    """
    Normalize Binance Spot bookTicker event.
    
    Args:
        envelope: Raw message envelope from WSClient
        symbol: Symbol name (uppercase, e.g., "BTCUSDT")
    
    Returns:
        NormalizedEvent with market="spot", type="bookTicker"
    
    Spot bookTicker fields:
        e: "bookTicker" (may not be present in spot!)
        u: Order book updateId
        s: Symbol
        b: Best bid price
        B: Best bid qty
        a: Best ask price
        A: Best ask qty
    
    Note: Spot bookTicker may not have "E" event time field.
    In that case, we use recv_ts_ms as ts_event_ms.
    """
    try:
        recv_ts_ms = envelope.get("recv_ts_ms")
        msg = envelope.get("msg", {})
        data = msg.get("data", {})
        
        if not data:
            return None
        
        # Spot bookTicker might not have "e" field in some cases
        # Check for presence of bid/ask fields instead
        event_type = data.get("e")
        if event_type and event_type != "bookTicker":
            return None
        
        # Must have bid/ask fields
        bid_px_str = data.get("b")
        ask_px_str = data.get("a")
        if bid_px_str is None or ask_px_str is None:
            return None
        
        # Validate symbol
        data_symbol = data.get("s", "").upper()
        if data_symbol and data_symbol != symbol:
            logger.warning(
                "normalizer_symbol_mismatch",
                extra={"expected": symbol, "got": data_symbol, "market": "spot"}
            )
            return None
        
        # Event time - may not be present in spot bookTicker
        ts_event_ms = data.get("E")
        if ts_event_ms is None:
            ts_event_ms = recv_ts_ms  # Use receive time as fallback
        
        bid_qty_str = data.get("B")
        ask_qty_str = data.get("A")
        
        if bid_qty_str is None or ask_qty_str is None:
            logger.warning("normalizer_missing_field", extra={"type": "bookTicker", "market": "spot"})
            return None
        
        try:
            bid_px = float(bid_px_str)
            bid_qty = float(bid_qty_str)
            ask_px = float(ask_px_str)
            ask_qty = float(ask_qty_str)
        except (ValueError, TypeError):
            return None
        
        mid_px = (bid_px + ask_px) / 2
        spread_px = ask_px - bid_px
        spread_bps = (spread_px / mid_px * 10000) if mid_px > 0 else 0.0
        lag_ms = recv_ts_ms - ts_event_ms if recv_ts_ms and ts_event_ms else 0
        
        return NormalizedEvent(
            schema_version=SCHEMA_VERSION,
            ts_event_ms=ts_event_ms,
            ts_recv_ms=recv_ts_ms,
            market="spot",
            symbol=symbol,
            type="bookTicker",
            data={
                "bid_px": bid_px,
                "bid_qty": bid_qty,
                "ask_px": ask_px,
                "ask_qty": ask_qty,
                "mid_px": mid_px,
                "spread_px": spread_px,
                "spread_bps": round(spread_bps, 2),
                "lag_ms": lag_ms,
            },
        )
        
    except Exception as e:
        logger.exception("normalizer_error", extra={"type": "bookTicker", "market": "spot", "error": str(e)})
        return None
