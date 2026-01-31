"""
Paper Trading Engine
====================

Virtual execution of edge decisions against real Polymarket orderbook.

Purpose:
- Track hypothetical P&L from edge signals
- Validate signal quality before live trading
- Measure execution feasibility (spread, depth)

Rules v1:
- UP → buy YES @ best_ask
- NO → buy NO @ best_ask  
- qty = 1 (fixed size)
- Mark-to-mid for P&L
- Skip if spread > threshold (but log)

Usage:
    paper = PaperEngine(max_spread_bps=500)
    
    # On each edge decision:
    paper.apply(edge, polymarket_snapshot)
    
    # Get current state:
    snap = paper.snapshot()
    # {"yes_qty": 1.0, "yes_avg_px": 0.48, "pnl": 0.02, ...}
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from collector.utils_time import now_ms

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single paper trade record."""
    ts_ms: int
    side: str  # "YES" | "NO"
    direction: str  # "BUY" | "SELL"
    qty: float
    price: float
    spread_bps: float
    depth_top1: float
    reason: str  # Edge reason


class PaperEngine:
    """
    Paper trading engine for virtual execution.
    
    Tracks positions in YES and NO tokens, calculates mark-to-mid P&L,
    and logs all trading activity.
    
    Args:
        trade_qty: Fixed quantity per trade (default: 1.0)
        max_spread_bps: Maximum spread to execute (default: 500 bps)
        
    Thread Safety:
        Methods should be called from a single thread (decision loop).
    """
    
    def __init__(
        self,
        trade_qty: float = 1.0,
        max_spread_bps: float = 500.0,
    ) -> None:
        self.trade_qty = trade_qty
        self.max_spread_bps = max_spread_bps
        
        # Positions
        self.yes_qty: float = 0.0
        self.yes_avg_px: float = 0.0
        self.no_qty: float = 0.0
        self.no_avg_px: float = 0.0
        
        # P&L tracking
        self.realized_pnl: float = 0.0
        self.unrealized_pnl: float = 0.0
        
        # Last mark prices
        self._last_yes_mid: Optional[float] = None
        self._last_no_mid: Optional[float] = None
        
        # Trade history
        self._trades: list[Trade] = []
        self._total_trades: int = 0
        self._skipped_trades: int = 0
        
        # Statistics
        self._wins: int = 0
        self._losses: int = 0
        
        logger.info(
            "paper_engine_initialized",
            extra={
                "trade_qty": trade_qty,
                "max_spread_bps": max_spread_bps,
            },
        )
    
    def apply(
        self,
        edge: Any,  # EdgeDecision
        polymarket: Optional[dict],
    ) -> Optional[Trade]:
        """
        Apply edge decision to paper portfolio.
        
        Args:
            edge: EdgeDecision with direction (UP/DOWN/NONE)
            polymarket: Polymarket snapshot with yes/no orderbooks
            
        Returns:
            Trade if executed, None otherwise
        """
        if polymarket is None:
            return None
        
        if not polymarket.get("connected", False):
            return None
        
        # Extract edge direction
        direction = self._get_direction(edge)
        if direction == "NONE":
            # Just update mark-to-mid
            self._update_marks(polymarket)
            return None
        
        # Check veto
        if self._is_vetoed(edge):
            self._update_marks(polymarket)
            return None
        
        # Determine which side to trade
        if direction == "UP":
            return self._buy_yes(edge, polymarket)
        elif direction == "DOWN":
            return self._buy_no(edge, polymarket)
        
        return None
    
    def _buy_yes(
        self,
        edge: Any,
        polymarket: dict,
    ) -> Optional[Trade]:
        """Execute buy YES order."""
        yes = polymarket.get("yes", {})
        
        best_ask = yes.get("best_ask")
        spread_bps = yes.get("spread_bps", 9999)
        depth_top1 = yes.get("depth_top1", 0)
        
        if best_ask is None:
            logger.debug("paper_skip_no_ask", extra={"side": "YES"})
            return None
        
        # Check spread
        if spread_bps > self.max_spread_bps:
            self._skipped_trades += 1
            logger.info(
                "paper_skip_spread",
                extra={
                    "side": "YES",
                    "spread_bps": round(spread_bps, 1),
                    "max_spread_bps": self.max_spread_bps,
                    "best_ask": best_ask,
                },
            )
            self._update_marks(polymarket)
            return None
        
        # Execute trade
        trade = self._execute_buy("YES", best_ask, spread_bps, depth_top1, edge)
        self._update_marks(polymarket)
        return trade
    
    def _buy_no(
        self,
        edge: Any,
        polymarket: dict,
    ) -> Optional[Trade]:
        """Execute buy NO order."""
        no = polymarket.get("no", {})
        
        best_ask = no.get("best_ask")
        spread_bps = no.get("spread_bps", 9999)
        depth_top1 = no.get("depth_top1", 0)
        
        if best_ask is None:
            logger.debug("paper_skip_no_ask", extra={"side": "NO"})
            return None
        
        # Check spread
        if spread_bps > self.max_spread_bps:
            self._skipped_trades += 1
            logger.info(
                "paper_skip_spread",
                extra={
                    "side": "NO",
                    "spread_bps": round(spread_bps, 1),
                    "max_spread_bps": self.max_spread_bps,
                    "best_ask": best_ask,
                },
            )
            self._update_marks(polymarket)
            return None
        
        # Execute trade
        trade = self._execute_buy("NO", best_ask, spread_bps, depth_top1, edge)
        self._update_marks(polymarket)
        return trade
    
    def _execute_buy(
        self,
        side: str,
        price: float,
        spread_bps: float,
        depth_top1: float,
        edge: Any,
    ) -> Trade:
        """Execute a buy order and update positions."""
        ts = now_ms()
        qty = self.trade_qty
        
        # Get reason from edge
        reason = self._get_reason(edge)
        
        # Create trade record
        trade = Trade(
            ts_ms=ts,
            side=side,
            direction="BUY",
            qty=qty,
            price=price,
            spread_bps=spread_bps,
            depth_top1=depth_top1,
            reason=reason,
        )
        
        # Update position
        if side == "YES":
            # Update average price
            total_cost = self.yes_qty * self.yes_avg_px + qty * price
            self.yes_qty += qty
            self.yes_avg_px = total_cost / self.yes_qty if self.yes_qty > 0 else 0
        else:  # NO
            total_cost = self.no_qty * self.no_avg_px + qty * price
            self.no_qty += qty
            self.no_avg_px = total_cost / self.no_qty if self.no_qty > 0 else 0
        
        # Record trade
        self._trades.append(trade)
        self._total_trades += 1
        
        # Log trade
        logger.info(
            "paper_trade",
            extra={
                "side": side,
                "qty": qty,
                "price": price,
                "spread_bps": round(spread_bps, 1),
                "depth_top1": round(depth_top1, 1),
                "reason": reason,
                "yes_qty": self.yes_qty,
                "no_qty": self.no_qty,
            },
        )
        
        return trade
    
    def _update_marks(self, polymarket: dict) -> None:
        """Update mark prices and unrealized P&L."""
        yes = polymarket.get("yes", {})
        no = polymarket.get("no", {})
        
        yes_mid = yes.get("mid")
        no_mid = no.get("mid")
        
        if yes_mid is not None:
            self._last_yes_mid = yes_mid
        if no_mid is not None:
            self._last_no_mid = no_mid
        
        # Calculate unrealized P&L (mark-to-mid)
        self.unrealized_pnl = 0.0
        
        if self.yes_qty > 0 and self._last_yes_mid is not None:
            yes_pnl = self.yes_qty * (self._last_yes_mid - self.yes_avg_px)
            self.unrealized_pnl += yes_pnl
        
        if self.no_qty > 0 and self._last_no_mid is not None:
            no_pnl = self.no_qty * (self._last_no_mid - self.no_avg_px)
            self.unrealized_pnl += no_pnl
    
    def _get_direction(self, edge: Any) -> str:
        """Extract direction from edge decision."""
        if hasattr(edge, "direction"):
            return edge.direction
        if isinstance(edge, dict):
            return edge.get("direction", "NONE")
        return "NONE"
    
    def _is_vetoed(self, edge: Any) -> bool:
        """Check if edge is vetoed."""
        if hasattr(edge, "veto"):
            return edge.veto
        if isinstance(edge, dict):
            return edge.get("veto", True)
        return True
    
    def _get_reason(self, edge: Any) -> str:
        """Get primary reason from edge."""
        reasons = []
        if hasattr(edge, "reasons"):
            reasons = edge.reasons
        elif isinstance(edge, dict):
            reasons = edge.get("reasons", [])
        
        return reasons[0] if reasons else "unknown"
    
    def snapshot(self) -> dict:
        """Get current paper trading state."""
        total_pnl = self.realized_pnl + self.unrealized_pnl
        
        return {
            "yes_qty": round(self.yes_qty, 4),
            "yes_avg_px": round(self.yes_avg_px, 4) if self.yes_qty > 0 else None,
            "no_qty": round(self.no_qty, 4),
            "no_avg_px": round(self.no_avg_px, 4) if self.no_qty > 0 else None,
            "realized_pnl": round(self.realized_pnl, 4),
            "unrealized_pnl": round(self.unrealized_pnl, 4),
            "total_pnl": round(total_pnl, 4),
            "total_trades": self._total_trades,
            "skipped_trades": self._skipped_trades,
            "last_yes_mid": self._last_yes_mid,
            "last_no_mid": self._last_no_mid,
        }
    
    def get_recent_trades(self, n: int = 10) -> list[dict]:
        """Get N most recent trades."""
        recent = self._trades[-n:] if n < len(self._trades) else self._trades
        return [
            {
                "ts_ms": t.ts_ms,
                "side": t.side,
                "direction": t.direction,
                "qty": t.qty,
                "price": t.price,
                "spread_bps": round(t.spread_bps, 1),
                "reason": t.reason,
            }
            for t in recent
        ]
    
    def reset(self) -> None:
        """Reset all positions and P&L."""
        self.yes_qty = 0.0
        self.yes_avg_px = 0.0
        self.no_qty = 0.0
        self.no_avg_px = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self._trades.clear()
        self._total_trades = 0
        self._skipped_trades = 0
        
        logger.info("paper_engine_reset")
