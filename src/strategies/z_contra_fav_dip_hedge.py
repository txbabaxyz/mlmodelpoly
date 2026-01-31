"""
================================================================================
STRATEGY: Z_CONTRA_FAV_DIP + HEDGE + OPT5 SIZING
================================================================================

Description:
------------
Strategy for buying the underdog when Binance Z-score confirms it,
while there's a temporary dip on the favorite side. Includes conditional hedging
with the opposite side and dynamic sizing based on confidence metrics.

Backtest Results (73 markets):
------------------------------
- PnL: +$256
- ROI: +8.4%
- Max Drawdown: -$77.5
- PnL/DD Ratio: 3.30
- Win Rate: 53%
- Trades: 442
- Hedges: 251 (57% of positions)

Author: Backtest System
Date: 2026-01-29
================================================================================
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Any
from enum import Enum
import time


# ==============================================================================
# CONSTANTS AND PARAMETERS
# ==============================================================================

class StrategyConfig:
    """All strategy parameters in one place"""
    
    # === ENTRY PARAMETERS ===
    MIN_INTERVAL_SEC = 10.0      # Minimum seconds between entries on one side
    TAU_MIN_SEC = 120            # Don't trade last 2 minutes of market
    UND_PRICE_MIN = 0.20         # Minimum underdog price for entry
    
    # === HEDGE PARAMETERS ===
    HEDGE_MIN_DELAY_SEC = 20     # Minimum seconds after entry before hedge
    HEDGE_MAX_DELAY_SEC = 120    # Maximum seconds after entry for hedge
    HEDGE_Z_MIN = 0.08           # Minimum |z| for hedge trigger
    HEDGE_FAIR_MIN = 0.55        # Minimum fair for hedge trigger
    HEDGE_SIZE_RATIO = 1.0       # Hedge = 100% of position size
    
    # === SIZING PARAMETERS (OPT5 Bias-focused) ===
    SIZE_BASE = 6.0              # Base position size
    SIZE_MIN = 2.0               # Minimum size
    SIZE_MAX = 10.0              # Maximum size
    
    # Thresholds for size adjustment
    FAIR_THRESHOLD = 0.52        # If fair > threshold, +1 to size
    SIGMA_THRESHOLD = 0.002      # If sigma > threshold, +1 to size
    TAU_THRESHOLD = 350          # If tau < threshold, +1 to size
    BIAS_THRESHOLD = 0.15        # If bias > threshold, reduce size
    UND_PRICE_CHEAP = 0.35       # If und_price < threshold, +1 to size


class Side(Enum):
    """Position side"""
    UP = "UP"
    DOWN = "DOWN"


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

@dataclass
class MarketData:
    """
    Market input data (received from features)
    
    Data sources:
    - pm_up_mid, pm_down_mid: Polymarket API (order book mid price)
    - z_score: Computed as Binance deviation from PM fair price
    - tau_sec: Time until market close (seconds)
    - up_dip, down_dip: Temporary price drop detector
    - fair_smooth_up/down: Fair model probability (0-1)
    - sigma_15m: 15-minute volatility
    - bias_strength: Directional movement strength
    """
    ts_ms: int                    # Timestamp in milliseconds
    pm_up_mid: float              # UP token price (mid)
    pm_down_mid: float            # DOWN token price (mid)
    z_score: float                # Z-score Binance vs PM
    tau_sec: float                # Seconds until close
    up_dip: bool                  # Dip on UP side
    down_dip: bool                # Dip on DOWN side
    fair_smooth_up: float = 0.5   # Fair probability UP
    fair_smooth_down: float = 0.5 # Fair probability DOWN
    sigma_15m: float = 0.002      # Volatility
    bias_strength: float = 0.3   # Bias strength


@dataclass
class Position:
    """Open position"""
    entry_ts_ms: int              # Entry timestamp
    side: Side                    # Side (UP/DOWN)
    entry_price: float            # Entry price
    size: float                   # Position size in USD
    
    # Hedge information
    hedged: bool = False
    hedge_ts_ms: Optional[int] = None
    hedge_side: Optional[Side] = None
    hedge_price: Optional[float] = None
    hedge_size: float = 0.0


@dataclass
class StrategyState:
    """Strategy state for one market"""
    last_up_ts_ms: int = 0        # Last entry on UP
    last_down_ts_ms: int = 0      # Last entry on DOWN
    open_positions: List[Position] = field(default_factory=list)


@dataclass
class TradeSignal:
    """Trade signal"""
    action: str                   # "ENTRY" or "HEDGE"
    side: Side                    # Side
    price: float                  # Price
    size: float                   # Size in USD
    reason: str                   # Signal reason


# ==============================================================================
# MAIN STRATEGY LOGIC
# ==============================================================================

class ZContraFavDipStrategy:
    """
    Z_CONTRA_FAV_DIP + HEDGE Strategy
    
    Entry logic:
    1. Determine favorite (side with higher price)
    2. Z-score must confirm UNDERDOG (not favorite)
    3. There must be a DIP on the favorite
    4. Underdog price must be > 0.20
    5. Minimum 10 seconds since last entry on this side
    6. Tau > 120 seconds (don't trade last 2 minutes)
    
    Hedge logic (20-120 sec after entry):
    - (Z confirms hedge side AND |z| > 0.08)
    - OR (fair model for hedge side > 0.55)
    
    Sizing (OPT5 Bias-focused):
    - Base size: $6
    - +1 if fair > 0.52
    - +1 if sigma > 0.002
    - +1 if tau < 350
    - -1...-4 if bias > 0.15
    - +1 if price < 0.35
    - Range: $2-$10
    """
    
    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
    
    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================
    
    def determine_favorite_underdog(
        self, 
        pm_up: float, 
        pm_down: float
    ) -> Tuple[Side, Side, float, float]:
        """
        Determine favorite and underdog by PM prices.
        
        Favorite = side with higher price
        Underdog = side with lower price
        
        Args:
            pm_up: UP token price
            pm_down: DOWN token price
            
        Returns:
            (favorite, underdog, fav_price, und_price)
        """
        if pm_up > pm_down:
            return Side.UP, Side.DOWN, pm_up, pm_down
        else:
            return Side.DOWN, Side.UP, pm_down, pm_up
    
    def z_confirms_side(self, z_score: float, side: Side) -> bool:
        """
        Check if Z-score confirms the given side.
        
        Z > 0 means Binance is above expected -> signal for UP
        Z < 0 means Binance is below expected -> signal for DOWN
        
        Args:
            z_score: Current z-score
            side: Side to check
            
        Returns:
            True if Z confirms the side
        """
        if side == Side.UP:
            return z_score > 0
        else:
            return z_score < 0
    
    def get_side_price(self, side: Side, pm_up: float, pm_down: float) -> float:
        """Return price for the given side"""
        return pm_up if side == Side.UP else pm_down
    
    def get_side_dip(self, side: Side, up_dip: bool, down_dip: bool) -> bool:
        """Return dip flag for the given side"""
        return up_dip if side == Side.UP else down_dip
    
    def get_side_fair(
        self, 
        side: Side, 
        fair_up: float, 
        fair_down: float
    ) -> float:
        """Return fair value for the given side"""
        return fair_up if side == Side.UP else fair_down
    
    def opposite_side(self, side: Side) -> Side:
        """Return opposite side"""
        return Side.DOWN if side == Side.UP else Side.UP
    
    # ==========================================================================
    # POSITION SIZE CALCULATION (OPT5 SIZING)
    # ==========================================================================
    
    def calculate_position_size(self, data: MarketData, side: Side) -> float:
        """
        Calculate position size using OPT5 (Bias-focused) formula.
        
        Formula:
        - Start with base size (6)
        - +1 if fair model > 0.52
        - +1 if volatility sigma > 0.002
        - +1 if tau < 350 (closer to market end)
        - -1...-4 if bias > 0.15 (higher bias = smaller size)
        - +1 if underdog price < 0.35 (cheap underdog)
        
        Args:
            data: Current market data
            side: Entry side
            
        Returns:
            Position size in USD (from 2 to 10)
        """
        cfg = self.config
        score = cfg.SIZE_BASE
        
        # Fair model confirms
        fair = self.get_side_fair(side, data.fair_smooth_up, data.fair_smooth_down)
        if fair > cfg.FAIR_THRESHOLD:
            score += min(1, (fair - cfg.FAIR_THRESHOLD) * 10)
        
        # High volatility = more confidence
        if data.sigma_15m > cfg.SIGMA_THRESHOLD:
            score += min(1, (data.sigma_15m - cfg.SIGMA_THRESHOLD) * 400)
        
        # Closer to market end = more information
        if data.tau_sec < cfg.TAU_THRESHOLD:
            score += 1
        
        # High bias = less confidence
        if data.bias_strength > cfg.BIAS_THRESHOLD:
            penalty = (data.bias_strength - cfg.BIAS_THRESHOLD) * 15
            score -= min(4, penalty)
        
        # Cheap underdog = more potential
        entry_price = self.get_side_price(side, data.pm_up_mid, data.pm_down_mid)
        if entry_price < cfg.UND_PRICE_CHEAP:
            score += 1
        
        return max(cfg.SIZE_MIN, min(cfg.SIZE_MAX, score))
    
    # ==========================================================================
    # ENTRY CONDITIONS CHECK
    # ==========================================================================
    
    def check_entry_conditions(
        self, 
        data: MarketData, 
        state: StrategyState
    ) -> Optional[TradeSignal]:
        """
        Check conditions for position entry.
        
        Entry conditions (all must be met):
        1. tau > TAU_MIN (not last 2 minutes)
        2. Z confirms underdog
        3. Z does NOT confirm favorite
        4. There is a DIP on favorite
        5. Underdog price > UND_PRICE_MIN
        6. At least MIN_INTERVAL since last entry on this side
        
        Args:
            data: Current market data
            state: Strategy state
            
        Returns:
            TradeSignal if conditions met, else None
        """
        cfg = self.config
        
        # 1. Check time until end
        if data.tau_sec < cfg.TAU_MIN_SEC:
            return None
        
        # 2. Determine favorite and underdog
        favorite, underdog, fav_price, und_price = self.determine_favorite_underdog(
            data.pm_up_mid, data.pm_down_mid
        )
        
        # 3. Check underdog price
        if und_price < cfg.UND_PRICE_MIN:
            return None
        
        # 4. Z must confirm UNDERDOG
        if not self.z_confirms_side(data.z_score, underdog):
            return None
        
        # 5. Z must NOT confirm favorite
        if self.z_confirms_side(data.z_score, favorite):
            return None
        
        # 6. Must have DIP on favorite
        fav_dip = self.get_side_dip(favorite, data.up_dip, data.down_dip)
        if not fav_dip:
            return None
        
        # 7. Check interval between trades
        if underdog == Side.UP:
            last_ts = state.last_up_ts_ms
        else:
            last_ts = state.last_down_ts_ms
        
        time_since_last = (data.ts_ms - last_ts) / 1000  # in seconds
        if time_since_last < cfg.MIN_INTERVAL_SEC:
            return None
        
        # All conditions met - generate signal
        entry_price = und_price
        size = self.calculate_position_size(data, underdog)
        
        return TradeSignal(
            action="ENTRY",
            side=underdog,
            price=entry_price,
            size=size,
            reason=f"Z={data.z_score:.3f} confirms {underdog.value}, "
                   f"fav_dip=True, und_price={und_price:.3f}"
        )
    
    # ==========================================================================
    # HEDGE CONDITIONS CHECK
    # ==========================================================================
    
    def check_hedge_conditions(
        self, 
        position: Position, 
        data: MarketData
    ) -> Optional[TradeSignal]:
        """
        Check conditions for hedging an open position.
        
        Hedge conditions (any of them):
        1. (Z confirms hedge side AND |z| > HEDGE_Z_MIN)
        2. OR (fair model for hedge side > HEDGE_FAIR_MIN)
        
        Hedge window: from HEDGE_MIN_DELAY to HEDGE_MAX_DELAY seconds after entry
        
        Args:
            position: Open position
            data: Current market data
            
        Returns:
            TradeSignal if hedge needed, else None
        """
        cfg = self.config
        
        # Already hedged
        if position.hedged:
            return None
        
        # Check time window
        time_since_entry = (data.ts_ms - position.entry_ts_ms) / 1000  # seconds
        
        if time_since_entry < cfg.HEDGE_MIN_DELAY_SEC:
            return None
        if time_since_entry > cfg.HEDGE_MAX_DELAY_SEC:
            return None  # Window closed
        
        # Hedge side - opposite of entry
        hedge_side = self.opposite_side(position.side)
        
        # Get data for hedge side
        hedge_price = self.get_side_price(
            hedge_side, data.pm_up_mid, data.pm_down_mid
        )
        fair_hedge = self.get_side_fair(
            hedge_side, data.fair_smooth_up, data.fair_smooth_down
        )
        
        # Check Z
        z_confirms_hedge = self.z_confirms_side(data.z_score, hedge_side)
        
        # CONDITION 1: Z confirms hedge with sufficient strength
        condition_z = z_confirms_hedge and abs(data.z_score) > cfg.HEDGE_Z_MIN
        
        # CONDITION 2: Fair model shows high probability
        condition_fair = fair_hedge > cfg.HEDGE_FAIR_MIN
        
        # Hedge if any condition met
        if condition_z or condition_fair:
            hedge_size = position.size * cfg.HEDGE_SIZE_RATIO
            
            reason_parts = []
            if condition_z:
                reason_parts.append(f"Z={data.z_score:.3f} confirms {hedge_side.value}")
            if condition_fair:
                reason_parts.append(f"fair={fair_hedge:.3f}")
            
            return TradeSignal(
                action="HEDGE",
                side=hedge_side,
                price=hedge_price,
                size=hedge_size,
                reason=", ".join(reason_parts)
            )
        
        return None
    
    # ==========================================================================
    # MAIN DATA PROCESSING METHOD
    # ==========================================================================
    
    def process_tick(
        self, 
        data: MarketData, 
        state: StrategyState
    ) -> List[TradeSignal]:
        """
        Process new data and return trade signals.
        
        Processing order:
        1. Check hedges for all open positions
        2. Check conditions for new entry
        
        Args:
            data: Current market data
            state: Strategy state (will be modified)
            
        Returns:
            List of trade signals (may be empty)
        """
        signals = []
        
        # 1. Check hedges for open positions
        for position in state.open_positions:
            hedge_signal = self.check_hedge_conditions(position, data)
            if hedge_signal:
                # Update position
                position.hedged = True
                position.hedge_ts_ms = data.ts_ms
                position.hedge_side = hedge_signal.side
                position.hedge_price = hedge_signal.price
                position.hedge_size = hedge_signal.size
                
                signals.append(hedge_signal)
        
        # 2. Check entry conditions
        entry_signal = self.check_entry_conditions(data, state)
        if entry_signal:
            # Update last entry timestamp
            if entry_signal.side == Side.UP:
                state.last_up_ts_ms = data.ts_ms
            else:
                state.last_down_ts_ms = data.ts_ms
            
            # Create new position
            new_position = Position(
                entry_ts_ms=data.ts_ms,
                side=entry_signal.side,
                entry_price=entry_signal.price,
                size=entry_signal.size
            )
            state.open_positions.append(new_position)
            
            signals.append(entry_signal)
        
        return signals
    
    # ==========================================================================
    # PnL CALCULATION
    # ==========================================================================
    
    @staticmethod
    def calculate_position_pnl(position: Position, winner: Side) -> float:
        """
        Calculate PnL for a position.
        
        Formula:
        - If won: contracts * $1 - size
        - If lost: -size
        
        Args:
            position: Position
            winner: Winning side
            
        Returns:
            PnL in USD
        """
        # Main position PnL
        contracts = position.size / position.entry_price
        if position.side == winner:
            pnl = contracts * 1.0 - position.size
        else:
            pnl = -position.size
        
        # Hedge PnL
        if position.hedged and position.hedge_price:
            hedge_contracts = position.hedge_size / position.hedge_price
            if position.hedge_side == winner:
                hedge_pnl = hedge_contracts * 1.0 - position.hedge_size
            else:
                hedge_pnl = -position.hedge_size
            pnl += hedge_pnl
        
        return pnl
    
    @staticmethod
    def determine_winner(
        pm_up_final: float, 
        pm_down_final: float,
        btc_final: Optional[float] = None,
        btc_ref: Optional[float] = None
    ) -> Tuple[Side, str]:
        """
        Determine market winner.
        
        Logic:
        1. If one PM side > 0.9 -> it won
        2. Compare with Binance (btc_final vs btc_ref)
        3. On conflict - PM priority if > 0.9
        
        Args:
            pm_up_final: Final UP price
            pm_down_final: Final DOWN price
            btc_final: Final BTC price on Binance
            btc_ref: Reference BTC price
            
        Returns:
            (winner, method) - winner and determination method
        """
        winner_pm = None
        winner_bn = None
        
        # By PM prices
        if pm_up_final > 0.9:
            winner_pm = Side.UP
        elif pm_down_final > 0.9:
            winner_pm = Side.DOWN
        elif pm_up_final > pm_down_final:
            winner_pm = Side.UP
        else:
            winner_pm = Side.DOWN
        
        # By Binance
        if btc_final and btc_ref:
            winner_bn = Side.UP if btc_final > btc_ref else Side.DOWN
        
        # Final decision
        if winner_pm and winner_bn:
            if winner_pm == winner_bn:
                return winner_pm, "confirmed"
            else:
                if pm_up_final > 0.9 or pm_down_final > 0.9:
                    return winner_pm, "PM_price>0.9"
                else:
                    return winner_bn, "BN_override"
        elif winner_pm:
            return winner_pm, "PM_only"
        else:
            return Side.UP, "default"


# ==============================================================================
# USAGE EXAMPLE
# ==============================================================================

def example_usage():
    """
    Example strategy usage.
    
    In real code you need to:
    1. Connect to Polymarket API to get prices
    2. Connect to Binance API to get z_score and other data
    3. Call process_tick() on each data update
    4. Send orders to exchange when receiving signals
    """
    
    # Create strategy
    strategy = ZContraFavDipStrategy()
    
    # Create state for market
    state = StrategyState()
    
    # Example data (in reality obtained from API)
    data = MarketData(
        ts_ms=int(time.time() * 1000),
        pm_up_mid=0.42,
        pm_down_mid=0.58,
        z_score=0.15,
        tau_sec=500,
        up_dip=True,
        down_dip=False,
        fair_smooth_up=0.45,
        fair_smooth_down=0.55,
        sigma_15m=0.003,
        bias_strength=0.10
    )
    
    # Process data
    signals = strategy.process_tick(data, state)
    
    # Handle signals
    for signal in signals:
        print(f"[{signal.action}] {signal.side.value} @ ${signal.price:.3f}")
        print(f"  Size: ${signal.size:.2f}")
        print(f"  Reason: {signal.reason}")
        
        # Here you would send order to exchange
        # execute_order(signal.side, signal.size, signal.price)


if __name__ == "__main__":
    example_usage()
