# Z_CONTRA_FAV_DIP + HEDGE Strategy

## Overview

This is a fully documented trading strategy for Polymarket BTC UP/DOWN markets.

### Backtest Results (73 markets):
| Metric | Value |
|--------|-------|
| **PnL** | +$256.10 |
| **ROI** | +8.38% |
| **Max Drawdown** | -$77.50 |
| **PnL/DD Ratio** | 3.30 |
| **Trades** | 442 |
| **Hedges** | 251 (57%) |

---

## Files

```
src/strategies/
├── z_contra_fav_dip_hedge.py    # Main strategy module
├── README_STRATEGY.md           # This documentation
```

---

## Installation and Usage

### 1. Dependencies

```bash
pip install numpy
```

### 2. Import Strategy

```python
from strategies.z_contra_fav_dip_hedge import (
    ZContraFavDipStrategy,
    StrategyState,
    MarketData,
    Side
)
```

---

## Strategy Logic

### Entry Conditions (Z_CONTRA_FAV_DIP)

Buy **UNDERDOG** when:

1. **tau > 120 sec** — don't trade last 2 minutes
2. **Z confirms underdog** — z > 0 for UP, z < 0 for DOWN
3. **Z does NOT confirm favorite** — exclude ambiguous signals
4. **There is a DIP on favorite** — temporary price drop on favorite
5. **Underdog price > 0.20** — not too cheap
6. **Interval > 10 sec** — since last entry on this side

```python
# Pseudocode
if tau > 120:
    fav, und = determine_favorite_underdog(pm_up, pm_down)
    
    if z_confirms(und) and not z_confirms(fav):
        if fav_dip and und_price > 0.20:
            if time_since_last_entry > 10:
                BUY(und)
```

### Hedge Conditions

Hedge **opposite side** 20-120 sec after entry if:

```
(Z confirms hedge side AND |z| > 0.08)
OR
(fair model for hedge side > 0.55)
```

```python
# Pseudocode
if 20 < time_since_entry < 120:
    hedge_side = opposite(entry_side)
    
    if (z_confirms(hedge_side) and abs(z) > 0.08) or fair[hedge_side] > 0.55:
        BUY(hedge_side, size=entry_size)
```

---

## Sizing Formula (OPT5 Bias-focused)

```python
def calculate_size(data, side):
    score = 6.0  # Base size
    
    # +1 if fair model confirms
    if fair[side] > 0.52:
        score += 1
    
    # +1 if high volatility
    if sigma_15m > 0.002:
        score += 1
    
    # +1 if closer to market end
    if tau < 350:
        score += 1
    
    # -1...-4 if high bias
    if bias > 0.15:
        score -= min(4, (bias - 0.15) * 15)
    
    # +1 if cheap underdog
    if entry_price < 0.35:
        score += 1
    
    return clamp(score, min=2, max=10)
```

**Average position size:** ~$4.40

---

## Input Data

### From Polymarket

| Field | Type | Description |
|-------|------|-------------|
| `pm_up_mid` | float | UP token price (mid bid/ask) |
| `pm_down_mid` | float | DOWN token price (mid bid/ask) |

### From Binance / Computed

| Field | Type | Description |
|-------|------|-------------|
| `z_score` | float | Z-score of BTC deviation from fair price |
| `tau_sec` | float | Seconds until market close |
| `up_dip` | bool | Dip detector for UP side |
| `down_dip` | bool | Dip detector for DOWN side |
| `fair_smooth_up` | float | Fair probability UP (0-1) |
| `fair_smooth_down` | float | Fair probability DOWN (0-1) |
| `sigma_15m` | float | 15-minute volatility |
| `bias_strength` | float | Directional movement strength |

### How z_score is Computed

```python
# fair_price - expected BTC price based on PM probabilities
# s_now - current BTC price on Binance
# sigma - volatility

z_score = (s_now - fair_price) / sigma

# z > 0: Binance above expected -> signal for UP
# z < 0: Binance below expected -> signal for DOWN
```

---

## Determining Winner

```python
def determine_winner(pm_up_final, pm_down_final, btc_final, btc_ref):
    # 1. If PM price > 0.9 - clear winner
    if pm_up_final > 0.9:
        return "UP"
    if pm_down_final > 0.9:
        return "DOWN"
    
    # 2. Compare with Binance
    btc_winner = "UP" if btc_final > btc_ref else "DOWN"
    pm_winner = "UP" if pm_up_final > pm_down_final else "DOWN"
    
    # 3. If they match - confirmed
    if btc_winner == pm_winner:
        return btc_winner
    
    # 4. On conflict - Binance takes priority
    return btc_winner
```

---

## PnL Calculation

```python
def calculate_pnl(position, winner):
    # Main position
    contracts = position.size / position.entry_price
    if position.side == winner:
        pnl = contracts * 1.0 - position.size  # Receive $1 per contract
    else:
        pnl = -position.size  # Lose the stake
    
    # Hedge (if exists)
    if position.hedged:
        hedge_contracts = position.hedge_size / position.hedge_price
        if position.hedge_side == winner:
            hedge_pnl = hedge_contracts * 1.0 - position.hedge_size
        else:
            hedge_pnl = -position.hedge_size
        pnl += hedge_pnl
    
    return pnl
```

**Example:**
- Entry: $5 on DOWN @ 0.42 -> contracts = 5/0.42 = 11.9
- Hedge: $5 on UP @ 0.58 -> contracts = 5/0.58 = 8.6
- If DOWN wins:
  - Main: 11.9 * $1 - $5 = +$6.90
  - Hedge: -$5
  - **Total: +$1.90**
- If UP wins:
  - Main: -$5
  - Hedge: 8.6 * $1 - $5 = +$3.60
  - **Total: -$1.40**

---

## All Constants

```python
# === ENTRIES ===
MIN_INTERVAL_SEC = 10       # Interval between entries
TAU_MIN_SEC = 120           # Don't trade last 2 min
UND_PRICE_MIN = 0.20        # Min underdog price

# === HEDGES ===
HEDGE_MIN_DELAY_SEC = 20    # Min hedge delay
HEDGE_MAX_DELAY_SEC = 120   # Max hedge delay
HEDGE_Z_MIN = 0.08          # |z| threshold for hedge
HEDGE_FAIR_MIN = 0.55       # fair threshold for hedge

# === SIZING ===
SIZE_BASE = 6.0             # Base size
SIZE_MIN = 2.0              # Minimum
SIZE_MAX = 10.0             # Maximum
FAIR_THRESHOLD = 0.52       # fair threshold
SIGMA_THRESHOLD = 0.002     # sigma threshold
TAU_THRESHOLD = 350         # tau threshold
BIAS_THRESHOLD = 0.15       # bias threshold
UND_PRICE_CHEAP = 0.35      # cheap underdog threshold
```

---

## Usage Example

```python
from strategies.z_contra_fav_dip_hedge import (
    ZContraFavDipStrategy,
    StrategyState,
    MarketData
)

# 1. Create strategy
strategy = ZContraFavDipStrategy()

# 2. Create state for market
state = StrategyState()

# 3. Process data in loop
while market_is_open:
    # Get data from API
    data = MarketData(
        ts_ms=current_timestamp_ms,
        pm_up_mid=get_pm_up_price(),
        pm_down_mid=get_pm_down_price(),
        z_score=calculate_z_score(),
        tau_sec=get_time_to_close(),
        up_dip=detect_up_dip(),
        down_dip=detect_down_dip(),
        fair_smooth_up=get_fair_up(),
        fair_smooth_down=get_fair_down(),
        sigma_15m=get_volatility(),
        bias_strength=get_bias()
    )
    
    # Process
    signals = strategy.process_tick(data, state)
    
    # Execute signals
    for signal in signals:
        if signal.action == "ENTRY":
            execute_buy_order(signal.side, signal.size, signal.price)
        elif signal.action == "HEDGE":
            execute_buy_order(signal.side, signal.size, signal.price)
    
    time.sleep(1)  # Update interval

# 4. After market close, calculate PnL
winner = strategy.determine_winner(pm_up_final, pm_down_final, btc_final, btc_ref)
total_pnl = sum(
    strategy.calculate_position_pnl(pos, winner) 
    for pos in state.open_positions
)
```

---

## Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     NEW DATA (every second)                      │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. CHECK HEDGES for open positions                             │
│     └─ For each position:                                       │
│        └─ If 20 < time_since_entry < 120 sec:                   │
│           └─ If (z_confirms_hedge AND |z|>0.08) OR fair>0.55:   │
│              └─ HEDGE SIGNAL -> Execute order                   │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. CHECK ENTRY                                                 │
│     └─ tau > 120?                                               │
│        └─ z_confirms_underdog?                                  │
│           └─ NOT z_confirms_favorite?                           │
│              └─ fav_dip?                                        │
│                 └─ und_price > 0.20?                            │
│                    └─ time_since_last > 10?                     │
│                       └─ ENTRY SIGNAL -> Execute order          │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
                        Wait for next tick
```

---

**Version:** 1.0  
**Date:** 2026-01-29
