"""
Feature Engine Module
======================

Computes real-time trading features from bars and tick data:
- CVD (Cumulative Volume Delta) + slopes
- RVOL (Relative Volume)
- Impulse (price move relative to historical volatility)
- Microprice (order book weighted price)
- Basis / Premium (futures vs spot spread)
- Anchored VWAP 15m + deviation in bps
- Orderbook imbalance + change_10s/30s + depth degradation
- Liquidations windows 30s/60s
- Gap detection (missing aggTrade)
- Absorption score 30s
- Quality mode (OK/DEGRADED/BAD)

All features are computed on-the-fly and available via snapshot().
"""

import logging
from collections import deque
from statistics import mean, median
from typing import Optional, Literal, TYPE_CHECKING

from collector.types import NormalizedEvent, SCHEMA_VERSION
from collector.utils_time import now_ms

if TYPE_CHECKING:
    from collector.metrics import Metrics
    from collector.market_context import MarketContextEngine
    from collector.context_klines_store import KlinesStore
    from collector.taapi.context_engine import TaapiContextEngine
    from collector.polymarket.book_store import PolymarketBookStore

logger = logging.getLogger(__name__)

# History sizes
BARS_5S_MAXLEN = 200
BARS_15S_MAXLEN = 100
BARS_1M_MAXLEN = 600  # 10 hours of 1m bars
CVD_HISTORY_MAXLEN = 120
RET_5S_HISTORY_MAXLEN = 120
RET_15S_HISTORY_MAXLEN = 40
RET_1M_HISTORY_MAXLEN = 600  # For impulse calculation

# Lookbacks for calculations
RVOL_5S_LOOKBACK = 60  # bars
RVOL_15S_LOOKBACK = 40  # bars
CVD_SLOPE_30S_BARS = 6  # 6 * 5s = 30s
CVD_SLOPE_60S_BARS = 12  # 12 * 5s = 60s

# Anchor constants
ANCHOR_WINDOW_MS = 900_000  # 15 minutes in ms
ANCHOR_MODES = ("AUTO_UTC", "MANUAL")

# Depth constants
DEPTH_THROTTLE_MS = 500  # Process depth max 2x per second
IMBALANCE_HISTORY_MAXLEN = 500  # ~30-60 seconds of history
DEPTH_STATS_WINDOW_MS = 10_000  # 10 seconds for drop ratio
DEGRADED_THRESHOLD = 0.80  # 80% drop rate → degraded
RECOVERY_THRESHOLD = 0.20  # 20% drop rate → recovered

# Liquidation constants
LIQ_HISTORY_MAXLEN = 5000  # ~60s at high activity

# Gap detection constants
GAP_THRESHOLD_MS = 5000  # 5 seconds without aggTrade

# Dev history for absorption
DEV_HISTORY_MAXLEN = 1000  # ~30-60 seconds

# Quality mode thresholds
LAG_BAD_THRESHOLD_MS = 3000  # p95 lag > 3s = BAD

# Context stale threshold
CONTEXT_STALE_THRESHOLD_SEC = 120  # 2 minutes

# Micro 1m constants
MICRO_1M_EMA_FAST = 9
MICRO_1M_EMA_SLOW = 21
MICRO_1M_TREND_THRESHOLD_BPS = 3  # 3 bps for trend detection
MICRO_1M_RVOL_LOOKBACK = 60  # bars for rvol
MICRO_1M_IMPULSE_LOOKBACK = 120  # bars for impulse median
MICRO_1M_SLOPE_BARS = 5  # 5 bars = 5 minutes


def current_utc_15m_bucket_start_ms(ts_ms: int) -> int:
    """
    Get the start of the current 15-minute UTC bucket.
    
    Args:
        ts_ms: Timestamp in milliseconds
    
    Returns:
        Bucket start timestamp (floored to 15-minute boundary)
    """
    return (ts_ms // ANCHOR_WINDOW_MS) * ANCHOR_WINDOW_MS


class FeatureEngine:
    """
    Real-time feature computation engine.
    
    Processes bars and tick events to compute trading features:
    - CVD and CVD slopes
    - RVOL (relative volume)
    - Impulse (relative price moves)
    - Microprice
    - Basis/Premium
    - Anchored VWAP 15m + deviation
    - Orderbook imbalance + changes
    - Liquidations windows
    - Gap detection
    - Absorption score
    - Quality mode
    """
    
    def __init__(
        self,
        symbol: str,
        topn: int = 10,
        depth_throttle_ms: int = DEPTH_THROTTLE_MS,
        market_context_engine: Optional["MarketContextEngine"] = None,
        klines_store: Optional["KlinesStore"] = None,
        taapi_context_engine: Optional["TaapiContextEngine"] = None,
    ):
        """
        Initialize feature engine.
        
        Args:
            symbol: Trading symbol (uppercase, e.g., "BTCUSDT")
            topn: Number of top levels for depth analysis
            depth_throttle_ms: Minimum ms between depth processing
            market_context_engine: Optional MarketContextEngine for HTF context
            klines_store: Optional KlinesStore for context age tracking
            taapi_context_engine: Optional TaapiContextEngine for TAAPI indicators
        """
        self.symbol = symbol
        self._topn = topn
        self._depth_throttle_ms = depth_throttle_ms
        self._market_context_engine = market_context_engine
        self._klines_store = klines_store
        self._taapi_context_engine = taapi_context_engine
        self._polymarket_book_store: Optional["PolymarketBookStore"] = None
        
        # Bar history
        self._futures_bars_5s: deque = deque(maxlen=BARS_5S_MAXLEN)
        self._futures_bars_15s: deque = deque(maxlen=BARS_15S_MAXLEN)
        self._futures_bars_1m: deque = deque(maxlen=BARS_1M_MAXLEN)
        self._spot_bars_5s: deque = deque(maxlen=BARS_5S_MAXLEN)
        self._spot_bars_15s: deque = deque(maxlen=BARS_15S_MAXLEN)
        self._spot_bars_1m: deque = deque(maxlen=BARS_1M_MAXLEN)
        
        # CVD (Cumulative Volume Delta)
        self._futures_cvd: float = 0.0
        self._spot_cvd: float = 0.0
        self._futures_cvd_history: deque = deque(maxlen=CVD_HISTORY_MAXLEN)
        self._spot_cvd_history: deque = deque(maxlen=CVD_HISTORY_MAXLEN)
        
        # Return history for impulse calculation
        self._futures_ret_5s_history: deque = deque(maxlen=RET_5S_HISTORY_MAXLEN)
        self._futures_ret_15s_history: deque = deque(maxlen=RET_15S_HISTORY_MAXLEN)
        self._futures_ret_1m_history: deque = deque(maxlen=RET_1M_HISTORY_MAXLEN)
        self._spot_ret_5s_history: deque = deque(maxlen=RET_5S_HISTORY_MAXLEN)
        self._spot_ret_15s_history: deque = deque(maxlen=RET_15S_HISTORY_MAXLEN)
        self._spot_ret_1m_history: deque = deque(maxlen=RET_1M_HISTORY_MAXLEN)
        
        # Tick-level data
        self._futures_mid: Optional[float] = None
        self._spot_mid: Optional[float] = None
        self._futures_microprice: Optional[float] = None
        self._spot_microprice: Optional[float] = None
        self._futures_mark_px: Optional[float] = None
        
        # Last update timestamps
        self._last_bar_ts: Optional[int] = None
        self._last_bookticker_ts: Optional[int] = None
        
        # ========== Anchored VWAP state ==========
        self._anchor_mode: str = "AUTO_UTC"
        self._anchor_time_ms: Optional[int] = None
        self._anchor_window_seconds: int = 900  # 15 minutes
        
        # AVWAP accumulators
        self._avwap_sum_pv: float = 0.0
        self._avwap_sum_v: float = 0.0
        
        # Computed AVWAP values
        self._avwap_15m: Optional[float] = None
        self._anchor_expired: bool = False
        
        # ========== Depth / Orderbook state ==========
        self._last_depth_processed_ms: int = 0
        self._imbalance: Optional[float] = None
        self._imbalance_ts_ms: Optional[int] = None
        self._imbalance_history: deque = deque(maxlen=IMBALANCE_HISTORY_MAXLEN)
        
        # Depth processing stats
        self._depth_processed_count: int = 0
        self._depth_dropped_count: int = 0
        
        # Rolling stats for degradation detection
        self._depth_events_history: deque = deque(maxlen=500)
        self._depth_degraded: bool = False
        self._depth_degraded_since_ms: Optional[int] = None
        
        # ========== Liquidations state ==========
        # History of (ts_recv_ms, side, qty)
        self._liq_events: deque = deque(maxlen=LIQ_HISTORY_MAXLEN)
        self._liq_qty_30s: float = 0.0
        self._liq_count_30s: int = 0
        self._liq_qty_60s: float = 0.0
        self._liq_count_60s: int = 0
        
        # ========== Gap detection state ==========
        self._last_futures_aggtrade_recv_ms: Optional[int] = None
        self._last_spot_aggtrade_recv_ms: Optional[int] = None
        self._gap_detected_futures: bool = False
        self._gap_detected_spot: bool = False
        
        # ========== Absorption score state ==========
        # Dev history for tracking dev changes
        self._dev_history: deque = deque(maxlen=DEV_HISTORY_MAXLEN)
        self._absorption_score_30s: Optional[float] = None
        
        # ========== BLOCK 2: Market Reference Tracker ==========
        # Tracks ref_px at window start and tau (time remaining)
        from collector.market_ref_tracker import MarketRefTracker
        self._market_ref_tracker = MarketRefTracker(window_len_sec=900)
        
        # ========== BLOCK 3: Volatility Estimator (S2 Multi-scale) ==========
        # Estimates sigma_fast/slow/blend from 1m bar closes
        from collector.volatility import VolEstimator
        from collector.config import settings
        self._vol_estimator = VolEstimator(
            fast_minutes=settings.VOL_FAST_MINUTES,
            slow_minutes=settings.VOL_SLOW_MINUTES,
            min_bars=settings.VOL_MIN_BARS,
        )
        
        # ========== BLOCK 5: Binance Spike Detector ==========
        # Detects 5s price spikes for timing signals
        from collector.spike_detector import BinanceSpikeDetector
        self._spike_detector = BinanceSpikeDetector(lookback=60, threshold=2.0)
        
        # ========== BLOCK 6: Polymarket Dip Detector ==========
        # Detects dips in UP/DOWN token prices
        from collector.pm_dip_detector import PolymarketDipDetector
        self._pm_dip_detector = PolymarketDipDetector(window_sec=60, threshold_bps=80)
        
        # ========== S4: Bias Model (slow directional context) ==========
        from collector.bias_model import BiasModel
        self._bias_model = BiasModel(
            slope_lookback=settings.BIAS_SLOPE_LOOKBACK,
            ema_fast=settings.BIAS_EMA_FAST,
            ema_slow=settings.BIAS_EMA_SLOW,
            up_threshold=settings.BIAS_UP_THRESHOLD,
            down_threshold=settings.BIAS_DOWN_THRESHOLD,
        )
        
        # ========== S1: Price Smoother (EMA 20s) ==========
        # Smooth BTC price for fair value calculations
        from collector.price_smoother import EmaSmoother
        from collector.config import settings
        self._price_smoother = EmaSmoother(ema_sec=settings.PRICE_EMA_SEC)
        
        # ========== S9: ROC Calculator for countertrend signals ==========
        from collector.roc_calculator import RocCalculator
        self._roc_calculator = RocCalculator(window_30s=30, window_60s=60)
        
        # ========== Event Recorder for backtesting ==========
        from typing import TYPE_CHECKING
        if TYPE_CHECKING:
            from collector.event_recorder import EventRecorder
        self._event_recorder: Optional["EventRecorder"] = None
        
        logger.info(
            "feature_engine_initialized",
            extra={
                "symbol": symbol,
                "anchor_mode": self._anchor_mode,
                "topn": topn,
                "depth_throttle_ms": depth_throttle_ms,
                "gap_threshold_ms": GAP_THRESHOLD_MS,
            },
        )
    
    # ========== Context Integration ==========
    
    def set_market_context_engine(self, engine: "MarketContextEngine") -> None:
        """
        Set the market context engine for HTF context integration.
        
        Args:
            engine: MarketContextEngine instance
        """
        self._market_context_engine = engine
        logger.info("feature_engine_context_engine_set")
    
    def set_klines_store(self, store: "KlinesStore") -> None:
        """
        Set the klines store for context age tracking.
        
        Args:
            store: KlinesStore instance
        """
        self._klines_store = store
        logger.info("feature_engine_klines_store_set")
    
    def set_taapi_context_engine(self, engine: "TaapiContextEngine") -> None:
        """
        Set the TAAPI context engine for indicator integration.
        
        Args:
            engine: TaapiContextEngine instance
        """
        self._taapi_context_engine = engine
        logger.info("feature_engine_taapi_context_set")
    
    def set_polymarket_book_store(self, store: "PolymarketBookStore") -> None:
        """
        Set the Polymarket book store for prediction market integration.
        
        Args:
            store: PolymarketBookStore instance
        """
        self._polymarket_book_store = store
        logger.info("feature_engine_polymarket_store_set")
    
    def set_event_recorder(self, recorder: "EventRecorder") -> None:
        """
        Set the EventRecorder for high-frequency logging.
        
        Args:
            recorder: EventRecorder instance
        """
        self._event_recorder = recorder
        logger.info("feature_engine_event_recorder_set")
    
    # ========== Liquidations Processing ==========
    
    def on_liquidation(self, event: NormalizedEvent) -> None:
        """
        Process a liquidation (forceOrder) event.
        
        Args:
            event: Normalized liquidation event (futures only)
        """
        if event.type != "liquidation" or event.market != "futures":
            return
        
        ts_now = event.ts_recv_ms or now_ms()
        data = event.data
        
        qty = data.get("qty", 0.0)
        side = data.get("side", "unknown")
        
        # Add to history
        self._liq_events.append((ts_now, side, qty))
        
        # Clean old events and recompute
        self._recompute_liquidations(ts_now)
    
    def _recompute_liquidations(self, ts_now: int) -> None:
        """
        Recompute liquidation metrics for 30s and 60s windows.
        """
        cutoff_60s = ts_now - 60_000
        cutoff_30s = ts_now - 30_000
        
        # Clean events older than 60s
        while self._liq_events and self._liq_events[0][0] < cutoff_60s:
            self._liq_events.popleft()
        
        # Reset counters
        qty_30s = 0.0
        count_30s = 0
        qty_60s = 0.0
        count_60s = 0
        
        for ts_ms, side, qty in self._liq_events:
            qty_60s += qty
            count_60s += 1
            
            if ts_ms >= cutoff_30s:
                qty_30s += qty
                count_30s += 1
        
        self._liq_qty_30s = qty_30s
        self._liq_count_30s = count_30s
        self._liq_qty_60s = qty_60s
        self._liq_count_60s = count_60s
    
    def _compute_liquidations_features(self) -> dict:
        """Compute liquidations features."""
        return {
            "qty_30s": round(self._liq_qty_30s, 4),
            "count_30s": self._liq_count_30s,
            "qty_60s": round(self._liq_qty_60s, 4),
            "count_60s": self._liq_count_60s,
        }
    
    # ========== Gap Detection ==========
    
    def update_aggtrade_timestamp(self, market: str, ts_recv_ms: int) -> None:
        """
        Update last aggTrade receive timestamp.
        
        Args:
            market: "futures" or "spot"
            ts_recv_ms: Receive timestamp in milliseconds
        """
        if market == "futures":
            self._last_futures_aggtrade_recv_ms = ts_recv_ms
        elif market == "spot":
            self._last_spot_aggtrade_recv_ms = ts_recv_ms
    
    def _check_gaps(self, ts_now: int) -> None:
        """
        Check for gaps in aggTrade streams.
        """
        # Futures gap
        if self._last_futures_aggtrade_recv_ms is not None:
            gap_ms = ts_now - self._last_futures_aggtrade_recv_ms
            self._gap_detected_futures = gap_ms > GAP_THRESHOLD_MS
        else:
            self._gap_detected_futures = False
        
        # Spot gap
        if self._last_spot_aggtrade_recv_ms is not None:
            gap_ms = ts_now - self._last_spot_aggtrade_recv_ms
            self._gap_detected_spot = gap_ms > GAP_THRESHOLD_MS
        else:
            self._gap_detected_spot = False
    
    # ========== Absorption Score ==========
    
    def _update_dev_history(self, ts_ms: int) -> None:
        """
        Update dev_from_avwap_bps history for absorption calculation.
        """
        dev_bps = self._compute_dev_from_avwap_bps()
        if dev_bps is not None:
            self._dev_history.append((ts_ms, dev_bps))
    
    def _compute_absorption_score(self) -> Optional[float]:
        """
        Compute absorption score (0-100).
        
        Based on:
        - Panic indicators (rvol, impulse, dev)
        - Stabilization indicators (imbalance change, cvd slope)
        """
        # Get current values
        rvol_5s = self._compute_rvol(self._futures_bars_5s, RVOL_5S_LOOKBACK)
        impulse_5s = self._compute_impulse(self._futures_ret_5s_history)
        dev_bps = self._compute_dev_from_avwap_bps()
        cvd_slope_30s = self._compute_cvd_slope(self._futures_cvd_history, CVD_SLOPE_30S_BARS)
        imb_change_10s = self._compute_imbalance_change(10_000)
        
        # Panic score (0-3)
        panic = 0
        if rvol_5s is not None and rvol_5s >= 2.5:
            panic += 1
        if impulse_5s is not None and impulse_5s >= 2.0:
            panic += 1
        if dev_bps is not None and abs(dev_bps) >= 10:
            panic += 1
        
        # Need at least some panic for absorption to be relevant
        if panic < 1:
            return None
        
        # Stabilization score (0-3)
        stab = 0
        
        # Imbalance change counter-move
        if imb_change_10s is not None and abs(imb_change_10s) > 0.05:
            stab += 1
        
        # CVD slope present and dev decreasing
        if cvd_slope_30s is not None and abs(cvd_slope_30s) > 0:
            # Check if dev is decreasing (need history)
            if self._is_dev_decreasing():
                stab += 1
        
        # Imbalance suggests absorption
        if self._imbalance is not None and abs(self._imbalance) > 0.3:
            stab += 1
        
        # Compute final score (panic contributes 30, stab contributes 20 each)
        score = min(panic * 30 + stab * 20, 100)
        
        return score
    
    def _is_dev_decreasing(self) -> bool:
        """
        Check if abs(dev_from_avwap_bps) is decreasing over last 15s.
        """
        if len(self._dev_history) < 3:
            return False
        
        ts_now = now_ms()
        target_ts = ts_now - 15_000
        
        # Find dev value closest to 15s ago
        past_dev = None
        min_diff = float('inf')
        
        for ts_ms, dev in self._dev_history:
            diff = abs(ts_ms - target_ts)
            if diff < min_diff:
                min_diff = diff
                past_dev = dev
        
        if past_dev is None or min_diff > 5000:
            return False
        
        current_dev = self._compute_dev_from_avwap_bps()
        if current_dev is None:
            return False
        
        # Dev is decreasing if absolute value is smaller now
        return abs(current_dev) < abs(past_dev)
    
    # ========== Quality Mode ==========
    
    def compute_quality(self, metrics_snapshot: Optional[dict] = None) -> dict:
        """
        Compute quality flags and mode.
        
        Args:
            metrics_snapshot: Optional metrics snapshot from Metrics class
        
        Returns:
            Quality dictionary with mode and flags
        """
        ts_now = now_ms()
        
        # Check gaps
        self._check_gaps(ts_now)
        
        # Default values
        ws_futures = True
        ws_spot = True
        lag_p95 = {"futures": {}, "spot": {}}
        
        # Extract from metrics if available
        if metrics_snapshot:
            ws_connected = metrics_snapshot.get("ws_connected", {})
            ws_futures = ws_connected.get("futures", True)
            ws_spot = ws_connected.get("spot", True)
            
            lag_p95 = {
                "futures": metrics_snapshot.get("lag_p95_ms", {}).get("futures", {}),
                "spot": metrics_snapshot.get("lag_p95_ms", {}).get("spot", {}),
            }
        
        # ========== Context readiness (Klines) ==========
        context_ready = False
        context_age_sec: Optional[float] = None
        context_stale = False
        
        if self._klines_store:
            # Check if we have klines for futures 1m
            last_kline = self._klines_store.get_last_kline("futures", "1m")
            if last_kline:
                context_ready = True
                # Context age is based on last kline open time (how old is the latest data)
                # For a 1m kline, this shows how fresh our data is
                context_age_sec = (ts_now - last_kline.open_time_ms) / 1000.0
                
                # Check if stale (> 120 seconds = 2 full 1m bars without update)
                if context_age_sec > CONTEXT_STALE_THRESHOLD_SEC:
                    context_stale = True
        
        # ========== TAAPI Context readiness ==========
        taapi_ready = False
        taapi_age_sec: Optional[float] = None
        taapi_stale = False
        
        if self._taapi_context_engine:
            # Get age from store via context engine
            taapi_store = getattr(self._taapi_context_engine, '_store', None)
            if taapi_store and taapi_store.is_ready():
                taapi_ready = True
                # Use 1m TF age as the freshest indicator
                taapi_age_sec = taapi_store.get_age_sec("1m")
                if taapi_age_sec and taapi_age_sec > CONTEXT_STALE_THRESHOLD_SEC:
                    taapi_stale = True
        
        # ========== Veto reasons ==========
        veto_reasons = []
        
        # Determine mode
        mode = "OK"
        
        # BAD conditions
        if not ws_futures:
            mode = "BAD"
            veto_reasons.append("ws_futures_disconnected")
        elif lag_p95.get("futures", {}).get("aggTrade", 0) > LAG_BAD_THRESHOLD_MS:
            mode = "BAD"
            veto_reasons.append("lag_too_high")
        
        # DEGRADED conditions (if not already BAD)
        if mode == "OK":
            if not ws_spot:
                mode = "DEGRADED"
                veto_reasons.append("ws_spot_disconnected")
            elif self._depth_degraded:
                mode = "DEGRADED"
                veto_reasons.append("depth_degraded")
            elif self._gap_detected_futures:
                mode = "DEGRADED"
                veto_reasons.append("gap_detected_futures")
        
        # Context gating (Klines)
        if not context_ready:
            if mode == "OK":
                mode = "DEGRADED"
            veto_reasons.append("context_not_ready")
        elif context_stale:
            if mode == "OK":
                mode = "DEGRADED"
            veto_reasons.append("context_stale")
        
        # TAAPI gating
        if not taapi_ready:
            if mode == "OK":
                mode = "DEGRADED"
            veto_reasons.append("taapi_not_ready")
        elif taapi_stale:
            if mode == "OK":
                mode = "DEGRADED"
            veto_reasons.append("taapi_stale")
        
        # ========== Polymarket readiness ==========
        polymarket_ws = False
        polymarket_age_sec: Optional[float] = None
        polymarket_stale = False
        
        if self._polymarket_book_store:
            snap = self._polymarket_book_store.snapshot()
            polymarket_ws = snap.get("connected", False)
            polymarket_age_sec = snap.get("age_sec")
            
            # Check if stale (> 5 seconds = data too old)
            if polymarket_age_sec is not None and polymarket_age_sec > 5.0:
                polymarket_stale = True
            
            # Polymarket gating - if enabled but not connected or stale
            if not polymarket_ws:
                if mode == "OK":
                    mode = "DEGRADED"
                veto_reasons.append("polymarket_ws_disconnected")
            elif polymarket_stale:
                if mode == "OK":
                    mode = "DEGRADED"
                veto_reasons.append("polymarket_stale")
        
        # ========== BLOCK 1: Trade Mode (Polymarket executability) ==========
        # This is separate from data quality mode.
        # trade_mode focuses on: can we actually execute a slice on Polymarket?
        
        trade_mode = "OK"
        trade_reasons: list[str] = []
        
        from collector.config import settings
        from collector.polymarket.normalize_updown import normalize_polymarket_updown
        
        if self._polymarket_book_store:
            pm_snap = self._polymarket_book_store.snapshot()
            
            # Normalize to UP/DOWN to get slice_ok and spread
            updown = normalize_polymarket_updown(
                pm_snap,
                settings.POLY_UP_IS_YES,
                settings.MIN_TOP1_USD,
                settings.MIN_TOP3_USD,
                settings.EDGE_BUFFER_BPS,
            )
            
            up = updown.get("up") or {}
            down = updown.get("down") or {}
            
            # Rule 1: Polymarket stale → BAD
            age_sec = updown.get("age_sec")
            if age_sec is not None and age_sec > settings.POLYMARKET_STALE_THRESHOLD_SEC:
                trade_mode = "BAD"
                trade_reasons.append("polymarket_stale")
            
            # Rule 2: Both sides have slice_ok=False → BAD (can't execute anything)
            slice_ok_up = up.get("slice_ok", False)
            slice_ok_down = down.get("slice_ok", False)
            
            if not slice_ok_up and not slice_ok_down:
                trade_mode = "BAD"
                trade_reasons.append("polymarket_no_slice")
            
            # Rule 3: Check spreads
            spread_up = up.get("spread_bps")
            spread_down = down.get("spread_bps")
            max_spread = max(
                spread_up if spread_up is not None else 0,
                spread_down if spread_down is not None else 0
            )
            
            # Spread > MAX_SPREAD_BPS_BAD → BAD
            if max_spread > settings.MAX_SPREAD_BPS_BAD:
                trade_mode = "BAD"
                trade_reasons.append("spread_too_wide")
            # Spread > MAX_SPREAD_BPS_DEGRADED → DEGRADED (if not already BAD)
            elif max_spread > settings.MAX_SPREAD_BPS_DEGRADED:
                if trade_mode == "OK":
                    trade_mode = "DEGRADED"
                trade_reasons.append("spread_high")
        else:
            # No Polymarket store = can't trade
            trade_mode = "BAD"
            trade_reasons.append("polymarket_disabled")
        
        return {
            "mode": mode,
            "ws_futures": ws_futures,
            "ws_spot": ws_spot,
            "lag_p95_ms": lag_p95,
            "gap_detected": {
                "futures": self._gap_detected_futures,
                "spot": self._gap_detected_spot,
            },
            "depth_degraded": self._depth_degraded,
            "context_ready": context_ready,
            "context_age_sec": round(context_age_sec, 1) if context_age_sec is not None else None,
            "context_stale": context_stale,
            "taapi_ready": taapi_ready,
            "taapi_age_sec": round(taapi_age_sec, 1) if taapi_age_sec is not None else None,
            "taapi_stale": taapi_stale,
            "polymarket_ws": polymarket_ws,
            "polymarket_age_sec": round(polymarket_age_sec, 1) if polymarket_age_sec is not None else None,
            "polymarket_stale": polymarket_stale,
            "degrade_reasons": veto_reasons if veto_reasons else [],
            # BLOCK 1: Trade executability mode
            "trade_mode": trade_mode,
            "trade_reasons": trade_reasons,
        }
    
    # ========== Depth Processing ==========
    
    def on_depth(self, event: NormalizedEvent) -> None:
        """
        Process a depth (orderbook) event.
        """
        if event.type != "depth" or event.market != "futures":
            return
        
        ts_now = event.ts_recv_ms or now_ms()
        
        # Throttling check
        time_since_last = ts_now - self._last_depth_processed_ms
        if time_since_last < self._depth_throttle_ms:
            self._depth_dropped_count += 1
            self._depth_events_history.append((ts_now, True))
            self._check_degradation(ts_now)
            return
        
        # Process depth
        self._last_depth_processed_ms = ts_now
        self._depth_processed_count += 1
        self._depth_events_history.append((ts_now, False))
        
        data = event.data
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        
        # Compute imbalance
        bid_sum = sum(b.get("qty", 0) for b in bids[:self._topn])
        ask_sum = sum(a.get("qty", 0) for a in asks[:self._topn])
        
        total = bid_sum + ask_sum
        if total > 0:
            self._imbalance = (bid_sum - ask_sum) / total
        else:
            self._imbalance = None
        
        self._imbalance_ts_ms = event.ts_event_ms or ts_now
        
        if self._imbalance is not None:
            self._imbalance_history.append((ts_now, self._imbalance))
        
        self._check_degradation(ts_now)
    
    def _check_degradation(self, ts_now: int) -> None:
        """Check and update depth degradation status."""
        cutoff = ts_now - DEPTH_STATS_WINDOW_MS
        while self._depth_events_history and self._depth_events_history[0][0] < cutoff:
            self._depth_events_history.popleft()
        
        if not self._depth_events_history:
            return
        
        total = len(self._depth_events_history)
        dropped = sum(1 for _, was_dropped in self._depth_events_history if was_dropped)
        
        if total < 5:
            return
        
        drop_ratio = dropped / total
        
        if not self._depth_degraded:
            if drop_ratio > DEGRADED_THRESHOLD:
                self._depth_degraded = True
                self._depth_degraded_since_ms = ts_now
                logger.warning(
                    "depth_degraded",
                    extra={"drop_ratio": round(drop_ratio, 2), "dropped": dropped, "total": total},
                )
        else:
            if drop_ratio < RECOVERY_THRESHOLD:
                recovery_window = 30_000
                if self._depth_degraded_since_ms:
                    time_degraded = ts_now - self._depth_degraded_since_ms
                    if time_degraded > recovery_window:
                        self._depth_degraded = False
                        self._depth_degraded_since_ms = None
                        logger.info("depth_recovered", extra={"drop_ratio": round(drop_ratio, 2)})
    
    def _compute_imbalance_change(self, lookback_ms: int) -> Optional[float]:
        """Compute imbalance change over lookback period."""
        if self._imbalance is None or not self._imbalance_history:
            return None
        
        ts_now = now_ms()
        target_ts = ts_now - lookback_ms
        
        past_imbalance = None
        min_diff = float('inf')
        
        for ts_ms, imb in self._imbalance_history:
            diff = abs(ts_ms - target_ts)
            if diff < min_diff:
                min_diff = diff
                past_imbalance = imb
        
        if past_imbalance is None or min_diff > 2000:
            return None
        
        return round(self._imbalance - past_imbalance, 4)
    
    def _compute_orderbook_features(self) -> dict:
        """Compute orderbook/depth features."""
        return {
            "topn": self._topn,
            "imbalance": round(self._imbalance, 4) if self._imbalance is not None else None,
            "imbalance_ts_ms": self._imbalance_ts_ms,
            "imbalance_change_10s": self._compute_imbalance_change(10_000),
            "imbalance_change_30s": self._compute_imbalance_change(30_000),
            "depth_processed_count": self._depth_processed_count,
            "depth_dropped_count": self._depth_dropped_count,
            "depth_degraded": self._depth_degraded,
        }
    
    def get_depth_summary(self) -> dict:
        """Get depth state summary for periodic logging."""
        return {
            "imbalance": round(self._imbalance, 4) if self._imbalance is not None else None,
            "change_10s": self._compute_imbalance_change(10_000),
            "change_30s": self._compute_imbalance_change(30_000),
            "processed": self._depth_processed_count,
            "dropped": self._depth_dropped_count,
            "depth_degraded": self._depth_degraded,
        }
    
    # ========== Anchor Control Methods ==========
    
    def set_anchor_mode(self, mode: str) -> dict:
        """Set anchor mode."""
        if mode not in ANCHOR_MODES:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {ANCHOR_MODES}")
        
        old_mode = self._anchor_mode
        self._anchor_mode = mode
        
        logger.info("anchor_mode_changed", extra={"old_mode": old_mode, "new_mode": mode})
        
        if mode == "AUTO_UTC":
            ts = now_ms()
            bucket_start = current_utc_15m_bucket_start_ms(ts)
            if self._anchor_time_ms != bucket_start:
                self._reset_anchor(bucket_start)
        
        return self.get_anchor_snapshot()
    
    def anchor_now(self, ts_ms: Optional[int] = None) -> dict:
        """Set anchor to current time (MANUAL mode only)."""
        if self._anchor_mode != "MANUAL":
            raise RuntimeError("anchor_now() only works in MANUAL mode")
        
        if ts_ms is None:
            ts_ms = now_ms()
        
        self._reset_anchor(ts_ms)
        logger.info("anchor_set_manually", extra={"anchor_time_ms": ts_ms})
        
        return self.get_anchor_snapshot()
    
    def _reset_anchor(self, anchor_time_ms: int) -> None:
        """Reset anchor and AVWAP accumulators."""
        self._anchor_time_ms = anchor_time_ms
        self._avwap_sum_pv = 0.0
        self._avwap_sum_v = 0.0
        self._avwap_15m = None
        self._anchor_expired = False
        
        logger.info("anchor_reset", extra={"anchor_time_ms": anchor_time_ms, "mode": self._anchor_mode})
    
    def _check_auto_anchor(self, ts_ms: int) -> None:
        """Check and update anchor in AUTO_UTC mode."""
        if self._anchor_mode != "AUTO_UTC":
            return
        
        bucket_start = current_utc_15m_bucket_start_ms(ts_ms)
        
        if self._anchor_time_ms is None:
            self._reset_anchor(bucket_start)
            return
        
        if bucket_start != current_utc_15m_bucket_start_ms(self._anchor_time_ms):
            logger.info(
                "anchor_auto_rotated",
                extra={"old_anchor_ms": self._anchor_time_ms, "new_anchor_ms": bucket_start, "avwap_at_rotation": self._avwap_15m},
            )
            self._reset_anchor(bucket_start)
    
    def get_anchor_snapshot(self) -> dict:
        """Get current anchor state."""
        dev_from_avwap_bps = self._compute_dev_from_avwap_bps()
        
        return {
            "mode": self._anchor_mode,
            "anchor_time_ms": self._anchor_time_ms,
            "avwap_15m": round(self._avwap_15m, 2) if self._avwap_15m else None,
            "dev_from_avwap_bps": dev_from_avwap_bps,
            "anchor_expired": self._anchor_expired,
        }
    
    def _compute_dev_from_avwap_bps(self) -> Optional[float]:
        """Compute deviation from AVWAP in basis points."""
        if self._avwap_15m is None or self._avwap_15m <= 0:
            return None
        
        price_ref = self._futures_microprice or self._futures_mid
        
        if price_ref is None:
            return None
        
        dev_bps = (price_ref - self._avwap_15m) / self._avwap_15m * 10000
        return round(dev_bps, 2)
    
    # ========== Event Handlers ==========
    
    def on_bar(self, bar: dict) -> None:
        """Process a closed bar."""
        market = bar.get("market")
        tf = bar.get("tf")
        delta_vol = bar.get("delta_vol", 0.0)
        open_px = bar.get("open")
        close_px = bar.get("close")
        high_px = bar.get("high")
        low_px = bar.get("low")
        volume = bar.get("volume_total", 0.0)
        t_open_ms = bar.get("t_open_ms", 0)
        t_close_ms = bar.get("t_close_ms", 0)
        
        # Record kline event for backtesting
        if self._event_recorder and open_px and close_px:
            self._event_recorder.record_bn_kline(
                market=market,
                tf=tf,
                open_px=open_px,
                high=high_px or max(open_px, close_px),
                low=low_px or min(open_px, close_px),
                close=close_px,
                volume=volume,
            )
        
        if market == "futures":
            if tf == "5s":
                self._futures_cvd += delta_vol
                self._futures_cvd_history.append(self._futures_cvd)
                
                self._futures_bars_5s.append({
                    "open": open_px,
                    "close": close_px,
                    "volume": volume,
                    "delta": delta_vol,
                    "t_open_ms": t_open_ms,
                    "t_close_ms": t_close_ms,
                })
                
                if open_px and open_px > 0:
                    ret = (close_px - open_px) / open_px
                    self._futures_ret_5s_history.append(ret)
                    
                    # BLOCK 5: Update spike detector with 5s return
                    self._spike_detector.update(ret)
                
                self._update_avwap(bar)
                
                # Update dev history for absorption
                self._update_dev_history(t_close_ms)
            
            elif tf == "15s":
                self._futures_bars_15s.append({
                    "open": open_px,
                    "close": close_px,
                    "volume": volume,
                    "delta": delta_vol,
                })
                
                if open_px and open_px > 0:
                    ret = (close_px - open_px) / open_px
                    self._futures_ret_15s_history.append(ret)
            
            elif tf == "1m":
                self._futures_bars_1m.append({
                    "open": open_px,
                    "close": close_px,
                    "volume": volume,
                    "delta": delta_vol,
                    "t_open_ms": t_open_ms,
                    "t_close_ms": t_close_ms,
                })
                
                if open_px and open_px > 0:
                    ret = (close_px - open_px) / open_px
                    self._futures_ret_1m_history.append(ret)
                
                # BLOCK 3: Update volatility estimator with 1m close
                if close_px and close_px > 0:
                    self._vol_estimator.update_1m_close(close_px, t_close_ms)
                    # S4: Update bias model with 1m close
                    self._bias_model.update_1m(close_px, t_close_ms)
        
        elif market == "spot":
            if tf == "5s":
                self._spot_cvd += delta_vol
                self._spot_cvd_history.append(self._spot_cvd)
                
                self._spot_bars_5s.append({
                    "open": open_px,
                    "close": close_px,
                    "volume": volume,
                    "delta": delta_vol,
                })
                
                if open_px and open_px > 0:
                    ret = (close_px - open_px) / open_px
                    self._spot_ret_5s_history.append(ret)
            
            elif tf == "15s":
                self._spot_bars_15s.append({
                    "open": open_px,
                    "close": close_px,
                    "volume": volume,
                    "delta": delta_vol,
                })
                
                if open_px and open_px > 0:
                    ret = (close_px - open_px) / open_px
                    self._spot_ret_15s_history.append(ret)
            
            elif tf == "1m":
                self._spot_bars_1m.append({
                    "open": open_px,
                    "close": close_px,
                    "volume": volume,
                    "delta": delta_vol,
                    "t_open_ms": t_open_ms,
                    "t_close_ms": t_close_ms,
                })
                
                if open_px and open_px > 0:
                    ret = (close_px - open_px) / open_px
                    self._spot_ret_1m_history.append(ret)
        
        self._last_bar_ts = now_ms()
    
    def _update_avwap(self, bar: dict) -> None:
        """Update AVWAP accumulators from a futures 5s bar."""
        t_open_ms = bar.get("t_open_ms", 0)
        t_close_ms = bar.get("t_close_ms", 0)
        close_px = bar.get("close")
        volume = bar.get("volume_total", 0.0)
        
        self._check_auto_anchor(t_close_ms)
        
        if self._anchor_time_ms is None:
            return
        
        if t_close_ms <= self._anchor_time_ms:
            return
        
        anchor_end_ms = self._anchor_time_ms + ANCHOR_WINDOW_MS
        
        if t_open_ms >= anchor_end_ms:
            if self._anchor_mode == "AUTO_UTC":
                self._check_auto_anchor(t_close_ms)
            else:
                self._anchor_expired = True
            return
        
        if close_px and close_px > 0 and volume > 0:
            self._avwap_sum_pv += close_px * volume
            self._avwap_sum_v += volume
            
            if self._avwap_sum_v > 0:
                self._avwap_15m = self._avwap_sum_pv / self._avwap_sum_v
    
    def on_bookticker(self, event: NormalizedEvent) -> None:
        """Process a bookTicker event."""
        if event.type != "bookTicker":
            return
        
        data = event.data
        market = event.market
        
        bid_px = data.get("bid_px")
        bid_qty = data.get("bid_qty")
        ask_px = data.get("ask_px")
        ask_qty = data.get("ask_qty")
        mid_px = data.get("mid_px")
        
        if market == "futures":
            if mid_px is not None:
                self._futures_mid = mid_px
            
            if all(v is not None and v > 0 for v in [bid_px, bid_qty, ask_px, ask_qty]):
                total_qty = bid_qty + ask_qty
                if total_qty > 0:
                    self._futures_microprice = (ask_px * bid_qty + bid_px * ask_qty) / total_qty
        
        elif market == "spot":
            if mid_px is not None:
                self._spot_mid = mid_px
            
            if all(v is not None and v > 0 for v in [bid_px, bid_qty, ask_px, ask_qty]):
                total_qty = bid_qty + ask_qty
                if total_qty > 0:
                    self._spot_microprice = (ask_px * bid_qty + bid_px * ask_qty) / total_qty
        
        ts = now_ms()
        self._last_bookticker_ts = ts
        
        # Record bn_ticker (throttled to ~5/sec for futures only)
        if self._event_recorder and market == "futures" and mid_px is not None:
            if not hasattr(self, "_last_bn_ticker_record_ts"):
                self._last_bn_ticker_record_ts = 0
            if ts - self._last_bn_ticker_record_ts >= 200:  # Every 200ms
                self._event_recorder.record_bn_ticker(
                    market=market,
                    bid=bid_px or 0,
                    ask=ask_px or 0,
                    mid=mid_px,
                )
                self._last_bn_ticker_record_ts = ts
    
    def on_markprice(self, event: NormalizedEvent) -> None:
        """Process a markPrice event."""
        if event.type != "markPrice":
            return
        
        data = event.data
        mark_px = data.get("mark_px")
        
        if mark_px is not None:
            self._futures_mark_px = mark_px
    
    # ========== Snapshot Methods ==========
    
    def snapshot(self, metrics_snapshot: Optional[dict] = None) -> dict:
        """
        Compute and return current feature snapshot.
        
        Args:
            metrics_snapshot: Optional metrics snapshot for quality computation
        
        Returns:
            Dictionary with all computed features.
        """
        ts = now_ms()
        
        # Check auto anchor on snapshot
        self._check_auto_anchor(ts)
        
        # Check gaps
        self._check_gaps(ts)
        
        # Recompute liquidations
        self._recompute_liquidations(ts)
        
        # Compute absorption score
        self._absorption_score_30s = self._compute_absorption_score()
        
        # Get context from market context engine if available
        context = None
        if self._market_context_engine:
            context = self._market_context_engine.snapshot()
        
        # Get TAAPI context if available
        taapi_context = None
        if self._taapi_context_engine:
            taapi_context = self._taapi_context_engine.snapshot()
        
        # Compute polymarket features and UP/DOWN mapping
        pm_features = self._compute_polymarket_features()
        pm_up_down = self._compute_polymarket_up_down(pm_features)
        pm_meta = self._compute_polymarket_meta()
        
        # ========== BLOCK 2: Market Reference Tracker ==========
        # Roll window if needed and set ref_px from current futures price
        self._market_ref_tracker.maybe_roll_window(ts)
        
        # Use futures mid or mark_px as reference price
        s_now_raw = self._futures_mid or self._futures_mark_px
        if s_now_raw is not None:
            self._market_ref_tracker.set_ref_if_missing(s_now_raw, ts)
        
        # ========== S1: Update price smoother ==========
        s_now_smooth = None
        if s_now_raw is not None:
            s_now_smooth = self._price_smoother.update(s_now_raw, ts)
        
        # ========== S9: Update ROC calculator ==========
        if s_now_raw is not None:
            self._roc_calculator.update(s_now_raw, ts)
        
        # Get market_ref snapshot and add S_now fields
        market_ref = self._market_ref_tracker.snapshot(ts)
        # Keep S_now for backward compatibility (uses raw)
        market_ref["S_now"] = round(s_now_raw, 2) if s_now_raw else None
        # S1: Add raw and smooth prices
        market_ref["S_now_raw"] = round(s_now_raw, 2) if s_now_raw else None
        market_ref["S_now_smooth"] = round(s_now_smooth, 2) if s_now_smooth else None
        market_ref["smooth_last_update_ms"] = self._price_smoother.last_ts_ms
        # Compute smooth age for diagnostics
        if self._price_smoother.last_ts_ms:
            market_ref["smooth_age_sec"] = round((ts - self._price_smoother.last_ts_ms) / 1000, 1)
        else:
            market_ref["smooth_age_sec"] = None
        
        # ========== BLOCK 3: Volatility Estimator (S2 Multi-scale) ==========
        # Get rvol_5s for blend weight calculation
        rvol_5s = self._compute_rvol(self._futures_bars_5s, RVOL_5S_LOOKBACK)
        vol_snap = self._vol_estimator.snapshot(rvol_5s=rvol_5s)
        
        # ========== BLOCK 4: Fair Model v2 (S3 FAST + SMOOTH) ==========
        from collector.fair_model import compute_fair_updown_open
        import math
        
        ref_px = market_ref.get("ref_px")
        tau_sec = market_ref.get("tau_sec")
        S_raw = market_ref.get("S_now_raw") or market_ref.get("S_now")
        S_smooth = market_ref.get("S_now_smooth")
        
        # Get sigma values from vol_snap (S2 multi-scale)
        sig_fast = vol_snap.get("sigma_fast_15m") or vol_snap.get("sigma_15m")
        sig_blend = vol_snap.get("sigma_blend_15m") or sig_fast
        n_bars = vol_snap.get("n_bars", 0)
        n_returns = vol_snap.get("n_returns", 0)
        last_vol_update_ms = vol_snap.get("last_update_ms")
        vol_reason = vol_snap.get("reason", "")
        
        # Helper to check inputs readiness
        def _ready_inputs(S, sig):
            if ref_px is None:
                return False, "ref_missing"
            if S is None:
                return False, "s_missing"
            if tau_sec is None or tau_sec <= 0:
                return False, "tau_missing"
            if sig is None:
                # Check if warmup
                if n_bars < self._vol_estimator.min_bars:
                    return False, f"sigma_warmup({n_bars}/{self._vol_estimator.min_bars})"
                return False, "sigma_missing"
            if isinstance(sig, float) and (math.isnan(sig) or math.isinf(sig)):
                return False, "sigma_nan"
            if sig <= 0:
                return False, "sigma_zero"
            return True, "ok"
        
        # Compute FAST fair (S_raw + sigma_fast)
        ok_fast, reason_fast = _ready_inputs(S_raw, sig_fast)
        fair_fast = {"up": None, "down": None, "z": None}
        if ok_fast:
            try:
                up, down, z = compute_fair_updown_open(ref_px, S_raw, tau_sec, sig_fast, drift=0.0)
                fair_fast = {"up": up, "down": down, "z": z}
            except Exception as e:
                ok_fast = False
                reason_fast = f"exception:{str(e)[:20]}"
        
        # Compute SMOOTH fair (S_smooth + sigma_blend)
        ok_smooth, reason_smooth = _ready_inputs(S_smooth, sig_blend)
        fair_smooth = {"up": None, "down": None, "z": None}
        if ok_smooth:
            try:
                up, down, z = compute_fair_updown_open(ref_px, S_smooth, tau_sec, sig_blend, drift=0.0)
                fair_smooth = {"up": up, "down": down, "z": z}
            except Exception as e:
                ok_smooth = False
                reason_smooth = f"exception:{str(e)[:20]}"
        
        # Expand sigma_missing reason with vol_reason
        if "sigma_missing" in reason_fast and vol_reason:
            reason_fast = vol_reason
        if "sigma_missing" in reason_smooth and vol_reason:
            reason_smooth = vol_reason
        
        # Build fair_status for diagnostics
        fair_status = {
            "fast_ready": ok_fast,
            "smooth_ready": ok_smooth,
            "fast_reason": reason_fast,
            "smooth_reason": reason_smooth,
            "ref_px": ref_px,
            "tau_sec": round(tau_sec, 1) if tau_sec else None,
            "S_now_raw": round(S_raw, 2) if S_raw else None,
            "S_now_smooth": round(S_smooth, 2) if S_smooth else None,
            "sigma_fast_15m": sig_fast,
            "sigma_blend_15m": sig_blend,
            "n_bars": n_bars,
            "n_returns": n_returns,
            "last_vol_update_ms": last_vol_update_ms,
            # Legacy fields for backward compatibility
            "ready": ok_fast,
            "reason": reason_fast,
            "S_now": round(S_raw, 2) if S_raw else None,
            "sigma_15m": sig_fast,
        }
        
        # Build fair_snap with fast/smooth + legacy
        fair_snap = {
            "fast": fair_fast,
            "smooth": fair_smooth,
            "status": fair_status,
            # Legacy fields (= FAST for backward compatibility)
            "fair_up": fair_fast["up"],
            "fair_down": fair_fast["down"],
            "z_score": fair_fast["z"],
        }
        
        # ========== BLOCK 5: Spike Detector ==========
        spikes_snap = self._spike_detector.snapshot()
        
        # ========== BLOCK 6: PM Dip Detector ==========
        # Update dip detector with current UP/DOWN prices
        if pm_up_down:
            up_mid = pm_up_down.get("up", {}).get("mid")
            down_mid = pm_up_down.get("down", {}).get("mid")
            self._pm_dip_detector.update_both(up_mid, down_mid, ts)
        
        pm_dips_snap = self._pm_dip_detector.snapshot()
        
        # ========== S4: Bias Model (with TAAPI integration) ==========
        # Pass taapi_context to BiasModel for hybrid bias calculation
        bias_snap = self._bias_model.snapshot(taapi_context=taapi_context)
        
        result = {
            "schema_version": SCHEMA_VERSION,
            "symbol": self.symbol,
            "ts_ms": ts,
            
            "futures": self._compute_futures_features(),
            "spot": self._compute_spot_features(),
            "basis": self._compute_basis_features(),
            "anchor": self.get_anchor_snapshot(),
            "orderbook": self._compute_orderbook_features(),
            "liquidations": self._compute_liquidations_features(),
            "absorption_score_30s": self._absorption_score_30s,
            "micro_1m": self._compute_micro_1m_section(),
            "context": context,
            "taapi_context": taapi_context,
            "polymarket": pm_features,
            "polymarket_meta": pm_meta,
            "polymarket_up_down": pm_up_down,
            "market_ref": market_ref,
            "vol": vol_snap,
            "fair": fair_snap,
            "fair_status": fair_status,
            "spikes": spikes_snap,
            "pm_dips": pm_dips_snap,
            "bias": bias_snap,
            "quality": self.compute_quality(metrics_snapshot),
            
            # ========== S9: ROC (Rate of Change) ==========
            "roc": self._roc_calculator.snapshot(),
            "countertrend": self._compute_countertrend_signal(fair_snap, fair_status),
        }
        
        # Record features for backtesting (throttled to avoid too much data)
        if self._event_recorder:
            self._event_recorder.record_features(result)
        
        return result
    
    def _compute_countertrend_signal(self, fair_snap: dict, fair_status: dict) -> dict:
        """
        Compute countertrend signal based on Z-score and ROC.
        
        Countertrend logic:
        - Z > 0 (BTC above ref) + ROC < -2bps → BUY_DOWN
        - Z < 0 (BTC below ref) + ROC > +2bps → BUY_UP
        
        Returns:
            Dictionary with countertrend signal info
        """
        # Get Z-score from fair
        z_score = fair_snap.get("z_score")
        if z_score is None:
            # Try fast z
            fast = fair_snap.get("fast", {})
            z_score = fast.get("z")
        
        if z_score is None:
            return {
                "signal": "NONE",
                "reason": "z_not_ready",
                "z_score": None,
                "roc_30s": None,
            }
        
        # Get countertrend signal from ROC calculator
        return self._roc_calculator.get_countertrend_signal(
            z_score=z_score,
            threshold_bps=2.0,
        )
    
    def _compute_futures_features(self) -> dict:
        """Compute futures-specific features."""
        return {
            "cvd": round(self._futures_cvd, 4),
            "cvd_slope_30s": self._compute_cvd_slope(self._futures_cvd_history, CVD_SLOPE_30S_BARS),
            "cvd_slope_60s": self._compute_cvd_slope(self._futures_cvd_history, CVD_SLOPE_60S_BARS),
            "rvol_5s": self._compute_rvol(self._futures_bars_5s, RVOL_5S_LOOKBACK),
            "rvol_15s": self._compute_rvol(self._futures_bars_15s, RVOL_15S_LOOKBACK),
            "impulse_5s": self._compute_impulse(self._futures_ret_5s_history),
            "impulse_15s": self._compute_impulse(self._futures_ret_15s_history),
            "microprice": round(self._futures_microprice, 2) if self._futures_microprice else None,
            "mid": round(self._futures_mid, 2) if self._futures_mid else None,
            "mark_px": round(self._futures_mark_px, 2) if self._futures_mark_px else None,
        }
    
    def _compute_spot_features(self) -> dict:
        """Compute spot-specific features."""
        return {
            "cvd": round(self._spot_cvd, 4),
            "cvd_slope_30s": self._compute_cvd_slope(self._spot_cvd_history, CVD_SLOPE_30S_BARS),
            "cvd_slope_60s": self._compute_cvd_slope(self._spot_cvd_history, CVD_SLOPE_60S_BARS),
            "microprice": round(self._spot_microprice, 2) if self._spot_microprice else None,
            "mid": round(self._spot_mid, 2) if self._spot_mid else None,
        }
    
    def _compute_basis_features(self) -> dict:
        """Compute basis/premium features."""
        basis_mid = None
        basis_bps = None
        premium_to_mark = None
        
        if self._futures_mid and self._spot_mid:
            basis_mid = round(self._futures_mid - self._spot_mid, 2)
            if self._spot_mid > 0:
                basis_bps = round((self._futures_mid - self._spot_mid) / self._spot_mid * 10000, 2)
        
        if self._futures_mid and self._futures_mark_px:
            premium_to_mark = round(self._futures_mid - self._futures_mark_px, 2)
        
        return {
            "basis_mid": basis_mid,
            "basis_bps": basis_bps,
            "premium_to_mark": premium_to_mark,
        }
    
    def _compute_cvd_slope(self, cvd_history: deque, bars_back: int) -> Optional[float]:
        """Compute CVD slope over N bars."""
        if len(cvd_history) < bars_back + 1:
            return None
        
        cvd_now = cvd_history[-1]
        cvd_past = cvd_history[-(bars_back + 1)]
        
        return round(cvd_now - cvd_past, 4)
    
    def _compute_rvol(self, bars: deque, lookback: int) -> Optional[float]:
        """Compute relative volume."""
        if len(bars) < 2:
            return None
        
        current_vol = bars[-1].get("volume", 0)
        
        hist_bars = list(bars)[:-1]
        if len(hist_bars) < min(lookback, 10):
            return None
        
        hist_bars = hist_bars[-lookback:]
        hist_vols = [b.get("volume", 0) for b in hist_bars if b.get("volume", 0) > 0]
        
        if not hist_vols:
            return None
        
        mean_vol = mean(hist_vols)
        
        if mean_vol <= 0:
            return None
        
        return round(current_vol / mean_vol, 2)
    
    def _compute_impulse(self, ret_history: deque) -> Optional[float]:
        """Compute impulse."""
        if len(ret_history) < 10:
            return None
        
        current_ret = ret_history[-1]
        
        hist_rets = list(ret_history)[:-1]
        abs_hist = [abs(r) for r in hist_rets]
        
        if not abs_hist:
            return None
        
        median_abs = median(abs_hist)
        
        if median_abs <= 0:
            return round(abs(current_ret) * 10000, 2)
        
        return round(abs(current_ret) / median_abs, 2)
    
    # ========== Micro 1m Features ==========
    
    def _calc_ema_simple(self, values: list[float], period: int) -> Optional[float]:
        """
        Calculate EMA on a list of values.
        
        Returns the last EMA value or None if insufficient data.
        """
        if len(values) < period:
            return None
        
        k = 2 / (period + 1)
        
        # First EMA is SMA
        ema = sum(values[:period]) / period
        
        # Continue from period onwards
        for i in range(period, len(values)):
            ema = values[i] * k + ema * (1 - k)
        
        return ema
    
    def _compute_micro_1m_features(self, bars_1m: deque, ret_1m_hist: deque) -> dict:
        """
        Compute micro 1m features for a market.
        
        Returns dict with:
            ema9, ema21, trend, slope_5m_bps, ret_1m, impulse_1m, rvol_1m, stage
        """
        result = {
            "ema9": None,
            "ema21": None,
            "trend": "FLAT",
            "slope_5m_bps": None,
            "ret_1m": None,
            "impulse_1m": None,
            "rvol_1m": None,
            "stage": "NEUTRAL",
        }
        
        if len(bars_1m) < 2:
            return result
        
        # Get close prices
        closes = [b.get("close") for b in bars_1m if b.get("close")]
        volumes = [b.get("volume", 0) for b in bars_1m]
        
        if not closes:
            return result
        
        current_close = closes[-1]
        
        # C) Current return (available early)
        if len(ret_1m_hist) >= 1:
            result["ret_1m"] = round(ret_1m_hist[-1] * 10000, 2)  # in bps
        
        # B) Slope 5m (5 bars) - available after 6 bars
        if len(closes) >= MICRO_1M_SLOPE_BARS + 1:
            close_now = closes[-1]
            close_5_ago = closes[-(MICRO_1M_SLOPE_BARS + 1)]
            
            if close_now > 0:
                slope_5m = close_now - close_5_ago
                result["slope_5m_bps"] = round(slope_5m / close_now * 10000, 2)
        
        # A) EMA crossover trend (requires 21 bars for EMA21)
        if len(closes) >= MICRO_1M_EMA_SLOW:
            ema9 = self._calc_ema_simple(closes, MICRO_1M_EMA_FAST)
            ema21 = self._calc_ema_simple(closes, MICRO_1M_EMA_SLOW)
            
            result["ema9"] = round(ema9, 2) if ema9 else None
            result["ema21"] = round(ema21, 2) if ema21 else None
            
            if ema9 and ema21 and current_close > 0:
                spread_bps = (ema9 - ema21) / current_close * 10000
                
                if spread_bps > MICRO_1M_TREND_THRESHOLD_BPS:
                    result["trend"] = "UP"
                elif spread_bps < -MICRO_1M_TREND_THRESHOLD_BPS:
                    result["trend"] = "DOWN"
                else:
                    result["trend"] = "FLAT"
        
        # D) Impulse 1m (anomaly detection) - requires 10+ bars
        if len(ret_1m_hist) >= 10:
            current_ret = abs(ret_1m_hist[-1])
            
            hist_rets = list(ret_1m_hist)[:-1]
            lookback = min(len(hist_rets), MICRO_1M_IMPULSE_LOOKBACK)
            hist_rets = hist_rets[-lookback:]
            abs_hist = [abs(r) for r in hist_rets]
            
            if abs_hist:
                median_abs = median(abs_hist)
                if median_abs > 0:
                    result["impulse_1m"] = round(current_ret / median_abs, 2)
                else:
                    result["impulse_1m"] = round(current_ret * 10000, 2)
        
        # E) RVOL 1m - requires 10+ bars
        if len(volumes) >= 10:
            current_vol = volumes[-1]
            
            lookback = min(len(volumes) - 1, MICRO_1M_RVOL_LOOKBACK)
            hist_vols = volumes[-(lookback + 1):-1]
            hist_vols = [v for v in hist_vols if v > 0]
            
            if hist_vols:
                mean_vol = mean(hist_vols)
                if mean_vol > 0:
                    result["rvol_1m"] = round(current_vol / mean_vol, 2)
        
        # F) Stage detection
        result["stage"] = self._determine_micro_stage(result)
        
        return result
    
    def _determine_micro_stage(self, micro: dict) -> str:
        """
        Determine market stage based on micro indicators.
        
        Returns: ACCEL_UP, ACCEL_DOWN, MEAN_REVERT, or NEUTRAL
        """
        trend = micro.get("trend", "FLAT")
        impulse = micro.get("impulse_1m")
        slope_bps = micro.get("slope_5m_bps")
        
        # ACCEL_UP: trend UP, strong impulse, positive slope
        if (
            trend == "UP"
            and impulse is not None
            and impulse > 1.5
            and slope_bps is not None
            and slope_bps > 5
        ):
            return "ACCEL_UP"
        
        # ACCEL_DOWN: trend DOWN, strong impulse, negative slope
        if (
            trend == "DOWN"
            and impulse is not None
            and impulse > 1.5
            and slope_bps is not None
            and slope_bps < -5
        ):
            return "ACCEL_DOWN"
        
        # MEAN_REVERT: far from AVWAP and impulse moderate
        dev_from_avwap = self._compute_dev_from_avwap_bps()
        if dev_from_avwap is not None and abs(dev_from_avwap) > 10:
            # Check if impulse is fading (would need history, simplified here)
            if impulse is not None and impulse < 1.0:
                return "MEAN_REVERT"
        
        return "NEUTRAL"
    
    def _compute_micro_1m_section(self) -> dict:
        """
        Compute full micro_1m section for snapshot.
        
        Returns dict with futures and spot micro features.
        """
        return {
            "futures": self._compute_micro_1m_features(
                self._futures_bars_1m,
                self._futures_ret_1m_history,
            ),
            "spot": self._compute_micro_1m_features(
                self._spot_bars_1m,
                self._spot_ret_1m_history,
            ),
        }
    
    def _compute_polymarket_meta(self) -> dict:
        """
        Compute Polymarket metadata section.
        
        Returns:
            Dictionary with market question, UP/YES mapping, and config.
        """
        from collector.config import settings
        
        market_id = None
        market_question = None
        
        if self._polymarket_book_store:
            snap = self._polymarket_book_store.snapshot()
            market_id = snap.get("market_id")
            # Question comes from resolver - currently not stored, placeholder
            # In future: get from MarketResolver metadata
        
        return {
            "market_id": market_id,
            "market_question": market_question,  # Placeholder for now
            "up_is_yes": settings.POLY_UP_IS_YES,
            "spread_veto_bps": settings.POLYMARKET_MAX_SPREAD_BPS,
            "min_depth_top3": settings.POLYMARKET_MIN_DEPTH,
        }
    
    def _compute_polymarket_up_down(self, pm_features: Optional[dict]) -> Optional[dict]:
        """
        Compute normalized UP/DOWN from YES/NO with slice_ok and required_edge.
        
        BLOCK 1: This is the key normalization step that:
        1. Maps YES/NO tokens to UP/DOWN based on POLY_UP_IS_YES config
        2. Computes slice_ok: whether there's enough liquidity for a $20 slice
        3. Computes required_edge_bps: minimum edge needed to overcome spread
        
        Args:
            pm_features: Polymarket features snapshot from _compute_polymarket_features()
            
        Returns:
            Dictionary with structure:
            {
                "up": {bid, ask, mid, spread_bps, depth_top1_usd, depth_top3_usd, 
                       slice_ok, required_edge_bps},
                "down": {...same...},
                "up_is_yes": bool,
                "ts_ms": int,
                "connected": bool,
                "age_sec": float
            }
        
        Notes:
            - slice_ok=True means we can execute at least one $20 slice
            - required_edge_bps tells us minimum edge to be profitable after spread
            - If pm_features is None, returns None (Polymarket not available)
        """
        from collector.config import settings
        from collector.polymarket.normalize_updown import normalize_polymarket_updown
        
        if not pm_features:
            return None
        
        # Use the new normalization module from BLOCK 1
        return normalize_polymarket_updown(
            pm_snapshot=pm_features,
            up_is_yes=settings.POLY_UP_IS_YES,
            min_top1=settings.MIN_TOP1_USD,
            min_top3=settings.MIN_TOP3_USD,
            edge_buffer_bps=settings.EDGE_BUFFER_BPS,
        )
    
    def _compute_polymarket_features(self) -> Optional[dict]:
        """
        Compute Polymarket features section for snapshot.
        
        Returns:
            Dictionary with YES/NO orderbook metrics, or None if not available.
            
        Structure:
            {
                "market_id": "btc-updown-15m-...",
                "yes": {
                    "mid": 0.52,
                    "spread_bps": 200,
                    "depth_top3": 5000,
                    "imbalance": 0.15,
                    ...
                },
                "no": {...},
                "connected": true,
                "ts_ms": 1234567890,
                "age_sec": 0.5
            }
        """
        if not self._polymarket_book_store:
            return None
        
        return self._polymarket_book_store.snapshot()
    
    def get_micro_1m_summary(self) -> dict:
        """Get micro 1m summary for periodic logging."""
        micro = self._compute_micro_1m_section()
        futures = micro.get("futures", {})
        
        return {
            "futures_trend": futures.get("trend"),
            "futures_slope_5m_bps": futures.get("slope_5m_bps"),
            "futures_impulse_1m": futures.get("impulse_1m"),
            "futures_rvol_1m": futures.get("rvol_1m"),
            "futures_stage": futures.get("stage"),
            "futures_bars_count": len(self._futures_bars_1m),
        }
    
    def get_short_summary(self) -> dict:
        """Get short summary for periodic logging."""
        snap = self.snapshot()
        futures = snap.get("futures", {})
        basis = snap.get("basis", {})
        anchor = snap.get("anchor", {})
        
        return {
            "cvd": futures.get("cvd"),
            "cvd_slope_30s": futures.get("cvd_slope_30s"),
            "rvol_5s": futures.get("rvol_5s"),
            "impulse_5s": futures.get("impulse_5s"),
            "basis_bps": basis.get("basis_bps"),
            "avwap_15m": anchor.get("avwap_15m"),
            "dev_from_avwap_bps": anchor.get("dev_from_avwap_bps"),
        }
    
    def get_anchor_summary(self) -> dict:
        """Get anchor state summary for periodic logging."""
        return {
            "mode": self._anchor_mode,
            "anchor_time_ms": self._anchor_time_ms,
            "avwap_15m": round(self._avwap_15m, 2) if self._avwap_15m else None,
            "dev_from_avwap_bps": self._compute_dev_from_avwap_bps(),
        }
    
    def get_final_summary(self, metrics_snapshot: Optional[dict] = None) -> dict:
        """
        Get final features summary for periodic logging.
        
        Includes liquidations, absorption, quality mode, gaps.
        """
        ts_now = now_ms()
        self._check_gaps(ts_now)
        self._recompute_liquidations(ts_now)
        quality = self.compute_quality(metrics_snapshot)
        
        return {
            "liq_qty_30s": round(self._liq_qty_30s, 4),
            "liq_count_30s": self._liq_count_30s,
            "absorption_score_30s": self._compute_absorption_score(),
            "quality_mode": quality.get("mode"),
            "gap_detected_futures": self._gap_detected_futures,
            "gap_detected_spot": self._gap_detected_spot,
        }
