#!/usr/bin/env python3
"""
CAL1 B2: Threshold Calibration Script
======================================

Grid search for optimal signal thresholds WITHOUT actual trading.

Evaluates signal quality using proxy metric:
    future_move_bps = (mid_{t+H} - ask_t) / ask_t * 10000

If signals consistently predict positive future moves, they are useful.

Parameters searched:
    - SPIKE_Z_TH: Z-score threshold for spike detection
    - DIP_TH_BPS: Basis points threshold for dip detection
    - EDGE_BUFFER_BPS: Buffer added to spread for required edge
    - NET_EDGE_MIN_BPS: Minimum net edge to trigger signal
    - H_SEC: Horizon in seconds for future move evaluation

Usage:
    python scripts/calibrate_thresholds.py --data-dir data/decisions --last-n-windows 200 --out results/calib_top.csv

Output:
    CSV with top parameter combinations and their metrics.
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Optional, Iterator

# Try to import pandas for parquet support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================
# Configuration
# ============================================================

# Parameter grid
SPIKE_Z_TH_GRID = [2.0, 2.5, 3.0, 3.5]
DIP_TH_BPS_GRID = [50, 80, 120, 160]
EDGE_BUFFER_BPS_GRID = [10, 25, 50]
NET_EDGE_MIN_BPS_GRID = [0, 25, 50, 100]
H_SEC_GRID = [30, 60, 120]

# Constraints
MIN_SIGNALS_PER_WINDOW = 5
MAX_SIGNALS_PER_WINDOW = 200


# ============================================================
# Data Loading (same as validate_dataset.py)
# ============================================================

def safe_get(dct: Optional[dict], path: str, default: Any = None) -> Any:
    """Safely get nested dict value by dot path."""
    if dct is None:
        return default
    keys = path.split(".")
    for key in keys:
        if isinstance(dct, dict):
            dct = dct.get(key)
        else:
            return default
        if dct is None:
            return default
    return dct


def load_jsonl_file(filepath: Path) -> Iterator[dict]:
    """Load records from JSONL file."""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def load_parquet_file(filepath: Path) -> Iterator[dict]:
    """Load records from Parquet file."""
    if not HAS_PANDAS:
        return iter([])
    df = pd.read_parquet(filepath)
    for _, row in df.iterrows():
        yield row.to_dict()


def load_data_files(data_dir: str, last_n_windows: int = 200) -> list[dict]:
    """Load tick data from directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    records = []
    jsonl_files = sorted(data_path.glob("**/*.jsonl"))
    parquet_files = sorted(data_path.glob("**/*.parquet"))
    
    for filepath in jsonl_files[-last_n_windows:]:
        for record in load_jsonl_file(filepath):
            records.append(record)
    
    for filepath in parquet_files[-last_n_windows:]:
        for record in load_parquet_file(filepath):
            records.append(record)
    
    print(f"Loaded {len(records)} records")
    return records


# ============================================================
# Signal Generation
# ============================================================

@dataclass
class TickData:
    """Extracted tick data for signal evaluation."""
    ts_ms: int
    
    # Spikes
    z_ret_5s: Optional[float]
    ret_5s_bps: Optional[float]
    
    # Dips
    up_dip_bps: Optional[float]
    down_dip_bps: Optional[float]
    
    # Edge
    fair_up: Optional[float]
    fair_down: Optional[float]
    market_up: Optional[float]
    market_down: Optional[float]
    ask_up: Optional[float]
    ask_down: Optional[float]
    spread_up_bps: Optional[float]
    spread_down_bps: Optional[float]
    
    # Window
    window_id: int


def extract_tick_data(record: dict) -> Optional[TickData]:
    """Extract relevant fields from a record."""
    ts = record.get("ts", 0)
    if ts == 0:
        return None
    
    signals = record.get("signals") or {}
    accumulate = record.get("accumulate") or {}
    pm_updown = record.get("polymarket_up_down") or {}
    fair = record.get("fair") or {}
    
    up = pm_updown.get("up") or {}
    down = pm_updown.get("down") or {}
    
    return TickData(
        ts_ms=ts,
        z_ret_5s=safe_get(signals, "z_ret_5s") or safe_get(accumulate, "raw_z_ret_5s"),
        ret_5s_bps=safe_get(signals, "ret_5s_bps") or safe_get(accumulate, "raw_ret_5s_bps"),
        up_dip_bps=safe_get(signals, "up_dip_bps") or safe_get(accumulate, "raw_pm_up_dip_bps"),
        down_dip_bps=safe_get(signals, "down_dip_bps") or safe_get(accumulate, "raw_pm_down_dip_bps"),
        fair_up=safe_get(fair, "fair_up") or safe_get(accumulate, "fair_up"),
        fair_down=safe_get(fair, "fair_down") or safe_get(accumulate, "fair_down"),
        market_up=safe_get(up, "mid") or safe_get(accumulate, "market_up"),
        market_down=safe_get(down, "mid") or safe_get(accumulate, "market_down"),
        ask_up=safe_get(up, "ask") or safe_get(accumulate, "ask_up"),
        ask_down=safe_get(down, "ask") or safe_get(accumulate, "ask_down"),
        spread_up_bps=safe_get(up, "spread_bps"),
        spread_down_bps=safe_get(down, "spread_bps"),
        window_id=ts // (15 * 60 * 1000),
    )


def compute_edge_bps(fair: Optional[float], market: Optional[float]) -> Optional[float]:
    """Compute edge in basis points."""
    if fair is None or market is None:
        return None
    return (fair - market) * 10000


def compute_net_edge(
    fair: Optional[float],
    market: Optional[float],
    spread_bps: Optional[float],
    edge_buffer_bps: float,
) -> Optional[float]:
    """Compute net edge after spread cost."""
    edge = compute_edge_bps(fair, market)
    if edge is None:
        return None
    
    required = (spread_bps / 2 if spread_bps else 0) + edge_buffer_bps
    return edge - required


def generate_signal(
    tick: TickData,
    spike_z_th: float,
    dip_th_bps: float,
    edge_buffer_bps: float,
    net_edge_min_bps: float,
) -> Optional[str]:
    """
    Generate signal based on parameters.
    
    Returns: "BUY_UP", "BUY_DOWN", or None
    """
    # Compute net edges
    net_edge_up = compute_net_edge(
        tick.fair_up, tick.market_up, tick.spread_up_bps, edge_buffer_bps
    )
    net_edge_down = compute_net_edge(
        tick.fair_down, tick.market_down, tick.spread_down_bps, edge_buffer_bps
    )
    
    # Check triggers
    down_spike = (
        tick.z_ret_5s is not None and
        tick.z_ret_5s <= -spike_z_th
    )
    up_spike = (
        tick.z_ret_5s is not None and
        tick.z_ret_5s >= spike_z_th
    )
    up_dip = (
        tick.up_dip_bps is not None and
        tick.up_dip_bps <= -dip_th_bps
    )
    down_dip = (
        tick.down_dip_bps is not None and
        tick.down_dip_bps <= -dip_th_bps
    )
    
    # BUY_UP: positive net edge + (down_spike OR up_dip)
    if (net_edge_up is not None and
        net_edge_up >= net_edge_min_bps and
        (down_spike or up_dip)):
        return "BUY_UP"
    
    # BUY_DOWN: positive net edge + (up_spike OR down_dip)
    if (net_edge_down is not None and
        net_edge_down >= net_edge_min_bps and
        (up_spike or down_dip)):
        return "BUY_DOWN"
    
    return None


# ============================================================
# Future Move Computation
# ============================================================

def build_price_index(ticks: list[TickData]) -> dict[int, TickData]:
    """Build index of ticks by timestamp for fast lookup."""
    return {t.ts_ms: t for t in ticks}


def find_future_tick(
    ticks: list[TickData],
    start_idx: int,
    h_sec: int,
) -> Optional[TickData]:
    """Find tick approximately h_sec in the future."""
    target_ts = ticks[start_idx].ts_ms + h_sec * 1000
    
    # Linear search forward (ticks should be sorted by time)
    for i in range(start_idx + 1, len(ticks)):
        if ticks[i].ts_ms >= target_ts:
            return ticks[i]
    
    return None


def compute_future_move_bps(
    signal_tick: TickData,
    future_tick: TickData,
    side: str,
) -> Optional[float]:
    """
    Compute future move in basis points.
    
    For BUY_UP: (future_mid_up - ask_up) / ask_up * 10000
    For BUY_DOWN: (future_mid_down - ask_down) / ask_down * 10000
    """
    if side == "BUY_UP":
        ask_now = signal_tick.ask_up
        mid_future = future_tick.market_up
    else:
        ask_now = signal_tick.ask_down
        mid_future = future_tick.market_down
    
    if ask_now is None or mid_future is None or ask_now == 0:
        return None
    
    return (mid_future - ask_now) / ask_now * 10000


# ============================================================
# Grid Search
# ============================================================

@dataclass
class ParamResult:
    """Result for a parameter combination."""
    spike_z_th: float
    dip_th_bps: float
    edge_buffer_bps: float
    net_edge_min_bps: float
    h_sec: int
    
    n_signals_up: int = 0
    n_signals_down: int = 0
    avg_future_move_up: float = 0.0
    avg_future_move_down: float = 0.0
    winrate_up: float = 0.0
    winrate_down: float = 0.0
    
    # Combined metrics
    avg_move_all: float = 0.0
    total_signals: int = 0
    signals_per_window: float = 0.0
    
    # Penalty flag
    penalty: bool = False
    penalty_reason: str = ""


def evaluate_params(
    ticks: list[TickData],
    spike_z_th: float,
    dip_th_bps: float,
    edge_buffer_bps: float,
    net_edge_min_bps: float,
    h_sec: int,
    n_windows: int,
) -> ParamResult:
    """Evaluate a parameter combination."""
    result = ParamResult(
        spike_z_th=spike_z_th,
        dip_th_bps=dip_th_bps,
        edge_buffer_bps=edge_buffer_bps,
        net_edge_min_bps=net_edge_min_bps,
        h_sec=h_sec,
    )
    
    # Generate signals and compute future moves
    up_moves = []
    down_moves = []
    signals_per_window = defaultdict(int)
    
    for i, tick in enumerate(ticks):
        signal = generate_signal(
            tick, spike_z_th, dip_th_bps, edge_buffer_bps, net_edge_min_bps
        )
        
        if signal is None:
            continue
        
        # Find future tick
        future_tick = find_future_tick(ticks, i, h_sec)
        if future_tick is None:
            continue
        
        # Compute future move
        move = compute_future_move_bps(tick, future_tick, signal)
        if move is None:
            continue
        
        # Record
        signals_per_window[tick.window_id] += 1
        
        if signal == "BUY_UP":
            up_moves.append(move)
        else:
            down_moves.append(move)
    
    # Compute metrics
    result.n_signals_up = len(up_moves)
    result.n_signals_down = len(down_moves)
    result.total_signals = len(up_moves) + len(down_moves)
    
    if up_moves:
        result.avg_future_move_up = sum(up_moves) / len(up_moves)
        result.winrate_up = sum(1 for m in up_moves if m > 0) / len(up_moves) * 100
    
    if down_moves:
        result.avg_future_move_down = sum(down_moves) / len(down_moves)
        result.winrate_down = sum(1 for m in down_moves if m > 0) / len(down_moves) * 100
    
    all_moves = up_moves + down_moves
    if all_moves:
        result.avg_move_all = sum(all_moves) / len(all_moves)
    
    if n_windows > 0 and signals_per_window:
        result.signals_per_window = result.total_signals / n_windows
    
    # Check constraints
    avg_signals_per_window = result.signals_per_window
    if avg_signals_per_window < MIN_SIGNALS_PER_WINDOW:
        result.penalty = True
        result.penalty_reason = f"too_few_signals({avg_signals_per_window:.1f}/window)"
    elif avg_signals_per_window > MAX_SIGNALS_PER_WINDOW:
        result.penalty = True
        result.penalty_reason = f"too_many_signals({avg_signals_per_window:.1f}/window)"
    
    return result


def run_grid_search(
    ticks: list[TickData],
    n_windows: int,
) -> list[ParamResult]:
    """Run grid search over all parameter combinations."""
    results = []
    
    total_combos = (
        len(SPIKE_Z_TH_GRID) *
        len(DIP_TH_BPS_GRID) *
        len(EDGE_BUFFER_BPS_GRID) *
        len(NET_EDGE_MIN_BPS_GRID) *
        len(H_SEC_GRID)
    )
    
    print(f"Running grid search over {total_combos} parameter combinations...")
    
    for i, (spike_z, dip_th, edge_buf, net_min, h_sec) in enumerate(product(
        SPIKE_Z_TH_GRID,
        DIP_TH_BPS_GRID,
        EDGE_BUFFER_BPS_GRID,
        NET_EDGE_MIN_BPS_GRID,
        H_SEC_GRID,
    )):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i + 1}/{total_combos}")
        
        result = evaluate_params(
            ticks, spike_z, dip_th, edge_buf, net_min, h_sec, n_windows
        )
        results.append(result)
    
    return results


# ============================================================
# Output
# ============================================================

def write_results_csv(results: list[ParamResult], output_path: Path):
    """Write results to CSV."""
    # Sort by avg_move_all (descending), filtering out penalized
    valid_results = [r for r in results if not r.penalty]
    valid_results.sort(key=lambda r: -r.avg_move_all)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow([
            "rank",
            "spike_z_th",
            "dip_th_bps",
            "edge_buffer_bps",
            "net_edge_min_bps",
            "h_sec",
            "n_signals_up",
            "n_signals_down",
            "total_signals",
            "signals_per_window",
            "avg_move_up_bps",
            "avg_move_down_bps",
            "avg_move_all_bps",
            "winrate_up_pct",
            "winrate_down_pct",
        ])
        
        # Top results
        for i, r in enumerate(valid_results[:50]):
            writer.writerow([
                i + 1,
                r.spike_z_th,
                r.dip_th_bps,
                r.edge_buffer_bps,
                r.net_edge_min_bps,
                r.h_sec,
                r.n_signals_up,
                r.n_signals_down,
                r.total_signals,
                f"{r.signals_per_window:.1f}",
                f"{r.avg_future_move_up:.2f}",
                f"{r.avg_future_move_down:.2f}",
                f"{r.avg_move_all:.2f}",
                f"{r.winrate_up:.1f}",
                f"{r.winrate_down:.1f}",
            ])
    
    print(f"Results written to: {output_path}")


def print_top_results(results: list[ParamResult], n: int = 10):
    """Print top N results."""
    valid_results = [r for r in results if not r.penalty]
    valid_results.sort(key=lambda r: -r.avg_move_all)
    
    print("\n" + "=" * 80)
    print("TOP PARAMETER COMBINATIONS (by avg_move_all)")
    print("=" * 80)
    
    print(f"\n{'#':>3} {'Z_TH':>5} {'DIP':>5} {'BUF':>5} {'MIN':>5} {'H':>4} "
          f"{'N_UP':>6} {'N_DN':>6} {'AVG_UP':>8} {'AVG_DN':>8} {'AVG_ALL':>8} "
          f"{'WR_UP':>6} {'WR_DN':>6}")
    print("-" * 80)
    
    for i, r in enumerate(valid_results[:n]):
        print(f"{i+1:>3} {r.spike_z_th:>5.1f} {r.dip_th_bps:>5.0f} "
              f"{r.edge_buffer_bps:>5.0f} {r.net_edge_min_bps:>5.0f} {r.h_sec:>4} "
              f"{r.n_signals_up:>6} {r.n_signals_down:>6} "
              f"{r.avg_future_move_up:>8.2f} {r.avg_future_move_down:>8.2f} "
              f"{r.avg_move_all:>8.2f} "
              f"{r.winrate_up:>5.1f}% {r.winrate_down:>5.1f}%")
    
    # Best by winrate
    valid_results.sort(key=lambda r: -(r.winrate_up + r.winrate_down) / 2)
    
    print("\n" + "-" * 80)
    print("TOP BY WINRATE")
    print("-" * 80)
    
    for i, r in enumerate(valid_results[:5]):
        avg_wr = (r.winrate_up + r.winrate_down) / 2
        print(f"{i+1:>3} Z={r.spike_z_th:.1f} DIP={r.dip_th_bps:.0f} "
              f"BUF={r.edge_buffer_bps:.0f} MIN={r.net_edge_min_bps:.0f} H={r.h_sec} "
              f"→ AvgWR={avg_wr:.1f}% (UP:{r.winrate_up:.1f}% DN:{r.winrate_down:.1f}%)")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate signal thresholds using grid search (CAL1)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/decisions",
        help="Directory containing tick data",
    )
    parser.add_argument(
        "--last-n-windows",
        type=int,
        default=200,
        help="Number of recent windows/files to analyze",
    )
    parser.add_argument(
        "--out",
        default="results/calib_top.csv",
        help="Output CSV file for top results",
    )
    
    args = parser.parse_args()
    
    # Load data
    records = load_data_files(args.data_dir, args.last_n_windows)
    
    if not records:
        print("No records found.")
        sys.exit(1)
    
    # Extract tick data
    ticks = []
    for record in records:
        tick = extract_tick_data(record)
        if tick:
            ticks.append(tick)
    
    print(f"Extracted {len(ticks)} valid ticks")
    
    # Sort by timestamp
    ticks.sort(key=lambda t: t.ts_ms)
    
    # Count windows
    windows = set(t.window_id for t in ticks)
    n_windows = len(windows)
    print(f"Data spans {n_windows} windows")
    
    # Run grid search
    results = run_grid_search(ticks, n_windows)
    
    # Output
    print_top_results(results)
    write_results_csv(results, Path(args.out))
    
    # Save best params as JSON
    valid_results = [r for r in results if not r.penalty]
    if valid_results:
        valid_results.sort(key=lambda r: -r.avg_move_all)
        best = valid_results[0]
        
        best_params = {
            "SPIKE_Z_TH": best.spike_z_th,
            "DIP_TH_BPS": best.dip_th_bps,
            "EDGE_BUFFER_BPS": best.edge_buffer_bps,
            "NET_EDGE_MIN_BPS": best.net_edge_min_bps,
            "H_SEC": best.h_sec,
            "metrics": {
                "avg_move_all_bps": best.avg_move_all,
                "winrate_up": best.winrate_up,
                "winrate_down": best.winrate_down,
                "total_signals": best.total_signals,
            }
        }
        
        best_path = Path(args.out).parent / "best_params.json"
        with open(best_path, "w") as f:
            json.dump(best_params, f, indent=2)
        print(f"Best params saved to: {best_path}")


if __name__ == "__main__":
    main()
