#!/usr/bin/env python3
"""
CAL1 C1: Window Visualization Script
=====================================

Plot a single 15-minute window to visually verify that:
- BUY_UP signals occur on price dips (not peaks)
- BUY_DOWN signals occur on price rises (not dips)

Usage:
    python scripts/plot_window.py --window-id 123456 --params results/best_params.json

Output:
    PNG image saved to results/window_{window_id}.png
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Any

# Check for matplotlib
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# Check for pandas
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================
# Data Loading (reuse from other scripts)
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


def load_jsonl_file(filepath: Path) -> list[dict]:
    """Load records from JSONL file."""
    records = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_parquet_file(filepath: Path) -> list[dict]:
    """Load records from Parquet file."""
    if not HAS_PANDAS:
        return []
    df = pd.read_parquet(filepath)
    return df.to_dict('records')


def load_window_data(data_dir: str, window_id: int) -> list[dict]:
    """Load all records for a specific window."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    records = []
    window_start_ms = window_id * 15 * 60 * 1000
    window_end_ms = window_start_ms + 15 * 60 * 1000
    
    # Load all files and filter by window
    jsonl_files = sorted(data_path.glob("**/*.jsonl"))
    parquet_files = sorted(data_path.glob("**/*.parquet"))
    
    for filepath in jsonl_files:
        for record in load_jsonl_file(filepath):
            ts = record.get("ts", 0)
            if window_start_ms <= ts < window_end_ms:
                records.append(record)
    
    for filepath in parquet_files:
        for record in load_parquet_file(filepath):
            ts = record.get("ts", 0)
            if window_start_ms <= ts < window_end_ms:
                records.append(record)
    
    records.sort(key=lambda r: r.get("ts", 0))
    return records


def extract_plot_data(records: list[dict]) -> dict:
    """Extract data for plotting."""
    data = {
        "ts": [],
        "datetime": [],
        
        # UP prices
        "up_mid": [],
        "up_ask": [],
        "fair_up": [],
        "net_edge_up": [],
        
        # DOWN prices
        "down_mid": [],
        "down_ask": [],
        "fair_down": [],
        "net_edge_down": [],
        
        # Signals
        "ret_5s_bps": [],
        "z_ret_5s": [],
        "up_dip_bps": [],
        "down_dip_bps": [],
        
        # Decisions
        "action": [],
    }
    
    for record in records:
        ts = record.get("ts", 0)
        data["ts"].append(ts)
        data["datetime"].append(datetime.fromtimestamp(ts / 1000))
        
        # Accumulate section
        accum = record.get("accumulate") or {}
        signals = record.get("signals") or {}
        fair = record.get("fair") or {}
        
        # Prices
        data["up_mid"].append(safe_get(accum, "market_up"))
        data["up_ask"].append(safe_get(accum, "ask_up"))
        data["fair_up"].append(safe_get(fair, "fair_up") or safe_get(accum, "fair_up"))
        data["net_edge_up"].append(safe_get(accum, "net_edge_up"))
        
        data["down_mid"].append(safe_get(accum, "market_down"))
        data["down_ask"].append(safe_get(accum, "ask_down"))
        data["fair_down"].append(safe_get(fair, "fair_down") or safe_get(accum, "fair_down"))
        data["net_edge_down"].append(safe_get(accum, "net_edge_down"))
        
        # Signals
        data["ret_5s_bps"].append(
            safe_get(signals, "ret_5s_bps") or safe_get(accum, "raw_ret_5s_bps")
        )
        data["z_ret_5s"].append(
            safe_get(signals, "z_ret_5s") or safe_get(accum, "raw_z_ret_5s")
        )
        data["up_dip_bps"].append(
            safe_get(signals, "up_dip_bps") or safe_get(accum, "raw_pm_up_dip_bps")
        )
        data["down_dip_bps"].append(
            safe_get(signals, "down_dip_bps") or safe_get(accum, "raw_pm_down_dip_bps")
        )
        
        # Action
        data["action"].append(safe_get(accum, "action", "WAIT"))
    
    return data


# ============================================================
# Plotting
# ============================================================

def plot_window(data: dict, window_id: int, params: dict, output_path: Path):
    """Create window visualization."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib not installed. Install with: pip install matplotlib")
        sys.exit(1)
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    
    dt = data["datetime"]
    
    # ==================== Panel 1: UP prices ====================
    ax1 = axes[0]
    
    up_mid = [v for v in data["up_mid"] if v is not None]
    up_ask = [v for v in data["up_ask"] if v is not None]
    fair_up = [v for v in data["fair_up"] if v is not None]
    
    if up_mid:
        ax1.plot(dt[:len(up_mid)], up_mid, 'b-', label='UP mid', linewidth=1)
    if up_ask:
        ax1.plot(dt[:len(up_ask)], up_ask, 'b--', label='UP ask', linewidth=0.5, alpha=0.5)
    if fair_up:
        ax1.plot(dt[:len(fair_up)], fair_up, 'g-', label='Fair UP', linewidth=1.5)
    
    # Mark BUY_UP signals
    for i, action in enumerate(data["action"]):
        if action == "ACCUMULATE_UP" and i < len(dt):
            mid = data["up_mid"][i]
            if mid is not None:
                ax1.scatter([dt[i]], [mid], c='green', s=100, marker='^', zorder=5)
    
    ax1.set_ylabel('UP Price')
    ax1.legend(loc='upper left')
    ax1.set_title(f'Window {window_id} - UP Prices & Fair Value')
    ax1.grid(True, alpha=0.3)
    
    # ==================== Panel 2: DOWN prices ====================
    ax2 = axes[1]
    
    down_mid = [v for v in data["down_mid"] if v is not None]
    down_ask = [v for v in data["down_ask"] if v is not None]
    fair_down = [v for v in data["fair_down"] if v is not None]
    
    if down_mid:
        ax2.plot(dt[:len(down_mid)], down_mid, 'r-', label='DOWN mid', linewidth=1)
    if down_ask:
        ax2.plot(dt[:len(down_ask)], down_ask, 'r--', label='DOWN ask', linewidth=0.5, alpha=0.5)
    if fair_down:
        ax2.plot(dt[:len(fair_down)], fair_down, 'orange', label='Fair DOWN', linewidth=1.5)
    
    # Mark BUY_DOWN signals
    for i, action in enumerate(data["action"]):
        if action == "ACCUMULATE_DOWN" and i < len(dt):
            mid = data["down_mid"][i]
            if mid is not None:
                ax2.scatter([dt[i]], [mid], c='red', s=100, marker='v', zorder=5)
    
    ax2.set_ylabel('DOWN Price')
    ax2.legend(loc='upper left')
    ax2.set_title('DOWN Prices & Fair Value')
    ax2.grid(True, alpha=0.3)
    
    # ==================== Panel 3: Net Edge ====================
    ax3 = axes[2]
    
    net_edge_up = [v if v is not None else 0 for v in data["net_edge_up"]]
    net_edge_down = [v if v is not None else 0 for v in data["net_edge_down"]]
    
    ax3.plot(dt[:len(net_edge_up)], net_edge_up, 'g-', label='Net Edge UP', linewidth=1)
    ax3.plot(dt[:len(net_edge_down)], net_edge_down, 'r-', label='Net Edge DOWN', linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    
    # Threshold line
    min_edge = params.get("NET_EDGE_MIN_BPS", 0)
    if min_edge > 0:
        ax3.axhline(y=min_edge, color='gray', linestyle=':', label=f'Min edge ({min_edge})')
    
    ax3.set_ylabel('Net Edge (bps)')
    ax3.legend(loc='upper left')
    ax3.set_title('Net Edge')
    ax3.grid(True, alpha=0.3)
    
    # ==================== Panel 4: Signals ====================
    ax4 = axes[3]
    
    # ret_5s
    ret_5s = [v if v is not None else 0 for v in data["ret_5s_bps"]]
    ax4.plot(dt[:len(ret_5s)], ret_5s, 'b-', label='ret_5s (bps)', linewidth=1, alpha=0.7)
    
    # Dips (secondary axis)
    ax4b = ax4.twinx()
    up_dip = [v if v is not None else 0 for v in data["up_dip_bps"]]
    down_dip = [v if v is not None else 0 for v in data["down_dip_bps"]]
    
    ax4b.plot(dt[:len(up_dip)], up_dip, 'g--', label='UP dip (bps)', linewidth=1, alpha=0.7)
    ax4b.plot(dt[:len(down_dip)], down_dip, 'r--', label='DOWN dip (bps)', linewidth=1, alpha=0.7)
    
    # Thresholds
    spike_z = params.get("SPIKE_Z_TH", 2.0)
    dip_th = params.get("DIP_TH_BPS", 80)
    ax4b.axhline(y=-dip_th, color='purple', linestyle=':', label=f'Dip threshold (-{dip_th})')
    
    # Mark signals
    for i, action in enumerate(data["action"]):
        if action in ["ACCUMULATE_UP", "ACCUMULATE_DOWN"] and i < len(dt):
            ax4.axvline(x=dt[i], color='yellow', alpha=0.3, linewidth=3)
    
    ax4.set_ylabel('ret_5s (bps)', color='blue')
    ax4b.set_ylabel('Dip (bps)', color='green')
    ax4.set_xlabel('Time')
    ax4.set_title('Signals: Returns & Dips')
    ax4.grid(True, alpha=0.3)
    
    # Combined legend
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Format x-axis
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show info about signals
    buy_up_count = sum(1 for a in data["action"] if a == "ACCUMULATE_UP")
    buy_down_count = sum(1 for a in data["action"] if a == "ACCUMULATE_DOWN")
    print(f"Signals in window: BUY_UP={buy_up_count}, BUY_DOWN={buy_down_count}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Plot a single window for visual validation (CAL1)"
    )
    parser.add_argument(
        "--window-id",
        type=int,
        required=True,
        help="Window ID to plot (ts_ms // 900000)",
    )
    parser.add_argument(
        "--data-dir",
        default="data/decisions",
        help="Directory containing tick data",
    )
    parser.add_argument(
        "--params",
        default="results/best_params.json",
        help="JSON file with parameters to display thresholds",
    )
    parser.add_argument(
        "--out",
        help="Output PNG file (default: results/window_{window_id}.png)",
    )
    
    args = parser.parse_args()
    
    # Load params
    params = {}
    params_path = Path(args.params)
    if params_path.exists():
        with open(params_path) as f:
            params = json.load(f)
        print(f"Loaded params: {params}")
    else:
        print(f"Warning: Params file not found: {args.params}")
    
    # Load window data
    print(f"Loading data for window {args.window_id}...")
    records = load_window_data(args.data_dir, args.window_id)
    
    if not records:
        print(f"No records found for window {args.window_id}")
        print(f"Window time: {datetime.fromtimestamp(args.window_id * 900)}")
        sys.exit(1)
    
    print(f"Found {len(records)} records")
    
    # Extract plot data
    data = extract_plot_data(records)
    
    # Determine output path
    if args.out:
        output_path = Path(args.out)
    else:
        output_path = Path(f"results/window_{args.window_id}.png")
    
    # Plot
    plot_window(data, args.window_id, params, output_path)


if __name__ == "__main__":
    main()
