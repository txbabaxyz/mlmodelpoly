#!/usr/bin/env python3
"""
CAL1 B1: Dataset Validation Script
===================================

Validates recorded ticks for signal correctness:
- Spike sign inversions (ret vs spike flag)
- Dip sign inversions (should be <= 0)
- Decision vs trigger alignment
- Decision vs edge alignment

Usage:
    python scripts/validate_dataset.py --data-dir data/decisions --last-n-windows 50

Output:
    - % spike inversions
    - % dip inversions
    - % decision mismatches
    - Top problematic windows
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Iterator

# Try to import pandas for parquet support
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# ============================================================
# Data Loading
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
        print(f"Warning: pandas not installed, skipping {filepath}")
        return iter([])
    
    df = pd.read_parquet(filepath)
    for _, row in df.iterrows():
        yield row.to_dict()


def load_data_files(data_dir: str, last_n_windows: int = 50) -> list[dict]:
    """
    Load tick data from directory.
    
    Supports both JSONL and Parquet formats.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    records = []
    
    # Find all data files
    jsonl_files = sorted(data_path.glob("**/*.jsonl"))
    parquet_files = sorted(data_path.glob("**/*.parquet"))
    
    # Load JSONL files
    for filepath in jsonl_files[-last_n_windows:]:
        for record in load_jsonl_file(filepath):
            records.append(record)
    
    # Load Parquet files
    for filepath in parquet_files[-last_n_windows:]:
        for record in load_parquet_file(filepath):
            records.append(record)
    
    print(f"Loaded {len(records)} records from {len(jsonl_files) + len(parquet_files)} files")
    return records


# ============================================================
# Validation Checks
# ============================================================

class ValidationStats:
    """Accumulates validation statistics."""
    
    def __init__(self):
        self.total_records = 0
        self.spike_inversions = 0
        self.dip_inversions = 0
        self.decision_trigger_mismatches = 0
        self.decision_edge_mismatches = 0
        
        # Per-window stats
        self.window_issues = defaultdict(list)
        
        # Distributions
        self.ret_5s_values = []
        self.up_dip_values = []
        self.down_dip_values = []
        
        # Detailed issues
        self.issues = []
    
    def add_issue(self, ts: int, issue_type: str, details: str):
        """Record an issue."""
        self.issues.append({
            "ts": ts,
            "type": issue_type,
            "details": details,
        })
        
        # Group by window (15 min)
        window_id = ts // (15 * 60 * 1000)
        self.window_issues[window_id].append(issue_type)


def validate_record(record: dict, stats: ValidationStats) -> list[str]:
    """
    Validate a single record.
    
    Returns list of warnings.
    """
    warnings = []
    ts = record.get("ts", 0)
    
    # Extract signals section
    signals = record.get("signals") or {}
    accumulate = record.get("accumulate") or {}
    
    # Get raw values
    ret_5s_bps = safe_get(signals, "ret_5s_bps") or safe_get(accumulate, "raw_ret_5s_bps")
    z_ret_5s = safe_get(signals, "z_ret_5s") or safe_get(accumulate, "raw_z_ret_5s")
    down_spike = safe_get(signals, "down_spike_5s", False)
    up_spike = safe_get(signals, "up_spike_5s", False)
    
    up_dip_bps = safe_get(signals, "up_dip_bps") or safe_get(accumulate, "raw_pm_up_dip_bps")
    down_dip_bps = safe_get(signals, "down_dip_bps") or safe_get(accumulate, "raw_pm_down_dip_bps")
    up_dip = safe_get(signals, "up_dip", False) or safe_get(accumulate, "up_dip", False)
    down_dip = safe_get(signals, "down_dip", False) or safe_get(accumulate, "down_dip", False)
    
    action = safe_get(accumulate, "action", "WAIT")
    net_edge_up = safe_get(accumulate, "net_edge_up")
    net_edge_down = safe_get(accumulate, "net_edge_down")
    
    # Store distributions
    if ret_5s_bps is not None:
        stats.ret_5s_values.append(ret_5s_bps)
    if up_dip_bps is not None:
        stats.up_dip_values.append(up_dip_bps)
    if down_dip_bps is not None:
        stats.down_dip_values.append(down_dip_bps)
    
    stats.total_records += 1
    
    # (1) Spike sign sanity
    SPIKE_THRESHOLD_BPS = 5.0
    
    if ret_5s_bps is not None:
        if ret_5s_bps > SPIKE_THRESHOLD_BPS and down_spike:
            stats.spike_inversions += 1
            details = f"down_spike=True but ret={ret_5s_bps:.1f}bps"
            warnings.append(f"spike_inversion: {details}")
            stats.add_issue(ts, "spike_inversion", details)
        
        if ret_5s_bps < -SPIKE_THRESHOLD_BPS and up_spike:
            stats.spike_inversions += 1
            details = f"up_spike=True but ret={ret_5s_bps:.1f}bps"
            warnings.append(f"spike_inversion: {details}")
            stats.add_issue(ts, "spike_inversion", details)
    
    # (2) Dip sign sanity: dip should be <= 0 when flag is True
    if up_dip and up_dip_bps is not None and up_dip_bps > 0:
        stats.dip_inversions += 1
        details = f"up_dip=True but dip_bps={up_dip_bps:.1f}"
        warnings.append(f"dip_inversion: {details}")
        stats.add_issue(ts, "dip_inversion", details)
    
    if down_dip and down_dip_bps is not None and down_dip_bps > 0:
        stats.dip_inversions += 1
        details = f"down_dip=True but dip_bps={down_dip_bps:.1f}"
        warnings.append(f"dip_inversion: {details}")
        stats.add_issue(ts, "dip_inversion", details)
    
    # (3) Decision-trigger alignment
    if action == "ACCUMULATE_UP":
        if not (down_spike or up_dip):
            stats.decision_trigger_mismatches += 1
            details = f"BUY_UP without down_spike or up_dip"
            warnings.append(f"decision_trigger_mismatch: {details}")
            stats.add_issue(ts, "trigger_mismatch", details)
    
    if action == "ACCUMULATE_DOWN":
        if not (up_spike or down_dip):
            stats.decision_trigger_mismatches += 1
            details = f"BUY_DOWN without up_spike or down_dip"
            warnings.append(f"decision_trigger_mismatch: {details}")
            stats.add_issue(ts, "trigger_mismatch", details)
    
    # (4) Decision-edge alignment
    if action == "ACCUMULATE_UP":
        if net_edge_up is None or net_edge_up <= 0:
            stats.decision_edge_mismatches += 1
            details = f"BUY_UP with net_edge_up={net_edge_up}"
            warnings.append(f"decision_edge_mismatch: {details}")
            stats.add_issue(ts, "edge_mismatch", details)
    
    if action == "ACCUMULATE_DOWN":
        if net_edge_down is None or net_edge_down <= 0:
            stats.decision_edge_mismatches += 1
            details = f"BUY_DOWN with net_edge_down={net_edge_down}"
            warnings.append(f"decision_edge_mismatch: {details}")
            stats.add_issue(ts, "edge_mismatch", details)
    
    return warnings


def compute_distribution_stats(values: list[float], name: str) -> dict:
    """Compute distribution statistics."""
    if not values:
        return {"name": name, "count": 0}
    
    values_sorted = sorted(values)
    n = len(values)
    
    return {
        "name": name,
        "count": n,
        "min": values_sorted[0],
        "max": values_sorted[-1],
        "mean": sum(values) / n,
        "median": values_sorted[n // 2],
        "p10": values_sorted[int(n * 0.1)],
        "p90": values_sorted[int(n * 0.9)],
        "negative_pct": sum(1 for v in values if v < 0) / n * 100,
        "positive_pct": sum(1 for v in values if v > 0) / n * 100,
    }


# ============================================================
# Report Generation
# ============================================================

def print_report(stats: ValidationStats):
    """Print validation report."""
    print("\n" + "=" * 60)
    print("DATASET VALIDATION REPORT (CAL1)")
    print("=" * 60)
    
    print(f"\nTotal records analyzed: {stats.total_records}")
    
    # Inversion rates
    print("\n--- INVERSION RATES ---")
    if stats.total_records > 0:
        spike_pct = stats.spike_inversions / stats.total_records * 100
        dip_pct = stats.dip_inversions / stats.total_records * 100
        trigger_pct = stats.decision_trigger_mismatches / stats.total_records * 100
        edge_pct = stats.decision_edge_mismatches / stats.total_records * 100
        
        print(f"Spike inversions:           {stats.spike_inversions:5d} ({spike_pct:.2f}%)")
        print(f"Dip inversions:             {stats.dip_inversions:5d} ({dip_pct:.2f}%)")
        print(f"Decision-trigger mismatch:  {stats.decision_trigger_mismatches:5d} ({trigger_pct:.2f}%)")
        print(f"Decision-edge mismatch:     {stats.decision_edge_mismatches:5d} ({edge_pct:.2f}%)")
    
    # Distribution stats
    print("\n--- DISTRIBUTION STATS ---")
    
    for name, values in [
        ("ret_5s_bps", stats.ret_5s_values),
        ("up_dip_bps", stats.up_dip_values),
        ("down_dip_bps", stats.down_dip_values),
    ]:
        dist = compute_distribution_stats(values, name)
        if dist["count"] > 0:
            print(f"\n{name}:")
            print(f"  Count: {dist['count']}")
            print(f"  Range: [{dist['min']:.1f}, {dist['max']:.1f}]")
            print(f"  Mean: {dist['mean']:.2f}, Median: {dist['median']:.2f}")
            print(f"  P10: {dist['p10']:.2f}, P90: {dist['p90']:.2f}")
            print(f"  Negative: {dist['negative_pct']:.1f}%, Positive: {dist['positive_pct']:.1f}%")
    
    # Top problematic windows
    print("\n--- TOP PROBLEMATIC WINDOWS ---")
    
    window_counts = [
        (wid, len(issues))
        for wid, issues in stats.window_issues.items()
    ]
    window_counts.sort(key=lambda x: -x[1])
    
    for wid, count in window_counts[:10]:
        ts_start = wid * 15 * 60 * 1000
        dt = datetime.fromtimestamp(ts_start / 1000)
        issues = stats.window_issues[wid]
        issue_types = defaultdict(int)
        for i in issues:
            issue_types[i] += 1
        types_str = ", ".join(f"{k}:{v}" for k, v in issue_types.items())
        print(f"  Window {wid} ({dt:%Y-%m-%d %H:%M}): {count} issues [{types_str}]")
    
    if not window_counts:
        print("  No issues found!")
    
    # Summary
    total_issues = (
        stats.spike_inversions +
        stats.dip_inversions +
        stats.decision_trigger_mismatches +
        stats.decision_edge_mismatches
    )
    
    print("\n" + "=" * 60)
    if total_issues == 0:
        print("✅ ALL CHECKS PASSED - No signal inversions or mismatches")
    else:
        print(f"⚠️  FOUND {total_issues} TOTAL ISSUES - Review needed")
    print("=" * 60)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Validate recorded ticks for signal correctness (CAL1)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/decisions",
        help="Directory containing tick data (JSONL or Parquet)",
    )
    parser.add_argument(
        "--last-n-windows",
        type=int,
        default=50,
        help="Number of recent windows/files to analyze",
    )
    parser.add_argument(
        "--output",
        help="Output file for detailed issues (JSON)",
    )
    
    args = parser.parse_args()
    
    # Load data
    records = load_data_files(args.data_dir, args.last_n_windows)
    
    if not records:
        print("No records found. Make sure data directory contains JSONL or Parquet files.")
        sys.exit(1)
    
    # Validate
    stats = ValidationStats()
    
    for record in records:
        validate_record(record, stats)
    
    # Print report
    print_report(stats)
    
    # Save detailed issues if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({
                "total_records": stats.total_records,
                "spike_inversions": stats.spike_inversions,
                "dip_inversions": stats.dip_inversions,
                "decision_trigger_mismatches": stats.decision_trigger_mismatches,
                "decision_edge_mismatches": stats.decision_edge_mismatches,
                "issues": stats.issues[:1000],  # Limit to 1000 issues
            }, f, indent=2)
        print(f"\nDetailed issues saved to: {args.output}")


if __name__ == "__main__":
    main()
