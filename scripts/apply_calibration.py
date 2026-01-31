#!/usr/bin/env python3
"""
CAL1 D1: Apply Calibration Results
===================================

Reads calibration results and generates config suggestions.

Usage:
    python scripts/apply_calibration.py --results results/calib_top.csv --best results/best_params.json

Output:
    - Prints suggested config changes
    - Optionally writes to config.py (with confirmation)
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def load_best_params(params_path: Path) -> dict:
    """Load best parameters from JSON."""
    if not params_path.exists():
        print(f"Error: Best params file not found: {params_path}")
        sys.exit(1)
    
    with open(params_path) as f:
        return json.load(f)


def load_top_results(results_path: Path, n: int = 10) -> list[dict]:
    """Load top results from CSV."""
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)
    
    results = []
    with open(results_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append(row)
            if len(results) >= n:
                break
    
    return results


def generate_config_snippet(params: dict) -> str:
    """Generate Python config snippet."""
    return f'''
# ============================================================
# CAL1 Calibrated Thresholds
# ============================================================
# Generated from calibration results
# Metrics: avg_move_all={params.get("metrics", {}).get("avg_move_all_bps", "N/A")} bps
#          winrate_up={params.get("metrics", {}).get("winrate_up", "N/A")}%
#          winrate_down={params.get("metrics", {}).get("winrate_down", "N/A")}%

# Spike detection
SPIKE_Z_THRESHOLD: float = {params.get("SPIKE_Z_TH", 2.0)}

# Dip detection
DIP_THRESHOLD_BPS: float = {params.get("DIP_TH_BPS", 80)}

# Edge calculation
EDGE_BUFFER_BPS: float = {params.get("EDGE_BUFFER_BPS", 25)}

# Signal generation
NET_EDGE_MIN_BPS: float = {params.get("NET_EDGE_MIN_BPS", 0)}

# Evaluation horizon (for calibration only)
CALIBRATION_HORIZON_SEC: int = {params.get("H_SEC", 60)}
'''


def main():
    parser = argparse.ArgumentParser(
        description="Apply calibration results to config (CAL1)"
    )
    parser.add_argument(
        "--results",
        default="results/calib_top.csv",
        help="Path to calibration results CSV",
    )
    parser.add_argument(
        "--best",
        default="results/best_params.json",
        help="Path to best params JSON",
    )
    parser.add_argument(
        "--output",
        help="Output file for config snippet (optional)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually apply to config.py (requires confirmation)",
    )
    
    args = parser.parse_args()
    
    # Load data
    params = load_best_params(Path(args.best))
    
    print("=" * 60)
    print("CALIBRATION RESULTS (CAL1)")
    print("=" * 60)
    
    print("\nBest Parameters:")
    print(f"  SPIKE_Z_TH:       {params.get('SPIKE_Z_TH', 'N/A')}")
    print(f"  DIP_TH_BPS:       {params.get('DIP_TH_BPS', 'N/A')}")
    print(f"  EDGE_BUFFER_BPS:  {params.get('EDGE_BUFFER_BPS', 'N/A')}")
    print(f"  NET_EDGE_MIN_BPS: {params.get('NET_EDGE_MIN_BPS', 'N/A')}")
    print(f"  H_SEC:            {params.get('H_SEC', 'N/A')}")
    
    metrics = params.get("metrics", {})
    print("\nMetrics:")
    print(f"  Avg Move All:   {metrics.get('avg_move_all_bps', 'N/A')} bps")
    print(f"  Winrate UP:     {metrics.get('winrate_up', 'N/A')}%")
    print(f"  Winrate DOWN:   {metrics.get('winrate_down', 'N/A')}%")
    print(f"  Total Signals:  {metrics.get('total_signals', 'N/A')}")
    
    # Generate config snippet
    snippet = generate_config_snippet(params)
    
    print("\n" + "-" * 60)
    print("SUGGESTED CONFIG SNIPPET")
    print("-" * 60)
    print(snippet)
    
    # Save to file if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(snippet)
        print(f"\nSnippet saved to: {args.output}")
    
    # Apply to config.py if requested
    if args.apply:
        print("\n" + "=" * 60)
        print("WARNING: This will modify src/collector/config.py")
        print("=" * 60)
        
        response = input("Continue? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            return
        
        config_path = Path("src/collector/config.py")
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            return
        
        # Read existing config
        with open(config_path) as f:
            config_content = f.read()
        
        # Check if CAL1 section exists
        if "# CAL1 Calibrated Thresholds" in config_content:
            print("CAL1 section already exists in config.py")
            print("Please update manually or remove the existing section first.")
            return
        
        # Find Settings class and add after
        # This is a simple append - manual review recommended
        print("Note: Appending to end of config.py")
        print("Manual review and placement recommended.")
        
        with open(config_path, "a") as f:
            f.write("\n" + snippet)
        
        print(f"Config updated: {config_path}")
        print("Please review and adjust placement as needed.")
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Review the suggested thresholds")
    print("2. Update src/collector/config.py manually or use --apply")
    print("3. Update spike_detector.py threshold if needed")
    print("4. Update pm_dip_detector.py threshold if needed")
    print("5. Restart collector to apply changes")
    print("6. Monitor sanity_warnings in logs")


if __name__ == "__main__":
    main()
