#!/usr/bin/env python3
"""
TUI Dashboard for Accumulate Strategy (BLOCKS 4-9)
=================================================

Live terminal dashboard showing:
- Trade mode + reasons
- UP/DOWN market prices + fair values
- Net edge + required edge
- Spikes & dips signals
- Accumulation recommendation

Usage:
    python scripts/tui_dashboard.py --url http://127.0.0.1:8000

Requirements:
    pip install rich httpx
"""

import argparse
import sys
import time
from datetime import datetime
from typing import Any, Optional

try:
    import httpx
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
except ImportError:
    print("Missing dependencies. Install with: pip install rich httpx")
    sys.exit(1)

# ============================================================
# Configuration
# ============================================================

REFRESH_INTERVAL = 1.0
API_TIMEOUT = 2.0


# ============================================================
# Helpers
# ============================================================

def safe_get(data: dict, path: str, default: Any = None) -> Any:
    """Safely get nested dict value by dot path."""
    if data is None:
        return default
    keys = path.split(".")
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
        if data is None:
            return default
    return data


def fmt(val: Any, decimals: int = 2, default: str = "—") -> str:
    """Format float or return default."""
    if val is None:
        return default
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def fmt_pct(val: Any, decimals: int = 1, default: str = "—") -> str:
    """Format as percentage."""
    if val is None:
        return default
    try:
        return f"{float(val) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return default


def fmt_bps(val: Any, decimals: int = 0, default: str = "—") -> str:
    """Format as basis points."""
    if val is None:
        return default
    try:
        return f"{float(val):.{decimals}f}"
    except (TypeError, ValueError):
        return default


def mode_color(mode: str) -> str:
    """Get color for mode."""
    if mode == "OK":
        return "green"
    elif mode == "DEGRADED":
        return "yellow"
    elif mode == "BAD":
        return "red"
    return "white"


def bool_indicator(val: bool, true_text: str = "●", false_text: str = "○") -> Text:
    """Return colored indicator."""
    if val:
        return Text(true_text, style="green bold")
    return Text(false_text, style="dim")


# ============================================================
# API Client
# ============================================================

class DashboardClient:
    """API client for dashboard data."""
    
    def __init__(self, base_url: str, timeout: float = API_TIMEOUT):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
        self.features: Optional[dict] = None
        self.accumulate: Optional[dict] = None
        self.last_error: Optional[str] = None
    
    def fetch_all(self) -> bool:
        """Fetch all endpoints. Returns True if successful."""
        try:
            # Fetch features
            resp = self.client.get(f"{self.base_url}/latest/features")
            if resp.status_code == 200:
                self.features = resp.json()
            
            # Fetch accumulate
            resp = self.client.get(f"{self.base_url}/latest/accumulate")
            if resp.status_code == 200:
                self.accumulate = resp.json()
            
            self.last_error = None
            return True
            
        except httpx.TimeoutException:
            self.last_error = "Timeout"
            return False
        except httpx.RequestError as e:
            self.last_error = str(type(e).__name__)
            return False
        except Exception as e:
            self.last_error = str(e)
            return False
    
    def close(self):
        self.client.close()


# ============================================================
# Panel Builders
# ============================================================

def build_status_panel(features: dict, accum: dict) -> Panel:
    """Build STATUS panel with trade_mode and reasons."""
    quality = safe_get(features, "quality") or {}
    trade_mode = quality.get("trade_mode", "UNKNOWN")
    trade_reasons = quality.get("trade_reasons", [])
    degrade_reasons = quality.get("degrade_reasons", [])
    
    # Build status line
    parts = []
    
    # Trade mode
    parts.append(Text(f"TRADE: {trade_mode}", style=mode_color(trade_mode)))
    
    # Data mode
    data_mode = quality.get("mode", "UNKNOWN")
    parts.append(Text(f"DATA: {data_mode}", style=mode_color(data_mode)))
    
    # Ages
    pm_age = quality.get("polymarket_age_sec")
    taapi_age = quality.get("taapi_age_sec")
    ages = f"PM:{fmt(pm_age, 1)}s TAAPI:{fmt(taapi_age, 1)}s"
    parts.append(Text(ages, style="dim"))
    
    # Reasons
    all_reasons = trade_reasons + degrade_reasons
    if all_reasons:
        reasons_str = ", ".join(all_reasons[:3])
        parts.append(Text(f"[{reasons_str}]", style="yellow"))
    
    status_line = Text(" │ ").join(parts)
    
    return Panel(status_line, title="STATUS", height=3, border_style="blue")


def build_market_panel(features: dict) -> Panel:
    """Build MARKET panel showing UP/DOWN prices."""
    pm_updown = safe_get(features, "polymarket_up_down") or {}
    up = pm_updown.get("up") or {}
    down = pm_updown.get("down") or {}
    
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("", width=6)
    table.add_column("MID", justify="right", width=7)
    table.add_column("ASK", justify="right", width=7, style="dim")
    table.add_column("SPREAD", justify="right", width=7, style="dim")
    table.add_column("DEPTH", justify="right", width=7, style="dim")
    table.add_column("SLICE", justify="center", width=5)
    
    # UP row
    table.add_row(
        Text("UP", style="bold green"),
        fmt_pct(up.get("mid"), 1),
        fmt_pct(up.get("ask"), 1),
        fmt_bps(up.get("spread_bps")),
        fmt(up.get("depth_top1_usd"), 0),
        bool_indicator(up.get("slice_ok", False), "OK", "—"),
    )
    
    # DOWN row
    table.add_row(
        Text("DOWN", style="bold red"),
        fmt_pct(down.get("mid"), 1),
        fmt_pct(down.get("ask"), 1),
        fmt_bps(down.get("spread_bps")),
        fmt(down.get("depth_top1_usd"), 0),
        bool_indicator(down.get("slice_ok", False), "OK", "—"),
    )
    
    return Panel(table, title="MARKET (UP/DOWN)", border_style="cyan")


def build_fair_panel(features: dict) -> Panel:
    """Build FAIR panel showing fair_fast and fair_smooth (S3)."""
    fair = safe_get(features, "fair") or {}
    fair_status = fair.get("status") or safe_get(features, "fair_status") or {}
    market_ref = safe_get(features, "market_ref") or {}
    vol = safe_get(features, "vol") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", width=12)
    table.add_column("Value", width=14, justify="right")
    
    # S3: Get fast and smooth fair
    fair_fast = fair.get("fast") or {}
    fair_smooth = fair.get("smooth") or {}
    
    fast_ready = fair_status.get("fast_ready", False)
    smooth_ready = fair_status.get("smooth_ready", False)
    fast_reason = fair_status.get("fast_reason", "unknown")
    smooth_reason = fair_status.get("smooth_reason", "unknown")
    
    # FAST fair (S_raw + σ_fast)
    fast_up = fair_fast.get("up")
    fast_down = fair_fast.get("down")
    fast_z = fair_fast.get("z")
    
    if fast_ready and fast_up is not None:
        table.add_row("FAST UP", Text(fmt_pct(fast_up, 1), style="bold green"))
        table.add_row("FAST DN", Text(fmt_pct(fast_down, 1), style="bold red"))
        table.add_row("FAST z", fmt(fast_z, 2))
    else:
        table.add_row("FAST UP", Text("N/A", style="dim"))
        table.add_row("FAST DN", Text("N/A", style="dim"))
        reason_short = fast_reason[:16] if len(fast_reason) > 16 else fast_reason
        table.add_row("FAST", Text(reason_short, style="yellow"))
    
    table.add_row("", "")
    
    # SMOOTH fair (S_smooth + σ_blend)
    smooth_up = fair_smooth.get("up")
    smooth_down = fair_smooth.get("down")
    smooth_z = fair_smooth.get("z")
    
    if smooth_ready and smooth_up is not None:
        table.add_row("SMTH UP", Text(fmt_pct(smooth_up, 1), style="bold cyan"))
        table.add_row("SMTH DN", Text(fmt_pct(smooth_down, 1), style="bold magenta"))
        table.add_row("SMTH z", fmt(smooth_z, 2))
    else:
        table.add_row("SMTH UP", Text("N/A", style="dim"))
        table.add_row("SMTH DN", Text("N/A", style="dim"))
        reason_short = smooth_reason[:16] if len(smooth_reason) > 16 else smooth_reason
        table.add_row("SMTH", Text(reason_short, style="yellow"))
    
    table.add_row("", "")
    
    # Reference and prices
    table.add_row("Ref PX", fmt(market_ref.get("ref_px"), 1))
    s_raw = market_ref.get("S_now_raw")
    s_smooth = market_ref.get("S_now_smooth")
    table.add_row("S raw", fmt(s_raw, 1))
    table.add_row("S smth", fmt(s_smooth, 1))
    table.add_row("Tau", f"{fmt(market_ref.get('tau_sec'), 0)}s")
    
    # Sigma values (compact)
    sig_fast = vol.get("sigma_fast_15m")
    sig_blend = vol.get("sigma_blend_15m")
    n_bars = vol.get("n_bars", 0)
    
    if sig_fast is not None:
        table.add_row("σF/σB", f"{fmt(sig_fast, 4)}/{fmt(sig_blend, 4)}")
    else:
        table.add_row("σ", Text(f"warmup ({n_bars})", style="yellow"))
    
    # Status indicator
    any_ready = fast_ready or smooth_ready
    status_text = "✓ READY" if any_ready else "○ WARMUP"
    status_style = "green" if any_ready else "yellow"
    
    return Panel(table, title=f"FAIR MODEL [{status_text}]", border_style=status_style)


def build_edge_panel(features: dict, accum: dict) -> Panel:
    """Build NET EDGE panel."""
    pm_updown = safe_get(features, "polymarket_up_down") or {}
    up = pm_updown.get("up") or {}
    down = pm_updown.get("down") or {}
    
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("", width=6)
    table.add_column("EDGE", justify="right", width=8)
    table.add_column("REQ", justify="right", width=8, style="dim")
    table.add_column("NET", justify="right", width=8)
    
    # Get edge values from accumulate decision
    edge_up = accum.get("edge_up_bps") if accum else None
    edge_down = accum.get("edge_down_bps") if accum else None
    net_up = accum.get("net_edge_up") if accum else None
    net_down = accum.get("net_edge_down") if accum else None
    req_up = up.get("required_edge_bps")
    req_down = down.get("required_edge_bps")
    
    # Style for net edge
    up_style = "green bold" if net_up and net_up > 0 else "white"
    down_style = "green bold" if net_down and net_down > 0 else "white"
    
    # UP row
    table.add_row(
        Text("UP", style="bold green"),
        fmt_bps(edge_up),
        fmt_bps(req_up),
        Text(fmt_bps(net_up), style=up_style),
    )
    
    # DOWN row
    table.add_row(
        Text("DOWN", style="bold red"),
        fmt_bps(edge_down),
        fmt_bps(req_down),
        Text(fmt_bps(net_down), style=down_style),
    )
    
    return Panel(table, title="NET EDGE (bps)", border_style="magenta")


def build_bias_panel(features: dict) -> Panel:
    """Build BIAS panel showing directional context (S4)."""
    bias = safe_get(features, "bias") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Label", width=10)
    table.add_column("Value", width=12, justify="right")
    
    # Overall bias
    bias_dir = bias.get("dir", "N/A")
    strength = bias.get("strength", 0)
    bias_up_prob = bias.get("bias_up_prob", 0.5)
    
    # Color based on direction
    dir_style = "green bold" if bias_dir == "UP" else "red bold" if bias_dir == "DOWN" else "yellow"
    table.add_row("Direction", Text(bias_dir, style=dir_style))
    
    # Strength bar
    strength_pct = int(strength * 100)
    filled = int(strength * 10)
    bar = "█" * filled + "░" * (10 - filled)
    table.add_row("Strength", f"{bar} {strength_pct}%")
    
    # Probability
    table.add_row("P(UP)", fmt_pct(bias_up_prob, 1))
    
    table.add_row("", "")
    
    # TF breakdown (compact)
    tf_breakdown = bias.get("tf_breakdown", {})
    for tf in ["1m", "5m", "15m", "1h"]:
        tf_data = tf_breakdown.get(tf, {})
        tf_dir = tf_data.get("dir", "N/A")
        tf_style = "green" if tf_dir == "UP" else "red" if tf_dir == "DOWN" else "dim"
        tf_bars = tf_data.get("n_bars", 0)
        table.add_row(tf, Text(f"{tf_dir} ({tf_bars})", style=tf_style))
    
    return Panel(table, title="BIAS (S4)", border_style="blue")


def build_signals_panel(features: dict) -> Panel:
    """Build SPIKES & DIPS panel."""
    spikes = safe_get(features, "spikes") or {}
    pm_dips = safe_get(features, "pm_dips") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Signal", width=14)
    table.add_column("Value", width=10, justify="right")
    table.add_column("Flag", width=4, justify="center")
    
    # Binance spikes
    table.add_row(
        Text("Binance 5s", style="cyan"),
        f"{fmt_bps(spikes.get('ret_5s_bps'))} bps",
        "",
    )
    table.add_row(
        "  Z-score",
        fmt(spikes.get("z_ret_5s"), 2),
        "",
    )
    table.add_row(
        "  Down spike",
        "",
        bool_indicator(spikes.get("down_spike_5s", False), "↓", "—"),
    )
    table.add_row(
        "  Up spike",
        "",
        bool_indicator(spikes.get("up_spike_5s", False), "↑", "—"),
    )
    
    table.add_row("", "", "")
    
    # PM dips
    table.add_row(
        Text("PM Dips", style="cyan"),
        "",
        "",
    )
    table.add_row(
        "  UP dip",
        f"{fmt_bps(pm_dips.get('up_dip_bps'))} bps",
        bool_indicator(pm_dips.get("up_dip", False)),
    )
    table.add_row(
        "  DOWN dip",
        f"{fmt_bps(pm_dips.get('down_dip_bps'))} bps",
        bool_indicator(pm_dips.get("down_dip", False)),
    )
    
    return Panel(table, title="SPIKES & DIPS", border_style="yellow")


def build_recommendation_panel(accum: dict, features: dict) -> Panel:
    """Build RECOMMENDATION panel with detailed veto reasons."""
    if not accum:
        return Panel(
            Text("No accumulate decision", style="dim"),
            title="RECOMMENDATION",
            border_style="white",
        )
    
    action = accum.get("action", "WAIT")
    reasons = accum.get("reasons", [])
    veto_reasons = accum.get("veto_reasons", [])
    
    # Get fair_status for detailed reason
    fair_status = safe_get(features, "fair_status") or {}
    fair_reason = fair_status.get("reason", "")
    
    # Action styling
    if action == "ACCUMULATE_UP":
        action_style = "bold green on dark_green"
        action_text = ">>> BUY UP <<<"
    elif action == "ACCUMULATE_DOWN":
        action_style = "bold red on dark_red"
        action_text = ">>> BUY DOWN <<<"
    else:
        action_style = "dim"
        action_text = "WAIT"
    
    lines = []
    lines.append(Text(action_text, style=action_style, justify="center"))
    lines.append(Text(""))
    
    # Triggers
    triggers = accum.get("triggers", {})
    trigger_parts = []
    if triggers.get("down_spike"):
        trigger_parts.append("↓spike")
    if triggers.get("up_spike"):
        trigger_parts.append("↑spike")
    if triggers.get("up_dip"):
        trigger_parts.append("up_dip")
    if triggers.get("down_dip"):
        trigger_parts.append("dn_dip")
    
    if trigger_parts:
        lines.append(Text(f"Triggers: {', '.join(trigger_parts)}", style="cyan"))
    else:
        lines.append(Text("Triggers: none", style="dim"))
    
    # Reasons
    if reasons:
        lines.append(Text(f"Go: {', '.join(reasons[:2])}", style="green"))
    
    # Veto with detail
    if veto_reasons:
        veto_str = veto_reasons[0] if veto_reasons else ""
        # Shorten for display
        if len(veto_str) > 35:
            veto_str = veto_str[:32] + "..."
        lines.append(Text(f"Veto: {veto_str}", style="red"))
    
    # Budget
    slices = accum.get("slices_this_window", 0)
    usd = accum.get("usd_this_window", 0)
    lines.append(Text(f"Budget: {slices}/30, ${usd:.0f}/$300", style="dim"))
    
    content = Text("\n").join(lines)
    
    return Panel(content, title="RECOMMENDATION", border_style="white", height=10)


def build_footer_panel(client: DashboardClient) -> Panel:
    """Build footer panel."""
    parts = []
    
    if client.last_error:
        parts.append(Text(f"Error: {client.last_error}", style="red"))
    else:
        parts.append(Text(f"Updated: {datetime.now().strftime('%H:%M:%S')}", style="dim"))
    
    text = Text(" │ ").join(parts)
    
    return Panel(text, height=3, border_style="dim")


# ============================================================
# Main Layout
# ============================================================

def build_layout(features: dict, accum: dict, client: DashboardClient) -> Layout:
    """Build complete layout."""
    layout = Layout()
    
    # Main structure
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    
    # Body split into left, center, right
    layout["body"].split_row(
        Layout(name="left"),
        Layout(name="center"),
        Layout(name="right"),
    )
    
    # Left column
    layout["left"].split_column(
        Layout(name="market"),
        Layout(name="fair"),
    )
    
    # Center column (S4 bias)
    layout["center"].split_column(
        Layout(name="bias"),
        Layout(name="signals"),
    )
    
    # Right column
    layout["right"].split_column(
        Layout(name="edge"),
        Layout(name="recommendation"),
    )
    
    # Assign panels
    features = features or {}
    accum = accum or {}
    
    layout["header"].update(build_status_panel(features, accum))
    layout["market"].update(build_market_panel(features))
    layout["fair"].update(build_fair_panel(features))
    layout["bias"].update(build_bias_panel(features))
    layout["edge"].update(build_edge_panel(features, accum))
    layout["signals"].update(build_signals_panel(features))
    layout["recommendation"].update(build_recommendation_panel(accum, features))
    layout["footer"].update(build_footer_panel(client))
    
    return layout


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TUI Dashboard for Accumulate Strategy")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="API base URL")
    parser.add_argument("--refresh", type=float, default=REFRESH_INTERVAL, help="Refresh interval")
    args = parser.parse_args()
    
    console = Console()
    client = DashboardClient(args.url)
    
    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                try:
                    client.fetch_all()
                    layout = build_layout(client.features, client.accumulate, client)
                    live.update(layout)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    client.last_error = str(e)
                
                time.sleep(args.refresh)
    
    except KeyboardInterrupt:
        pass
    finally:
        client.close()
        console.print("\n[dim]Dashboard stopped.[/dim]")


if __name__ == "__main__":
    main()
