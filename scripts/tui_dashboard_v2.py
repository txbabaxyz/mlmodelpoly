#!/usr/bin/env python3
"""
TUI Dashboard v2 (S8)
=====================

Unified dashboard showing all layers:
- Polymarket UP/DOWN prices
- Fair (fast/smooth)
- Volatility (fast/slow/blend)
- Bias (multi-TF)
- Events (spikes/dips)
- Net Edge (fast/smooth)
- Decision (action + confidence)

Usage:
    python scripts/tui_dashboard_v2.py
    python scripts/tui_dashboard_v2.py --host localhost --port 8000
"""

import argparse
import httpx
import time
from typing import Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


# ============================================================
# Helpers
# ============================================================

def safe_get(d: Optional[dict], path: str, default=None):
    """Safely get nested dict value by dot-separated path."""
    if d is None:
        return default
    keys = path.split(".")
    val = d
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def fmt(val, decimals: int = 2) -> str:
    """Format number or return '-'."""
    if val is None:
        return "-"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def fmt_pct(val, decimals: int = 1) -> str:
    """Format as percentage."""
    if val is None:
        return "-"
    return f"{val * 100:.{decimals}f}%"


def fmt_bps(val) -> str:
    """Format basis points."""
    if val is None:
        return "-"
    return f"{val:+.0f}"


def color_val(val, threshold_good=0, threshold_bad=0, invert=False) -> Text:
    """Color value based on thresholds."""
    if val is None:
        return Text("-", style="dim")
    if invert:
        if val < threshold_bad:
            style = "green"
        elif val > threshold_good:
            style = "red"
        else:
            style = "white"
    else:
        if val > threshold_good:
            style = "green"
        elif val < threshold_bad:
            style = "red"
        else:
            style = "white"
    return Text(fmt(val, 1), style=style)


# ============================================================
# API Client
# ============================================================

class DashboardClient:
    """HTTP client for collector API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=5.0)
        self.last_features: Optional[dict] = None
        self.last_edge: Optional[dict] = None
        self.last_error: Optional[str] = None
        self.fetch_count = 0
    
    def fetch(self) -> tuple[Optional[dict], Optional[dict]]:
        """Fetch features and edge from API."""
        try:
            # Fetch features
            resp_features = self.client.get(f"{self.base_url}/latest/features")
            if resp_features.status_code == 200:
                self.last_features = resp_features.json()
            
            # Fetch edge (S6 decision v2)
            resp_edge = self.client.get(f"{self.base_url}/latest/edge")
            if resp_edge.status_code == 200:
                self.last_edge = resp_edge.json()
            
            self.last_error = None
            self.fetch_count += 1
            
        except Exception as e:
            self.last_error = str(e)[:50]
        
        return self.last_features, self.last_edge


# ============================================================
# Panel Builders
# ============================================================

def build_status_panel(features: dict, edge: dict) -> Panel:
    """Build STATUS panel."""
    quality = safe_get(features, "quality") or {}
    fair_status = safe_get(features, "fair.status") or {}
    
    mode = quality.get("mode", "?")
    trade_mode = quality.get("trade_mode", "?")
    trade_reasons = quality.get("trade_reasons", [])
    
    # Colors
    mode_style = "green" if mode == "OK" else "yellow" if mode == "DEGRADED" else "red"
    trade_style = "green" if trade_mode == "OK" else "yellow" if trade_mode == "DEGRADED" else "red"
    
    parts = []
    parts.append(Text(f"MODE: ", style="bold"))
    parts.append(Text(mode, style=mode_style))
    parts.append(Text("  TRADE: ", style="bold"))
    parts.append(Text(trade_mode, style=trade_style))
    
    if trade_reasons:
        parts.append(Text(f" ({', '.join(trade_reasons[:2])})", style="dim"))
    
    # Fair status
    fast_ready = fair_status.get("fast_ready", False)
    smooth_ready = fair_status.get("smooth_ready", False)
    fast_reason = fair_status.get("fast_reason", "?")
    smooth_reason = fair_status.get("smooth_reason", "?")
    
    parts.append(Text("\nFAIR: ", style="bold"))
    parts.append(Text("F:" + ("✓" if fast_ready else fast_reason[:10]), 
                      style="green" if fast_ready else "yellow"))
    parts.append(Text(" S:" + ("✓" if smooth_ready else smooth_reason[:10]),
                      style="green" if smooth_ready else "yellow"))
    
    text = Text()
    for p in parts:
        text.append(p)
    
    return Panel(text, title="STATUS", border_style="blue")


def build_polymarket_panel(features: dict) -> Panel:
    """Build POLYMARKET panel."""
    pm = safe_get(features, "polymarket_up_down") or {}
    pm_raw = safe_get(features, "polymarket") or {}
    up = pm.get("up") or {}
    down = pm.get("down") or {}
    
    # Fallback to raw polymarket data
    yes_raw = pm_raw.get("yes") or {}
    no_raw = pm_raw.get("no") or {}
    
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("", width=6)
    table.add_column("MID", justify="right", width=6)
    table.add_column("ASK", justify="right", width=6)
    table.add_column("SPR", justify="right", width=5)
    table.add_column("D3", justify="right", width=6)
    table.add_column("OK", width=3)
    
    # UP row (YES token)
    up_mid = up.get("mid") or yes_raw.get("mid")
    up_ask = up.get("ask") or up.get("best_ask") or yes_raw.get("best_ask")
    up_spread = up.get("spread_bps") or yes_raw.get("spread_bps")
    up_depth = up.get("depth_top3_usd") or up.get("depth_top3") or yes_raw.get("depth_top3")
    up_ok = "✓" if up.get("slice_ok") else "✗"
    
    table.add_row(
        Text("UP", style="bold green"),
        fmt(up_mid, 3),
        fmt(up_ask, 3),
        fmt(up_spread, 0) if up_spread and up_spread < 10000 else "-",
        fmt(up_depth, 0),
        Text(up_ok, style="green" if up.get("slice_ok") else "red"),
    )
    
    # DOWN row (NO token)
    down_mid = down.get("mid") or no_raw.get("mid")
    down_ask = down.get("ask") or down.get("best_ask") or no_raw.get("best_ask")
    down_spread = down.get("spread_bps") or no_raw.get("spread_bps")
    down_depth = down.get("depth_top3_usd") or down.get("depth_top3") or no_raw.get("depth_top3")
    down_ok = "✓" if down.get("slice_ok") else "✗"
    
    table.add_row(
        Text("DOWN", style="bold red"),
        fmt(down_mid, 3),
        fmt(down_ask, 3),
        fmt(down_spread, 0) if down_spread and down_spread < 10000 else "-",
        fmt(down_depth, 0),
        Text(down_ok, style="green" if down.get("slice_ok") else "red"),
    )
    
    # Connection status
    connected = pm_raw.get("connected", False)
    age = pm_raw.get("age_sec")
    status = f"{'✓' if connected else '✗'} {fmt(age, 1)}s" if age else ("✓" if connected else "✗")
    table.add_row("", "", "", "", "", "")
    table.add_row(Text("WS", style="dim"), Text(status, style="green" if connected else "red"), "", "", "", "")
    
    return Panel(table, title="POLYMARKET", border_style="magenta")


def build_fair_panel(features: dict) -> Panel:
    """Build FAIR panel (fast/smooth)."""
    fair = safe_get(features, "fair") or {}
    fast = fair.get("fast") or {}
    smooth = fair.get("smooth") or {}
    
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("", width=6)
    table.add_column("UP", justify="right", width=7)
    table.add_column("DOWN", justify="right", width=7)
    table.add_column("Z", justify="right", width=6)
    
    # FAST
    table.add_row(
        Text("FAST", style="bold cyan"),
        fmt_pct(fast.get("up"), 1),
        fmt_pct(fast.get("down"), 1),
        fmt(fast.get("z"), 2),
    )
    
    # SMOOTH
    table.add_row(
        Text("SMOOTH", style="bold yellow"),
        fmt_pct(smooth.get("up"), 1),
        fmt_pct(smooth.get("down"), 1),
        fmt(smooth.get("z"), 2),
    )
    
    return Panel(table, title="FAIR MODEL", border_style="cyan")


def build_vol_panel(features: dict) -> Panel:
    """Build VOLATILITY panel."""
    vol = safe_get(features, "vol") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("L", width=6)
    table.add_column("V", width=20)
    
    sig_fast = vol.get("sigma_fast_15m")
    sig_slow = vol.get("sigma_slow_15m")
    sig_blend = vol.get("sigma_blend_15m")
    blend_w = vol.get("blend_w")
    n_bars = vol.get("n_bars", 0)
    reason = vol.get("reason", "")
    
    if sig_fast is not None:
        table.add_row("σF", fmt(sig_fast, 4))
        table.add_row("σS", fmt(sig_slow, 4))
        table.add_row("σB", f"{fmt(sig_blend, 4)} w={fmt(blend_w, 2)}")
        table.add_row("bars", str(n_bars))
    else:
        table.add_row("σ", Text(f"warmup ({n_bars})", style="yellow"))
        table.add_row("", Text(reason[:16], style="dim"))
    
    return Panel(table, title="VOLATILITY", border_style="green")


def build_bias_panel(features: dict) -> Panel:
    """Build BIAS panel with TAAPI integration."""
    bias = safe_get(features, "bias") or {}
    taapi = safe_get(features, "taapi_context") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("L", width=10)
    table.add_column("V", width=18)
    
    # Overall
    bias_dir = bias.get("dir", "?")
    strength = bias.get("strength", 0)
    bias_up = bias.get("bias_up_prob", 0.5)
    taapi_integrated = bias.get("taapi_integrated", False)
    
    dir_style = "green bold" if bias_dir == "UP" else "red bold" if bias_dir == "DOWN" else "yellow"
    
    # Show TAAPI integration status
    integ_mark = "✓" if taapi_integrated else "✗"
    integ_style = "green" if taapi_integrated else "red"
    table.add_row("Direction", Text(bias_dir, style=dir_style))
    table.add_row("TAAPI", Text(integ_mark, style=integ_style))
    
    # Strength bar
    filled = int(strength * 10)
    bar = "█" * filled + "░" * (10 - filled)
    table.add_row("Strength", f"{bar} {int(strength*100)}%")
    
    table.add_row("P(UP)", fmt_pct(bias_up, 1))
    
    # Own score vs TAAPI score
    own_score = bias.get("own_score")
    taapi_score = bias.get("taapi_score")
    if own_score is not None:
        table.add_row("Own", fmt_pct(own_score, 1))
    if taapi_score is not None:
        table.add_row("TAAPI", fmt_pct(taapi_score, 1))
    
    # TF breakdown (compact)
    tf_bd = bias.get("tf_breakdown", {})
    for tf in ["1m", "5m", "15m", "1h"]:
        tf_data = tf_bd.get(tf, {})
        tf_dir = tf_data.get("dir", "?")
        n_bars = tf_data.get("n_bars", 0)
        taapi_bias = tf_data.get("taapi_bias", "?")
        
        own_sym = "↑" if tf_dir == "UP" else "↓" if tf_dir == "DOWN" else "→"
        taapi_sym = "↑" if taapi_bias == "UP" else "↓" if taapi_bias == "DOWN" else "→"
        
        style = "green" if tf_dir == "UP" else "red" if tf_dir == "DOWN" else "yellow"
        text = f"{own_sym}({n_bars}) T:{taapi_sym}"
        table.add_row(tf, Text(text, style=style))
    
    # TAAPI regime and alignment
    regime = taapi.get("regime_15m", "?")
    alignment = taapi.get("alignment_score", 0)
    table.add_row("Regime", regime)
    table.add_row("Align", str(alignment))
    
    return Panel(table, title="BIAS + TAAPI", border_style="blue")


def build_events_panel(features: dict) -> Panel:
    """Build EVENTS panel (spikes/dips)."""
    signals = safe_get(features, "signals") or {}
    spikes = safe_get(features, "spikes") or {}
    pm_dips = safe_get(features, "pm_dips") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("L", width=10)
    table.add_column("V", width=16)
    
    # Binance - prefer spikes over signals
    ret_5s = spikes.get("ret_5s_bps") or signals.get("ret_5s_bps")
    z_ret = spikes.get("z_ret_5s") or signals.get("z_ret_5s")
    up_spike = spikes.get("up_spike_5s") or signals.get("up_spike_5s")
    down_spike = spikes.get("down_spike_5s") or signals.get("down_spike_5s")
    
    table.add_row("ret_5s", f"{fmt_bps(ret_5s)} bps")
    table.add_row("z_ret", fmt(z_ret, 2))
    
    spike_str = ""
    if up_spike:
        spike_str += "↑SPIKE "
    if down_spike:
        spike_str += "↓SPIKE"
    if not spike_str:
        spike_str = "-"
    spike_style = "yellow bold" if (up_spike or down_spike) else "dim"
    table.add_row("BN spike", Text(spike_str, style=spike_style))
    
    # PM dips
    up_dip = pm_dips.get("up_dip", False)
    down_dip = pm_dips.get("down_dip", False)
    up_dip_bps = pm_dips.get("up_dip_bps")
    down_dip_bps = pm_dips.get("down_dip_bps")
    
    dip_str = ""
    if up_dip:
        dip_str += f"UP({fmt_bps(up_dip_bps)}) "
    if down_dip:
        dip_str += f"DN({fmt_bps(down_dip_bps)})"
    if not dip_str:
        dip_str = "-"
    dip_style = "cyan bold" if (up_dip or down_dip) else "dim"
    table.add_row("PM dip", Text(dip_str, style=dip_style))
    
    return Panel(table, title="EVENTS", border_style="yellow")


def build_roc_panel(features: dict, edge: dict) -> Panel:
    """Build ROC (Rate of Change) and Countertrend panel."""
    roc = safe_get(features, "roc") or {}
    countertrend = safe_get(features, "countertrend") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("L", width=12)
    table.add_column("V", width=14)
    
    # ROC values
    roc_30s = roc.get("roc_30s")
    roc_60s = roc.get("roc_60s")
    direction = roc.get("direction", "?")
    strength = roc.get("strength", 0)
    ready = roc.get("ready", False)
    
    # Direction with style
    if direction == "UP":
        dir_style = "green bold"
        dir_symbol = "↑"
    elif direction == "DOWN":
        dir_style = "red bold"
        dir_symbol = "↓"
    else:
        dir_style = "yellow"
        dir_symbol = "→"
    
    table.add_row("Direction", Text(f"{dir_symbol} {direction}", style=dir_style))
    table.add_row("ROC 30s", f"{fmt(roc_30s, 2)} bps" if roc_30s else "-")
    table.add_row("ROC 60s", f"{fmt(roc_60s, 2)} bps" if roc_60s else "-")
    
    # Strength bar
    filled = int(strength * 10)
    bar = "█" * filled + "░" * (10 - filled)
    table.add_row("Strength", f"{bar} {int(strength*100)}%")
    
    # Countertrend signal
    ct_signal = countertrend.get("signal", "NONE")
    if ct_signal == "BUY_DOWN":
        ct_style = "red bold"
        ct_text = "⚡ BUY DOWN"
    elif ct_signal == "BUY_UP":
        ct_style = "green bold"
        ct_text = "⚡ BUY UP"
    else:
        ct_style = "dim"
        ct_text = "-"
    
    table.add_row("Countertrend", Text(ct_text, style=ct_style))
    
    # From decision
    if edge:
        ct_used = edge.get("countertrend_used", False)
        if ct_used:
            table.add_row("CT Used", Text("✓ YES", style="green bold"))
    
    return Panel(table, title="ROC + COUNTERTREND", border_style="magenta")


def build_net_edge_panel(edge: dict) -> Panel:
    """Build NET EDGE panel."""
    net_fast = edge.get("net_edge_fast", {}) if edge else {}
    net_smooth = edge.get("net_edge_smooth", {}) if edge else {}
    required = edge.get("required_edge_bps", {}) if edge else {}
    
    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("", width=8)
    table.add_column("UP", justify="right", width=8)
    table.add_column("DOWN", justify="right", width=8)
    
    # FAST
    fast_up = net_fast.get("up")
    fast_down = net_fast.get("down")
    up_style = "green bold" if fast_up and fast_up > 0 else "white"
    down_style = "green bold" if fast_down and fast_down > 0 else "white"
    table.add_row(
        "FAST",
        Text(fmt_bps(fast_up), style=up_style),
        Text(fmt_bps(fast_down), style=down_style),
    )
    
    # SMOOTH
    smooth_up = net_smooth.get("up")
    smooth_down = net_smooth.get("down")
    up_style = "cyan" if smooth_up and smooth_up > 0 else "dim"
    down_style = "cyan" if smooth_down and smooth_down > 0 else "dim"
    table.add_row(
        "SMOOTH",
        Text(fmt_bps(smooth_up), style=up_style),
        Text(fmt_bps(smooth_down), style=down_style),
    )
    
    # REQUIRED
    table.add_row(
        Text("REQ", style="dim"),
        Text(fmt_bps(required.get("up")), style="dim"),
        Text(fmt_bps(required.get("down")), style="dim"),
    )
    
    return Panel(table, title="NET EDGE (bps)", border_style="magenta")


def build_decision_panel(edge: dict) -> Panel:
    """Build DECISION panel."""
    if not edge:
        return Panel(Text("No decision", style="dim"), title="DECISION", border_style="red")
    
    action = edge.get("action", "WAIT")
    candidate = edge.get("candidate_side", "NONE")
    confidence = edge.get("confidence", 0)
    level = edge.get("confidence_level", "LOW")
    reasons = edge.get("confidence_reasons", [])[:3]
    veto = edge.get("veto", False)
    veto_reasons = edge.get("veto_reasons", [])
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("L", width=10)
    table.add_column("V", width=20)
    
    # Action
    action_style = "green bold" if "ACCUMULATE" in action else "yellow bold" if "WATCH" in action else "dim"
    table.add_row("ACTION", Text(action, style=action_style))
    
    # Candidate
    cand_style = "green" if candidate == "UP" else "red" if candidate == "DOWN" else "dim"
    table.add_row("SIDE", Text(candidate, style=cand_style))
    
    # Confidence bar
    conf_pct = int(confidence * 100)
    filled = int(confidence * 10)
    bar = "█" * filled + "░" * (10 - filled)
    level_style = "green" if level == "HIGH" else "yellow" if level == "MED" else "red"
    table.add_row("CONF", Text(f"{bar} {conf_pct}% [{level}]", style=level_style))
    
    # Reasons (top 3)
    if reasons:
        table.add_row("WHY", ", ".join(reasons[:2]))
    
    # Veto
    if veto and veto_reasons:
        table.add_row("VETO", Text(", ".join(veto_reasons[:2]), style="red"))
    
    border_style = "green" if "ACCUMULATE" in action else "yellow" if "WATCH" in action else "red"
    return Panel(table, title="DECISION", border_style=border_style)


def build_market_ref_panel(features: dict) -> Panel:
    """Build MARKET REF panel."""
    mr = safe_get(features, "market_ref") or {}
    
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("L", width=8)
    table.add_column("V", width=14)
    
    table.add_row("ref_px", fmt(mr.get("ref_px"), 2))
    table.add_row("S_raw", fmt(mr.get("S_now_raw"), 2))
    table.add_row("S_smooth", fmt(mr.get("S_now_smooth"), 2))
    
    tau = mr.get("tau_sec")
    table.add_row("tau", f"{fmt(tau, 0)}s" if tau else "-")
    
    return Panel(table, title="MARKET REF", border_style="white")


# ============================================================
# Main Layout
# ============================================================

def build_layout(features: dict, edge: dict, client: DashboardClient) -> Layout:
    """Build complete unified layout."""
    layout = Layout()
    
    # Main structure: header, body, footer
    layout.split_column(
        Layout(name="header", size=4),
        Layout(name="body"),
        Layout(name="footer", size=3),
    )
    
    # Body: 3 columns
    layout["body"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="center", ratio=1),
        Layout(name="right", ratio=1),
    )
    
    # Left column
    layout["left"].split_column(
        Layout(name="polymarket"),
        Layout(name="market_ref"),
    )
    
    # Center column
    layout["center"].split_column(
        Layout(name="fair"),
        Layout(name="vol"),
        Layout(name="bias"),
    )
    
    # Right column
    layout["right"].split_column(
        Layout(name="events"),
        Layout(name="roc"),
        Layout(name="net_edge"),
        Layout(name="decision"),
    )
    
    # Assign panels
    features = features or {}
    edge = edge or {}
    
    layout["header"].update(build_status_panel(features, edge))
    layout["polymarket"].update(build_polymarket_panel(features))
    layout["market_ref"].update(build_market_ref_panel(features))
    layout["fair"].update(build_fair_panel(features))
    layout["vol"].update(build_vol_panel(features))
    layout["bias"].update(build_bias_panel(features))
    layout["events"].update(build_events_panel(features))
    layout["roc"].update(build_roc_panel(features, edge))
    layout["net_edge"].update(build_net_edge_panel(edge))
    layout["decision"].update(build_decision_panel(edge))
    
    # Footer
    footer_text = Text()
    footer_text.append(f"Fetch #{client.fetch_count}", style="dim")
    if client.last_error:
        footer_text.append(f"  ERROR: {client.last_error}", style="red")
    footer_text.append("  [q] quit", style="dim")
    layout["footer"].update(Panel(footer_text, border_style="dim"))
    
    return layout


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="TUI Dashboard v2 (S8)")
    parser.add_argument("--host", default="localhost", help="Collector host")
    parser.add_argument("--port", type=int, default=8000, help="Collector port")
    parser.add_argument("--refresh", type=float, default=1.0, help="Refresh interval (seconds)")
    args = parser.parse_args()
    
    base_url = f"http://{args.host}:{args.port}"
    client = DashboardClient(base_url)
    console = Console()
    
    console.print(f"[bold]TUI Dashboard v2[/] connecting to {base_url}...")
    
    try:
        with Live(console=console, refresh_per_second=2, screen=True) as live:
            while True:
                features, edge = client.fetch()
                layout = build_layout(features, edge, client)
                live.update(layout)
                time.sleep(args.refresh)
    except KeyboardInterrupt:
        console.print("\n[dim]Dashboard closed.[/]")


if __name__ == "__main__":
    main()
