"""Live terminal dashboard for the Sparsify Virtual Memory AI Runtime."""
from __future__ import annotations

import time
from typing import Any, Dict
from rich.align import Align
from rich.console import Console, RenderableType
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

from sparsify.runtime.telemetry import TelemetryRecorder

console = Console()


class LiveDashboard:
    """Terminal UI rendering real-time memory scheduling, prefetching, and throughput statistics."""

    def __init__(self, telemetry: TelemetryRecorder) -> None:
        self.telemetry = telemetry
        self.live: Live | None = None
        self.layout = self.make_layout()
        self.output_text = ""

    def make_layout(self) -> Layout:
        """Create the dashboard panels layout."""
        layout = Layout()
        
        # Split into header, body, and output text panels
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body", ratio=1),
            Layout(name="output", size=6)
        )
        
        # Split body into sidebar (telemetry stats) and main (cache mapping)
        layout["body"].split_row(
            Layout(name="sidebar", ratio=2),
            Layout(name="cache_map", ratio=3)
        )
        
        return layout

    def start(self) -> None:
        """Initialize the rich live loop."""
        self.live = Live(self.layout, console=console, refresh_per_second=10, screen=True)
        self.live.start()

    def stop(self) -> None:
        """Stop the rich live loop."""
        if self.live:
            self.live.stop()

    def update_token_text(self, text: str) -> None:
        """Append generated tokens to the text output panel."""
        self.output_text += text
        self.update()

    def update(self) -> None:
        """Re-render all dashboard panels with current telemetry metrics."""
        if not self.live:
            return

        metrics = self.telemetry.get_metrics()

        # 1. Update Header Panel
        self.layout["header"].update(
            Panel(
                Align.center(
                    Text.from_markup(
                        "[bold yellow]⚡ Sparsify Runtime — Virtual Memory for Artificial Intelligence ⚡[/bold yellow] "
                        "[dim]| Active Memory Operating System[/dim]"
                    )
                ),
                border_style="yellow"
            )
        )

        # 2. Update Sidebar (Core Statistics)
        stats_table = Table.grid(padding=(0, 1))
        stats_table.add_column("Stat", style="bold cyan")
        stats_table.add_column("Value", style="bold green", justify="right")

        stats_table.add_row("Total Model Size", f"{metrics['total_model_size_mb']:.1f} MB")
        stats_table.add_row("Active Memory Footprint", f"{metrics['active_memory_footprint_mb']:.1f} MB")
        
        ratio_pct = metrics['sparsify_ratio'] * 100.0
        stats_table.add_row("Sparsify Ratio (Active/Total)", f"{ratio_pct:.3f}%")
        stats_table.add_row("SSD Read Bandwidth", f"{metrics['ssd_bandwidth_mbs']:.2f} MB/s")
        stats_table.add_row("Cache Hit Ratio", f"{metrics['cache_hit_ratio']*100.0:.1f}%")
        stats_table.add_row("Eviction Frequency", f"{metrics['cache_evictions']} evictions")
        stats_table.add_row("Avg Expert Load Latency", f"{metrics['expert_load_latency_ms']:.2f} ms")
        stats_table.add_row("Active Experts Loaded", f"{metrics['active_experts_count']}")
        stats_table.add_row("Tokens Generated", f"{metrics['total_tokens']}")
        stats_table.add_row("First Token Latency", f"{metrics['first_token_latency_ms']:.2f} ms")
        stats_table.add_row("Inference Speed", f"{metrics['tokens_per_sec']:.2f} tokens/s")
        stats_table.add_row("Prefetch Hit Ratio", f"{metrics['prefetch_hit_ratio']*100.0:.1f}%")
        
        # Add visual progress bar for RAM budget utilization
        progress = Progress(
            TextColumn("[bold cyan]Memory Budget Usage[/bold cyan]"),
            BarColumn(bar_width=15, complete_style="yellow", finished_style="green"),
            TextColumn("[bold green]{task.percentage:>3.0f}%[/bold green]")
        )
        progress.add_task("Usage", total=100, completed=int(min(100.0, ratio_pct * 10.0))) # Scale visual representation

        self.layout["sidebar"].update(
            Panel(
                Align.center(stats_table),
                title="[bold yellow]System Telemetry[/bold yellow]",
                border_style="cyan"
            )
        )

        # 3. Update Cache Mapping Visualization
        # Display an ASCII grid representation of the 8 layers and 16 experts
        # marking which ones are currently resident in VRAM!
        cache_grid = Table(title="VRAM Resident Expert Map (8 Layers * 16 Experts)", border_style="dim", title_style="bold yellow", show_header=True)
        cache_grid.add_column("L / E", style="bold cyan")
        for e in range(16):
            cache_grid.add_column(f"{e:X}", justify="center")

        # Mocking active checks (checks self.telemetry cache registry keys)
        # Note: self._loaded_experts tells us which keys are active.
        # We fetch active keys from telemetry variables
        active_keys = getattr(self.telemetry, "active_keys", set())

        for l in range(8):
            row_vals = [f"L{l}"]
            for e in range(16):
                if (l, e) in active_keys:
                    row_vals.append("[bold green]■[/bold green]")
                else:
                    row_vals.append("[dim grey]·[/dim grey]")
            cache_grid.add_row(*row_vals)

        self.layout["cache_map"].update(
            Panel(
                Align.center(cache_grid),
                title="[bold yellow]Virtual Memory Paging Map[/bold yellow]",
                border_style="yellow"
            )
        )

        # 4. Update Output Text Panel
        self.layout["output"].update(
            Panel(
                Text(self.output_text or "Waiting for generation...", style="italic green"),
                title="[bold yellow]Text Completion Stream[/bold yellow]",
                border_style="green"
            )
        )
