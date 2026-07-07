"""Sparsify CLI — main entry-point."""
from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="sparsify")
def main() -> None:
    """Sparsify — Inference memory profiler and optimization research framework."""


# ---------------------------------------------------------------------------
# sparsify info
# ---------------------------------------------------------------------------


@main.command()
@click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON.")
def info(as_json: bool) -> None:
    """Display system information."""
    from sparsify.profiler.system_profiler import format_system_info, get_system_info

    sys_info = get_system_info()

    if as_json:
        console.print_json(json.dumps(sys_info.to_dict(), default=str))
    else:
        console.print(format_system_info(sys_info))


# ---------------------------------------------------------------------------
# sparsify profile-model
# ---------------------------------------------------------------------------


@main.command("profile-model")
@click.argument("model_path", type=click.Path(exists=True))
@click.option("--json", "-j", "as_json", is_flag=True, help="Output as JSON.")
@click.option("--save", "-s", is_flag=True, help="Save run to the local database.")
@click.option("--export", "-e", is_flag=True, help="Export profile as JSON file.")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output.")
@click.pass_context
def profile_model_cmd(
    ctx: click.Context,
    model_path: str,
    as_json: bool,
    save: bool,
    export: bool,
    verbose: bool,
) -> None:
    """Profile a GGUF model file for memory usage."""
    from sparsify.profiler.model_profiler import format_profile_table, profile_model
    from sparsify.storage.database import SparsifyDB
    from sparsify.utils.config import ensure_dirs
    from sparsify.visualization.export import export_profile_json

    path = Path(model_path)

    # Validate extension
    if path.suffix.lower() != ".gguf":
        console.print("[red]Error:[/red] Only .gguf model files are supported.")
        ctx.exit(1)
        return

    try:
        profile = profile_model(path)
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {model_path}")
        ctx.exit(1)
        return
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        ctx.exit(1)
        return

    profile_dict = profile.to_dict()

    # Display
    if as_json:
        console.print_json(json.dumps(profile_dict, indent=2, default=str))
    else:
        console.print(format_profile_table(profile))

    # Persist
    if save:
        ensure_dirs()
        with SparsifyDB() as db:
            run_id = db.save_profile_run(profile_dict)
        console.print(f"\n[green]Saved[/green] run [bold]{run_id}[/bold]")

    # Export
    if export:
        ensure_dirs()
        export_path = export_profile_json(profile_dict)
        console.print(f"[green]Exported[/green] to [bold]{export_path}[/bold]")


# ---------------------------------------------------------------------------
# sparsify history
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--limit",
    "-n",
    default=20,
    show_default=True,
    help="Max number of runs to display.",
)
def history(limit: int) -> None:
    """Show recent profile runs."""
    from sparsify.storage.database import SparsifyDB
    from sparsify.utils.config import DB_PATH, ensure_dirs

    ensure_dirs()

    if not DB_PATH.exists():
        console.print("[dim]No profile history yet. Run [bold]sparsify profile-model[/bold] first.[/dim]")
        return

    with SparsifyDB() as db:
        runs = db.list_profile_runs(limit=limit)

    if not runs:
        console.print("[dim]No profile runs recorded yet.[/dim]")
        return

    table = Table(title="Profile History", border_style="dim", title_style="bold cyan")
    table.add_column("Run ID", style="bold", max_width=12)
    table.add_column("Model")
    table.add_column("Architecture")
    table.add_column("Timestamp")

    for run in runs:
        table.add_row(
            run["run_id"][:12],
            run.get("model_path", ""),
            run.get("architecture", "—"),
            run.get("timestamp", ""),
        )

    console.print(table)


# ---------------------------------------------------------------------------
# sparsify experiment
# ---------------------------------------------------------------------------


@main.command()
@click.argument("model_path", type=click.STRING)
@click.option(
    "--max-heads",
    "-h",
    "max_heads",
    default=20,
    show_default=True,
    help="Max number of attention heads to test for speed.",
)
def experiment(model_path: str, max_heads: int) -> None:
    """Run all 8 research validation experiments on an MLX model."""
    from sparsify.experiments.runner import execute_all_experiments

    try:
        execute_all_experiments(model_path, max_heads_to_test=max_heads)
    except Exception as exc:
        console.print(f"[red]Error during experiments:[/red] {exc}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# sparsify domain-locality
# ---------------------------------------------------------------------------


@main.command("domain-locality")
@click.argument("model_path", type=click.STRING)
def domain_locality(model_path: str) -> None:
    """Run Research Experiment SP-009: Domain Locality Analysis on an MLX model."""
    from sparsify.experiments.domain_locality import run_domain_locality_experiment

    try:
        run_domain_locality_experiment(model_path)
    except Exception as exc:
        console.print(f"[red]Error during domain locality analysis:[/red] {exc}")
        raise click.Abort()


# ---------------------------------------------------------------------------
# Model registry commands: pull / models / list / run / serve
# ---------------------------------------------------------------------------

from sparsify.runtime.model_registry import (  # noqa: E402
    KNOWN_ALIASES, MODELS_DIR, resolve_hf_id, register, all_models, get as reg_get,
)


def _local_model_complete(local_path: Path) -> bool:
    """True when a usable model is already fully on disk: config present and
    every weight file referenced by the safetensors index exists non-empty."""
    if not (local_path / "config.json").exists():
        return False
    index_file = local_path / "model.safetensors.index.json"
    if index_file.exists():
        try:
            with open(index_file) as f:
                shards = set(json.load(f)["weight_map"].values())
        except (json.JSONDecodeError, KeyError):
            return False
        return all(
            (local_path / s).exists() and (local_path / s).stat().st_size > 0
            for s in shards
        )
    single = local_path / "model.safetensors"
    return single.exists() and single.stat().st_size > 0


@main.command("pull")
@click.argument("model")
@click.option("--force", is_flag=True, help="Re-download even if the model is already on disk.")
def pull_cmd(model: str, force: bool) -> None:
    """Download a model from HuggingFace and register it locally.

    MODEL can be a Sparsify alias (e.g. mixtral:8x7b) or any HuggingFace
    repo id (e.g. mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit).
    """
    import os
    import time
    import concurrent.futures
    from huggingface_hub import snapshot_download, hf_hub_url, HfApi
    import huggingface_hub.utils

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    huggingface_hub.utils.disable_progress_bars()

    hf_id = resolve_hf_id(model)
    safe_name = hf_id.replace("/", "--")
    local_path = MODELS_DIR / safe_name
    local_path.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold cyan]Sparsify pull[/bold cyan]  {hf_id}\n")

    if not force and _local_model_complete(local_path):
        size_bytes = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
        register(hf_id, local_path, size_bytes)
        console.print(f"[bold green]✓ Already downloaded[/bold green]  "
                      f"({size_bytes / 1e9:.2f} GB on disk)")
        console.print(f"  Path : [dim]{local_path}[/dim]")
        console.print(f"  [dim]Use --force to re-download.[/dim]\n")
        return

    # Show remote repo size before downloading
    remote_bytes = 0
    try:
        api = HfApi()
        info = api.repo_info(repo_id=hf_id, files_metadata=True)
        siblings = info.siblings or []
        remote_bytes = sum(
            getattr(s, "size", 0) or 0
            for s in siblings
        )
        if remote_bytes > 0:
            console.print(f"  Remote size  : [bold white]{remote_bytes / 1e9:.1f} GB[/bold white]")
    except Exception:
        pass  # network unavailable or private repo — proceed anyway

    console.print(f"  Destination  : [dim]{local_path}[/dim]")
    console.print(f"  Format       : MLX 4-bit quantised (Apple Silicon)\n")
    console.print("[yellow]Downloading via hf_transfer (maximized bandwidth)…[/yellow]")

    from rich.progress import Progress, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

    def get_download_size(local_path_obj, hf_repo_id):
        total = 0
        
        # 1. Size of all files currently in the destination directory
        if local_path_obj.exists():
            for dirpath, _, filenames in os.walk(local_path_obj):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        try: total += os.path.getsize(fp)
                        except OSError: pass
                        
        # 2. Size of .incomplete in-progress files in the central HF cache
        # hf_transfer downloads here first before copying to local_dir (especially across partitions)
        cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{hf_repo_id.replace('/', '--')}")
        if os.path.exists(cache_dir):
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    if f.endswith(".incomplete"):
                        fp = os.path.join(dirpath, f)
                        try: total += os.path.getsize(fp)
                        except OSError: pass
                        
        return total

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                snapshot_download,
                repo_id=hf_id, 
                local_dir=str(local_path)
            )
            
            if remote_bytes > 0:
                with Progress(
                    TextColumn("[bold blue]{task.description}", justify="right"),
                    BarColumn(bar_width=None),
                    "[progress.percentage]{task.percentage:>3.1f}%",
                    "•",
                    DownloadColumn(),
                    "•",
                    TransferSpeedColumn(),
                    "•",
                    TimeRemainingColumn(),
                    console=console
                ) as progress:
                    task = progress.add_task("Downloading...", total=remote_bytes)
                    
                    highest_size = 0
                    while not future.done():
                        current_size = get_download_size(local_path, hf_id)
                        highest_size = max(highest_size, current_size)
                        progress.update(task, completed=min(highest_size, remote_bytes))
                        time.sleep(0.1)
                    
                    progress.update(task, completed=remote_bytes)
            else:
                # Fallback if we couldn't get size (e.g. no network but cached, or private repo)
                with console.status("Downloading...", spinner="dots"):
                    future.result()

            future.result()  # raise if error
    except Exception as e:
        console.print(f"[red]Download failed: {e}[/red]")
        raise SystemExit(1)

    # Measure actual bytes on disk
    size_bytes = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
    register(hf_id, local_path, size_bytes)

    console.print(f"\n[bold green]✓ Pulled {hf_id}[/bold green]")
    console.print(f"  Disk usage   : [bold white]{size_bytes / 1e9:.2f} GB[/bold white]")
    console.print(f"  Path         : [dim]{local_path}[/dim]\n")


@main.command("models")
def models_cmd() -> None:
    """Discover available supported models that can be pulled."""
    table = Table(title="Sparsify Supported Models", title_style="bold cyan")
    table.add_column("Alias", style="bold green", no_wrap=True)
    table.add_column("HuggingFace Repo ID", style="dim")
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Arch", justify="right", no_wrap=True)
    table.add_column("Status", no_wrap=True)

    seen = set()
    for alias, entry in KNOWN_ALIASES.items():
        if entry["hf"] in seen:
            continue
        seen.add(entry["hf"])
        arch = "[bold magenta]MoE[/bold magenta]" if entry["moe"] else "Dense"
        status = "[green]verified[/green]" if entry.get("tested") else "[dim]community[/dim]"
        table.add_row(alias, entry["hf"], f"{entry.get('gb', 0):.0f} GB", arch, status)

    console.print(table)
    console.print("\n[dim]To download any of these models, run:[/dim]")
    console.print("  [bold white]sparsify pull <alias>[/bold white]\n")


@main.command("list")
def list_cmd() -> None:
    """List all models downloaded onto this machine."""
    from sparsify.runtime.model_registry import alias_for

    models = all_models()

    table = Table(title="Sparsify Local Models", title_style="bold cyan")
    table.add_column("Model", style="bold white")
    table.add_column("Size", justify="right", style="green", no_wrap=True)
    table.add_column("Status", style="bold", no_wrap=True)
    table.add_column("Name", style="cyan", no_wrap=True)

    if not models:
        console.print("[dim]No models pulled yet. Run:[/dim]  sparsify pull olmoe:1b-7b")
        return

    for m in models:
        status = "[green]Ready[/green]" if m["available"] else "[red]Missing[/red]"
        tag = alias_for(m["hf_id"]) or m["hf_id"].split("/")[-1]
        table.add_row(
            m["hf_id"],
            f"{m['size_gb']:.1f} GB",
            status,
            tag,
        )

    console.print(table)
    console.print()
    console.print("[dim]Chat with any of them:[/dim]  sparsify run <name>   "
                  "[dim]— any unique part of the id works too, e.g.[/dim] "
                  "sparsify run qwen3")


def _resolve_memory_limit(model_path: Path, memory_limit: int | None) -> int | None:
    """Persist an explicitly requested limit; otherwise use the saved one.
    Returns None when nothing was ever configured — the engine then sizes
    the expert cache automatically from measured free RAM."""
    settings_file = model_path / "sparsify_settings.json"
    if memory_limit is not None:
        with open(settings_file, "w") as f:
            json.dump({"memory_limit_gb": memory_limit}, f)
        console.print(f"[dim]Saved memory limit: {memory_limit} GB[/dim]")
        return memory_limit
    if settings_file.exists():
        try:
            with open(settings_file) as f:
                return json.load(f).get("memory_limit_gb")
        except (json.JSONDecodeError, OSError):
            return None
    return None


@main.command("run")
@click.argument("model")
@click.option("--max-tokens", default=512, help="Maximum tokens to generate.")
@click.option("--memory-limit", type=int, default=None, help="Explicit RAM limit in GB (saves as default for this model).")
def run_cmd(model: str, max_tokens: int, memory_limit: int | None) -> None:
    """Start an interactive chat session with a local model.

    MODEL accepts an alias (qwen:30b-a3b), a full HF id, or any unique
    part of a local model's name (e.g. "qwen3").
    """
    from sparsify.runtime import backend
    from sparsify.runtime.model_registry import resolve_local, alias_for

    try:
        be = backend.require()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    resolved = resolve_local(model)
    if resolved is None:
        console.print(f"[red]No local model matches '{model}'.[/red]")
        local = [m for m in all_models() if m["available"]]
        if local:
            console.print("[dim]On this machine:[/dim]")
            for m in local:
                tag = alias_for(m["hf_id"]) or m["hf_id"]
                console.print(f"  sparsify run {tag}")
        console.print(f"[dim]Or download it:[/dim]  sparsify pull {model}")
        raise SystemExit(1)
    hf_id, model_path = resolved

    memory_limit = _resolve_memory_limit(model_path, memory_limit)

    from sparsify.runtime.chat_generation import SparsifyEngine
    from sparsify.runtime.tui import ChatUI

    ui = ChatUI(console)
    ui.banner(hf_id, model_path, be.device, memory_limit)
    # The engine loads and runs on the UI's dedicated worker thread (MLX
    # streams are thread-bound); the prompt stays live during generation.
    ui.run(lambda: SparsifyEngine(model_path, max_tokens=max_tokens,
                                  memory_limit_gb=memory_limit))


@main.command("serve")
@click.argument("model", required=False)
@click.option("--port", "-p", default=7777, show_default=True, help="Port to listen on.")
@click.option("--max-tokens", default=1024, show_default=True)
@click.option("--memory-limit", type=float, default=None,
              help="Expert-cache budget in GB (default: auto from free RAM).")
def serve_cmd(model: str | None, port: int, max_tokens: int, memory_limit: float | None) -> None:
    """Run the Sparsify API server (OpenAI-compatible).

    MODEL is optional: without it the server starts empty and loads
    whichever model each request names — like Ollama. With it, that model
    is loaded eagerly and used as the default.
    """
    from sparsify.runtime.server import serve

    def log(msg: str) -> None:
        console.print(f"[dim]{msg}[/dim]")

    console.print(f"\n[bold cyan]Sparsify API[/bold cyan]  http://localhost:{port}")
    console.print("  [dim]POST /v1/chat/completions · GET /v1/models · GET /health[/dim]")
    console.print("  [dim]Ctrl-C to stop.[/dim]\n")
    try:
        serve(port=port, model=model, memory_limit_gb=memory_limit,
              max_tokens=max_tokens, log=log)
    except OSError as exc:
        import json as _json
        import urllib.request
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
                health = _json.load(r)
        except OSError:
            health = None
        if health and health.get("runtime") == "sparsify":
            loaded = health.get("loaded") or "no model loaded"
            console.print(f"[yellow]A Sparsify server is already running on "
                          f"port {port}[/yellow] ({loaded}).")
            console.print("[dim]Use it directly, stop the login service with "
                          "'sparsify stop', or pick another port: "
                          f"sparsify serve --port {port + 1}[/dim]")
        else:
            console.print(f"[red]Could not bind port {port}: {exc}[/red]")
            console.print(f"[dim]Something else owns this port — try: "
                          f"sparsify serve --port {port + 1}[/dim]")
        raise SystemExit(1)


_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.daylink.sparsify.plist"


def _brand_runtime(script: Path) -> None:
    """Make the process show up as 'sparsify-runtime' instead of 'python3.12'.

    Activity Monitor displays the executable's name, so we copy the venv's
    interpreter stub under our own name and point the console script's
    shebang at it. Safe to re-run (pip reinstalls reset the shebang).
    """
    import shutil as _shutil

    bin_dir = script.parent
    python = bin_dir / "python3"
    runtime = bin_dir / "sparsify-runtime"
    if not (script.exists() and python.exists()):
        return
    try:
        if not runtime.exists():
            _shutil.copy(python.resolve(), runtime)
            runtime.chmod(0o755)
        lines = script.read_text().splitlines(keepends=True)
        if lines and lines[0].startswith("#!") and "sparsify-runtime" not in lines[0]:
            lines[0] = f"#!{runtime}\n"
            script.write_text("".join(lines))
    except OSError:
        pass  # cosmetic feature — never block the service over it


@main.command("start")
@click.option("--port", "-p", default=7777, show_default=True)
def start_cmd(port: int) -> None:
    """Install and start the background API service (launchd, runs at login)."""
    import shutil
    import subprocess
    import sys
    import time
    import urllib.request

    # launchd agents cannot read external volumes (macOS privacy protection),
    # so the service must run from the internal install when one exists.
    internal = Path.home() / ".sparsify" / "venv" / "bin" / "sparsify"
    sparsify_bin = str(internal) if internal.exists() else (
        shutil.which("sparsify") or sys.argv[0]
    )
    _brand_runtime(Path(sparsify_bin))
    if sparsify_bin.startswith("/Volumes/"):
        console.print(
            "[yellow]Warning:[/yellow] the service binary lives on an external "
            "volume, which launchd usually cannot read. Run ./install.sh to "
            "create an internal install first."
        )
    log_dir = Path.home() / ".sparsify" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "server.log"
    plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.daylink.sparsify</string>
  <key>ProgramArguments</key><array>
    <string>{sparsify_bin}</string><string>serve</string>
    <string>--port</string><string>{port}</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>5</integer>
  <key>StandardOutPath</key><string>{log_file}</string>
  <key>StandardErrorPath</key><string>{log_file}</string>
</dict></plist>
"""
    _PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _PLIST_PATH.write_text(plist)
    subprocess.run(["launchctl", "unload", str(_PLIST_PATH)],
                   capture_output=True)
    subprocess.run(["launchctl", "load", str(_PLIST_PATH)], check=True)

    # Verify it actually came up — never report success on a crash loop.
    deadline = time.time() + 15
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://localhost:{port}/health", timeout=2
            ) as resp:
                if resp.status == 200:
                    console.print(
                        f"[bold green]Sparsify service running[/bold green] on "
                        f"http://localhost:{port} (starts automatically at login)"
                    )
                    console.print(f"  [dim]logs: {log_file} · stop: sparsify stop[/dim]")
                    return
        except OSError:
            time.sleep(1)
    console.print("[red]Service did not come up within 15s.[/red] Last log lines:")
    if log_file.exists():
        for line in log_file.read_text().splitlines()[-8:]:
            console.print(f"  [dim]{line}[/dim]")
    raise SystemExit(1)


@main.command("stop")
def stop_cmd() -> None:
    """Stop the background API service and remove it from login items."""
    import subprocess

    if _PLIST_PATH.exists():
        subprocess.run(["launchctl", "unload", str(_PLIST_PATH)],
                       capture_output=True)
        _PLIST_PATH.unlink()
        console.print("[bold green]Sparsify service stopped.[/bold green]")
    else:
        console.print("[dim]No Sparsify service installed.[/dim]")


@main.command("stats")
def stats_cmd() -> None:
    """Show live hardware memory statistics."""
    import mlx.core as mx

    active_gb = mx.get_active_memory() / 1e9
    peak_gb = mx.get_peak_memory() / 1e9
    cache_gb = mx.get_cache_memory() / 1e9

    table = Table(title="Sparsify Hardware Stats", title_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Active unified memory", f"{active_gb:.3f} GB")
    table.add_row("Peak unified memory (session)", f"{peak_gb:.3f} GB")
    table.add_row("MLX cache memory", f"{cache_gb:.3f} GB")
    console.print(table)

    models = all_models()
    if models:
        console.print()
        mt = Table(title="Pulled Models", title_style="bold")
        mt.add_column("HF Model ID", style="bold white")
        mt.add_column("Disk", justify="right", style="green")
        mt.add_column("Status", style="bold")
        for m in models:
            status = "[green]Ready[/green]" if m["available"] else "[red]Missing[/red]"
            mt.add_row(m["hf_id"], f"{m['size_gb']:.1f} GB", status)
        console.print(mt)


@main.command("inspect")
@click.argument("model")
def inspect_cmd(model: str) -> None:
    """Print architecture details for a pulled model."""
    from transformers import AutoConfig

    hf_id = resolve_hf_id(model)
    safe_name = hf_id.replace("/", "--")
    model_path = MODELS_DIR / safe_name

    if not model_path.exists():
        console.print(f"[red]Model not found locally. Run:[/red]  sparsify pull {model}")
        raise SystemExit(1)

    try:
        cfg = AutoConfig.from_pretrained(str(model_path))
    except Exception as e:
        console.print(f"[red]Could not read config: {e}[/red]")
        raise SystemExit(1)

    size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())

    table = Table(title=f"Model: {hf_id}", title_style="bold cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")
    table.add_row("Architecture", type(cfg).__name__.replace("Config", ""))
    table.add_row("Vocab size", str(getattr(cfg, "vocab_size", "—")))
    table.add_row("Hidden dim", str(getattr(cfg, "hidden_size", "—")))
    table.add_row("Layers", str(getattr(cfg, "num_hidden_layers", "—")))
    table.add_row("Attention heads", str(getattr(cfg, "num_attention_heads", "—")))

    # MoE fields (present on Mixtral, Qwen-MoE etc.)
    n_experts = getattr(cfg, "num_local_experts", None) or getattr(cfg, "num_experts", None)
    top_k = getattr(cfg, "num_experts_per_tok", None) or getattr(cfg, "top_k", None)
    if n_experts:
        table.add_row("Experts per layer", str(n_experts))
    if top_k:
        table.add_row("Active experts / token", str(top_k))

    table.add_row("Disk size", f"{size_bytes / 1e9:.2f} GB")
    table.add_row("Local path", str(model_path))
    console.print(table)
















if __name__ == "__main__":
    main()
