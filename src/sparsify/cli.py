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
# sparsify prototype
# ---------------------------------------------------------------------------




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
    table.add_column("Architecture", justify="right")

    for alias, hf_id in KNOWN_ALIASES.items():
        is_moe = "mixtral" in alias.lower() or "qwen:30b" in alias.lower()
        arch = "[bold magenta]MoE[/bold magenta]" if is_moe else "Dense"
        table.add_row(alias, hf_id, arch)

    console.print(table)
    console.print("\n[dim]To download any of these models, run:[/dim]")
    console.print("  [bold white]sparsify pull <alias>[/bold white]\n")


@main.command("list")
def list_cmd() -> None:
    """List all models downloaded onto this machine."""
    models = all_models()

    table = Table(title="Sparsify Local Models", title_style="bold cyan")
    table.add_column("HF Model ID", style="bold white", no_wrap=True)
    table.add_column("Size", justify="right", style="green")
    table.add_column("Pulled", style="dim")
    table.add_column("Status", style="bold")

    if not models:
        console.print("[dim]No models pulled yet. Run:[/dim]  sparsify pull mixtral:8x7b")
        return

    for m in models:
        status = "[green]Ready[/green]" if m["available"] else "[red]Missing[/red]"
        table.add_row(
            m["hf_id"],
            f"{m['size_gb']:.1f} GB",
            m.get("pulled_at", "—")[:10],
            status,
        )

    console.print(table)
    console.print()
    console.print("[dim]Tip: sparsify run <model-id>   sparsify serve <model-id>[/dim]")


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
    """Start an interactive chat session with a local model."""
    from sparsify.runtime.chat_generation import SparsifyEngine
    import json

    hf_id = resolve_hf_id(model)
    safe_name = hf_id.replace("/", "--")
    model_path = MODELS_DIR / safe_name

    if not model_path.exists():
        console.print(f"[red]Model not found locally. Run:[/red]  sparsify pull {model}")
        raise SystemExit(1)

    memory_limit = _resolve_memory_limit(model_path, memory_limit)

    console.print(f"\n[bold cyan]Sparsify Runtime[/bold cyan]\n")
    console.print(f"  Model    : [bold white]{hf_id}[/bold white]")
    console.print(f"  Path     : [dim]{model_path}[/dim]")
    console.print(f"  Backend  : Apple MLX  (unified memory, Neural Engine)")
    cap = f"{memory_limit} GB" if memory_limit is not None else "auto (sized from free RAM)"
    console.print(f"  RAM Cap  : [bold yellow]{cap}[/bold yellow] (LRU Expert Cache)\n")

    console.print("[dim]Loading weights…[/dim]")

    try:
        engine = SparsifyEngine(model_path, max_tokens=max_tokens, memory_limit_gb=memory_limit)
    except Exception as e:
        console.print(f"[red]Failed to load model: {e}[/red]")
        raise SystemExit(1)

    ready = (f"[bold green]Ready.[/bold green]  "
             f"Backbone: [bold white]{engine.model_memory_gb:.2f} GB[/bold white] resident")
    if engine.paging is not None:
        p = engine.paging.stats()
        ready += (f", experts: [bold white]{p['paged_gb']:.1f} GB[/bold white] on SSD, "
                  f"cache budget: [bold white]{engine.memory_limit_gb:.1f} GB[/bold white]")
    console.print(ready + "\n")
    console.print("[dim]Type your message. Press Enter to submit. (Esc+Enter for newline). /exit or Ctrl-C to quit.[/dim]\n")

    from sparsify.runtime.tui import ClaudeCodeUI
    ui = ClaudeCodeUI()

    while True:
        prompt = ui.ask()
        if prompt is None or prompt.lower() in {"/exit", "/quit", "exit", "quit"}:
            break
        if not prompt:
            continue
        
        # Stream markdown response
        ui.stream_response(engine.generate_stream(prompt))

    console.print("\n[bold]Session ended.[/bold]")


@main.command("serve")
@click.argument("model")
@click.option("--port", "-p", default=11434, show_default=True, help="Port to listen on.")
@click.option("--max-tokens", default=512, show_default=True)
@click.option("--memory-limit", type=int, default=None, help="Explicit RAM limit in GB (saves as default for this model).")
def serve_cmd(model: str, port: int, max_tokens: int, memory_limit: int | None) -> None:
    """Serve a model via an OpenAI-compatible REST API."""
    import json as _json
    import time as _time
    import uuid
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from sparsify.runtime.chat_generation import SparsifyEngine

    hf_id = resolve_hf_id(model)
    safe_name = hf_id.replace("/", "--")
    model_path = MODELS_DIR / safe_name

    if not model_path.exists():
        console.print(f"[red]Model not found locally. Run:[/red]  sparsify pull {model}")
        raise SystemExit(1)
        
    memory_limit = _resolve_memory_limit(model_path, memory_limit)

    cap = f"{memory_limit}GB RAM limit" if memory_limit is not None else "auto RAM limit"
    console.print(f"[dim]Loading {hf_id} with {cap}…[/dim]")
    engine = SparsifyEngine(model_path, max_tokens=max_tokens, memory_limit_gb=memory_limit)
    console.print(
        f"[bold green]Serving {hf_id}[/bold green]  "
        f"at [bold white]http://localhost:{port}/v1/chat/completions[/bold green]\n"
        "Press Ctrl-C to stop."
    )

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_response(404); self.end_headers(); return
            length = int(self.headers.get("Content-Length", 0))
            body = _json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            prompt = messages[-1]["content"] if messages else ""

            # Run real inference
            import io, sys as _sys
            buf = io.StringIO()
            old_stdout = _sys.stdout
            _sys.stdout = buf
            try:
                engine.generate(prompt)
            finally:
                _sys.stdout = old_stdout
            content = buf.getvalue().strip()

            resp = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(_time.time()),
                "model": hf_id,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }],
            }
            payload = _json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_): pass

    HTTPServer(("localhost", port), Handler).serve_forever()


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














