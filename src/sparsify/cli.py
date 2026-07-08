"""Sparsify CLI — main entry-point."""
from __future__ import annotations

import json
from pathlib import Path

import click
from sparsify import __version__
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="sparsify")
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


def _require_models_dir() -> bool:
    """Fail with a human explanation when the models directory is on an
    unplugged volume, instead of a mkdir traceback."""
    from sparsify.runtime.model_registry import models_dir_status

    status = models_dir_status()
    if status == "unmounted":
        volume = Path(*MODELS_DIR.parts[:3])
        console.print(f"[red]Your models directory is on a disk that isn't "
                      f"connected:[/red] {MODELS_DIR}")
        console.print(f"  [dim]Plug in {volume}, or point Sparsify elsewhere:[/dim]")
        console.print(f"  [dim]· one-off:[/dim]   SPARSIFY_MODELS_DIR=~/models sparsify …")
        console.print(f"  [dim]· permanent:[/dim] edit ~/.sparsify/config.json "
                      f"{{\"models_dir\": \"…\"}}")
        return False
    return True


def _pick_model_interactively() -> str | None:
    """Catalog picker for `sparsify pull` with no argument."""
    import sys

    downloaded = {m["hf_id"] for m in all_models() if m["available"]}
    entries = []
    seen_hf = set()
    for alias, e in KNOWN_ALIASES.items():
        if e["hf"] in seen_hf:
            continue
        seen_hf.add(e["hf"])
        entries.append((alias, e))
    entries.sort(key=lambda ae: (not ae[1]["moe"], ae[1]["gb"]))

    table = Table(title="Pull a model", title_style="bold cyan")
    table.add_column("#", justify="right", style="bold")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Size", justify="right", style="green")
    table.add_column("Arch")
    table.add_column("Status")
    for i, (alias, e) in enumerate(entries, 1):
        arch = "[bold magenta]MoE[/bold magenta]" if e["moe"] else "Dense"
        if e["hf"] in downloaded:
            status = "[green]downloaded[/green]"
        elif e["tested"]:
            status = "[green]verified[/green]"
        else:
            status = "[dim]community[/dim]"
        table.add_row(str(i), alias, f"{e['gb']:.1f} GB", arch, status)
    console.print(table)
    console.print("[dim]Any Hugging Face MLX repo id also works: "
                  "sparsify pull mlx-community/…[/dim]")

    if not sys.stdin.isatty():
        console.print("[red]No model given and no terminal to ask — run: "
                      "sparsify pull <name>[/red]")
        return ("invalid", None)
    choice = click.prompt("Number to pull (or blank to cancel)",
                          default="", show_default=False).strip()
    if not choice:
        return ("cancel", None)
    try:
        idx = int(choice)
        if not 1 <= idx <= len(entries):
            raise ValueError
    except ValueError:
        console.print(f"[red]'{choice}' is not a row number above.[/red]")
        return ("invalid", None)
    return ("ok", entries[idx - 1][0])


@main.command("pull")
@click.argument("model", required=False)
@click.option("--force", is_flag=True, help="Re-download even if the model is already on disk.")
def pull_cmd(model: str | None, force: bool) -> None:
    """Download a model from HuggingFace and register it locally.

    MODEL can be a Sparsify alias (e.g. mixtral:8x7b) or any HuggingFace
    repo id (e.g. mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit). With no
    MODEL, an interactive picker shows the catalog.
    """
    import os
    import time
    import concurrent.futures
    from huggingface_hub import snapshot_download, hf_hub_url, HfApi
    import huggingface_hub.utils
    from sparsify.runtime.model_registry import suggest_alias

    if not _require_models_dir():
        raise SystemExit(1)

    if model is None:
        status, model = _pick_model_interactively()
        if status == "cancel":
            raise SystemExit(0)
        if model is None:
            raise SystemExit(1)

    # a colon means the user meant an alias — catch typos before they turn
    # into a bogus Hugging Face repo id ("qwen3:30b" is not a repo)
    if ":" in model and model.lower() not in KNOWN_ALIASES:
        hint = suggest_alias(model)
        console.print(f"[red]Unknown model name '{model}'.[/red]")
        if hint:
            console.print(f"  Did you mean:  [bold cyan]sparsify pull {hint}[/bold cyan]")
        console.print("  [dim]See the catalog: sparsify models · or pull any "
                      "HF repo id: sparsify pull mlx-community/…[/dim]")
        raise SystemExit(1)

    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    huggingface_hub.utils.disable_progress_bars()

    hf_id = resolve_hf_id(model)
    safe_name = hf_id.replace("/", "--")
    local_path = MODELS_DIR / safe_name
    try:
        local_path.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as exc:
        console.print(f"[red]Cannot create {local_path}:[/red] {exc}")
        raise SystemExit(1)

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
                    TextColumn("{task.fields[stats]}"),
                    console=console
                ) as progress:
                    task = progress.add_task(
                        "Downloading", total=remote_bytes,
                        stats="starting…")

                    # Speed measured from byte deltas (exponential moving
                    # average so it reads steady, not jittery), MB units.
                    highest = 0
                    last_bytes, last_t, speed = 0, time.monotonic(), 0.0
                    while not future.done():
                        highest = max(highest, get_download_size(local_path, hf_id))
                        done_b = min(highest, remote_bytes)
                        now = time.monotonic()
                        if now - last_t >= 0.5:
                            inst = (done_b - last_bytes) / (now - last_t)
                            speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
                            last_bytes, last_t = done_b, now
                        eta = (remote_bytes - done_b) / speed if speed > 1 else None
                        eta_s = (f" · {int(eta // 60)}m {int(eta % 60):02d}s left"
                                 if eta is not None and eta < 360000 else "")
                        progress.update(task, completed=done_b, stats=(
                            f"{done_b / 1e6:,.0f} / {remote_bytes / 1e6:,.0f} MB"
                            f" · {speed / 1e6:,.1f} MB/s{eta_s}"))
                        time.sleep(0.25)

                    progress.update(task, completed=remote_bytes,
                                    stats=f"{remote_bytes / 1e6:,.0f} MB · done")
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
    from sparsify.runtime.model_registry import alias_for, models_dir_status

    if models_dir_status() == "unmounted":
        console.print(f"[yellow]Your models directory is on a disk that isn't "
                      f"connected:[/yellow] {MODELS_DIR}")
        console.print("[dim]Plug it back in, or point Sparsify elsewhere with "
                      "SPARSIFY_MODELS_DIR / ~/.sparsify/config.json[/dim]")
        raise SystemExit(1)

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
    total = sum(m["size_gb"] for m in models if m["available"])
    console.print(f"[dim]Stored in[/dim] {MODELS_DIR} [dim]· {total:.1f} GB total[/dim]")
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
@click.option("--max-tokens", default=0, help="Max tokens per reply; 0 = unlimited (until the model finishes or the context window fills).")
@click.option("--memory-limit", type=int, default=None, help="Explicit RAM limit in GB (saves as default for this model).")
@click.option("--read-only", "read_only", is_flag=True, help="Restrict tools to read-only (no file writes, no shell).")
@click.option("--no-shell", is_flag=True, help="Allow file read/write but NOT shell execution.")
@click.option("--workspace", type=click.Path(), default=None, help="Directory the agent reads/writes/runs in (default: ~/.sparsify/workspace).")
def run_cmd(model: str, max_tokens: int, memory_limit: int | None,
            read_only: bool, no_shell: bool, workspace: str | None) -> None:
    """Start an interactive chat session with a local model.

    MODEL accepts an alias (qwen:30b-a3b), a full HF id, or any unique
    part of a local model's name (e.g. "qwen3").

    Tools are ON by default (fetch_url, web_search, read/write files,
    run shell) — scoped to a workspace directory so the blast radius is
    contained. Point it wider with --workspace ~, soften with --no-shell,
    or turn tools off with --read-only. Tool-capable models only (qwen:30b
    yes; small models without a tool template just chat).
    """
    from sparsify.runtime import backend
    from sparsify.runtime.model_registry import resolve_local, alias_for

    try:
        be = backend.require()
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    if not _require_models_dir():
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
    from sparsify.runtime.tools import ToolPolicy

    ws = Path(workspace) if workspace else None
    ui = ChatUI(console)
    ui.banner(hf_id, model_path, be.device, memory_limit)
    try:  # non-blocking, cached; never delays startup on a network hiccup
        from sparsify.runtime import updater
        st = updater.check()
        if st["update_available"]:
            console.print(f"[yellow]● update available[/yellow] "
                          f"([dim]{st['current']} → {st['latest']}[/dim]) — "
                          "run [bold]sparsify update[/bold]")
    except Exception:
        pass
    if read_only:
        ui.policy = ToolPolicy.read_only(ws)
        ui._tools_on = True
        console.print(f"[dim]tools: read-only (fetch, search, read) in "
                      f"{ui.policy.workspace}[/dim]")
    else:
        # tools on by default, scoped to the workspace (contained blast radius)
        ui.policy = ToolPolicy.from_flags(agent=True, workspace=ws,
                                          allow_shell=not no_shell)
        ui._tools_on = True
        shell = "no shell" if no_shell else "shell"
        console.print(f"[dim]tools: read · write · {shell} in "
                      f"{ui.policy.workspace} — /tools off to disable, "
                      f"--workspace to widen[/dim]")
    # The engine loads and runs on the UI's dedicated worker thread (MLX
    # streams are thread-bound); the prompt stays live during generation.
    ui.run(lambda: SparsifyEngine(model_path, max_tokens=max_tokens,
                                  memory_limit_gb=memory_limit))


@main.command("serve")
@click.argument("model", required=False)
@click.option("--port", "-p", default=7777, show_default=True, help="Port to listen on.")
@click.option("--max-tokens", default=0, help="Max tokens per reply; 0 = unlimited.")
@click.option("--memory-limit", type=float, default=None,
              help="Expert-cache budget in GB (default: auto from free RAM).")
@click.option("--shell", "with_shell", is_flag=True, help="Also allow run_shell for requests (off by default on the server: a webpage you visit could reach localhost, and shell = code execution).")
@click.option("--read-only", "read_only", is_flag=True, help="Restrict tools to read-only (no file writes).")
@click.option("--workspace", type=click.Path(), default=None, help="Directory agent tools operate in (default: ~/.sparsify/workspace).")
def serve_cmd(model: str | None, port: int, max_tokens: int, memory_limit: float | None,
              with_shell: bool, read_only: bool, workspace: str | None) -> None:
    """Run the Sparsify API server (OpenAI-compatible).

    MODEL is optional: without it the server starts empty and loads
    whichever model each request names — like Ollama. With it, that model
    is loaded eagerly and used as the default.

    Tools for 'tools:auto' requests are ON by default: fetch/search/read
    and workspace-confined file writes. Shell is OFF by default here
    (unlike interactive `run`) because the server is network-reachable —
    add --shell only if you trust every client of this port. Tool grant
    is a startup decision, never something a request grants itself.
    """
    from sparsify.runtime.server import serve
    from sparsify.runtime.tools import ToolPolicy

    def log(msg: str) -> None:
        console.print(f"[dim]{msg}[/dim]")

    ws = Path(workspace) if workspace else None
    policy = (ToolPolicy.read_only(ws) if read_only
              else ToolPolicy.from_flags(agent=True, workspace=ws,
                                         allow_shell=with_shell))

    console.print(f"\n[bold cyan]Sparsify API[/bold cyan]  http://localhost:{port}")
    console.print("  [dim]POST /v1/chat/completions · GET /v1/models · GET /health[/dim]")
    tiers = "read" if read_only else ("read · write · shell" if with_shell
                                      else "read · write")
    console.print(f"  [dim]tools ({tiers}) in {policy.workspace}[/dim]")
    if with_shell:
        console.print("  [yellow]shell enabled[/yellow] — any client of this "
                      "port can run commands.")
    console.print("  [dim]Ctrl-C to stop.[/dim]\n")
    try:
        serve(port=port, model=model, memory_limit_gb=memory_limit,
              max_tokens=max_tokens, policy=policy, log=log)
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

    import os

    # launchd agents cannot read external volumes (macOS privacy protection),
    # so the service must run from the internal install when one exists.
    home = Path(os.environ.get("SPARSIFY_HOME", str(Path.home() / ".sparsify")))
    internal = home / "venv" / "bin" / "sparsify"
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


@main.command("version")
def version_cmd() -> None:
    """Show the installed version and whether an update is available."""
    from sparsify.runtime import updater

    st = updater.check(force=True)
    console.print(f"[bold cyan]Sparsify[/bold cyan] {__version__}"
                  + (f"  ([dim]{st['current']}[/dim])" if st["current"] else ""))
    if st["update_available"]:
        console.print(f"[bold yellow]Update available[/bold yellow] "
                      f"([dim]{st['current']} → {st['latest']}[/dim]) — run: "
                      f"[bold]sparsify update[/bold]")
    elif st["latest"]:
        console.print("[green]You're on the latest version.[/green]")
    else:
        console.print("[dim]Could not reach GitHub to check for updates.[/dim]")


@main.command("update")
@click.option("--no-restart", is_flag=True, help="Don't restart the login service after updating.")
def update_cmd(no_restart: bool) -> None:
    """Update Sparsify to the latest version (git pull + reinstall)."""
    from sparsify.runtime import updater

    def log(msg: str) -> None:
        console.print(f"[dim]{msg}[/dim]")

    try:
        old, new = updater.do_update(log=log, restart=not no_restart)
    except RuntimeError as exc:
        console.print(f"[red]Update failed:[/red] {exc}")
        raise SystemExit(1)
    if old == new:
        return
    console.print(f"[bold green]Updated[/bold green] {old} → {new}. "
                  "Restart 'sparsify run' sessions to load the new version.")


@main.command("ps")
@click.option("--port", "-p", default=7777, show_default=True)
def ps_cmd(port: int) -> None:
    """Show the running API service: loaded model, memory, cache stats."""
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3) as r:
            health = _json.load(r)
    except (OSError, ValueError):
        console.print(f"[dim]No Sparsify server on port {port}.[/dim] "
                      "Start one: sparsify start   (or: sparsify serve)")
        return
    if not isinstance(health, dict) or health.get("runtime") != "sparsify":
        console.print(f"[yellow]Port {port} is answering, but it isn't "
                      f"Sparsify.[/yellow] [dim]Try: sparsify serve --port {port + 1}[/dim]")
        return

    table = Table(title=f"Sparsify service · localhost:{port}",
                  title_style="bold cyan", show_header=False, box=None,
                  padding=(0, 2))
    table.add_row("status", "[green]running[/green]")
    table.add_row("loaded model", health.get("loaded") or "[dim]none — loads on first request[/dim]")
    table.add_row("models dir", "[green]accessible[/green]"
                  if health.get("models_dir_accessible")
                  else "[red]not accessible[/red]")
    stats = health.get("stats")
    if stats:
        table.add_row("resident", f"{stats['resident_bytes']/1e9:.2f} / "
                                  f"{stats['budget_bytes']/1e9:.2f} GB expert cache")
        table.add_row("hit rate", f"{stats['hit_rate']*100:.1f}%  "
                                  f"(hits {stats['hits']:,} · misses {stats['misses']:,})")
        table.add_row("SSD reads", f"{stats['bytes_read']/1e9:.2f} GB total")
    console.print(table)


# System-wide launcher locations. A module constant so tests can stub it.
_SYSTEM_BIN_DIRS = (Path("/usr/local/bin"),)


def _our_launchers(home: Path) -> list[Path]:
    """Launcher symlinks that provably resolve into *home* — never a
    same-named foreign binary (PyPI also has a 'sparsify' package)."""
    import os

    found = []
    home_real = Path(os.path.realpath(home))
    for bin_dir in (Path.home() / ".local" / "bin", *_SYSTEM_BIN_DIRS):
        p = bin_dir / "sparsify"
        try:
            if not p.is_symlink():
                continue
            target = Path(os.path.realpath(p))
            if home_real == target or home_real in target.parents:
                found.append(p)
        except OSError:
            continue
    return found


def _sparsify_model_dirs() -> list[Path]:
    """Model directories that are ours to delete: pulled-model layouts only
    (config.json present). Anything else in a shared models dir is left."""
    if not MODELS_DIR.exists():
        return []
    return [d for d in sorted(MODELS_DIR.iterdir())
            if d.is_dir() and (d / "config.json").exists()]


def _remove_tree_except(root: Path, spare_top: Path | None) -> None:
    """rmtree *root*, optionally sparing one immediate child."""
    import shutil

    if spare_top is None:
        shutil.rmtree(root)
        return
    for child in root.iterdir():
        if child == spare_top:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        else:
            shutil.rmtree(child)


@main.command("uninstall")
@click.option("--yes", is_flag=True, help="Do not ask for confirmation.")
@click.option("--keep-models", is_flag=True,
              help="Leave downloaded model weights on disk.")
def uninstall_cmd(yes: bool, keep_models: bool) -> None:
    """Remove Sparsify from this machine — service, install, launchers,
    and (unless --keep-models) downloaded models."""
    import os
    import shutil
    import subprocess
    import sys
    from sparsify.runtime.model_registry import models_dir_status

    home = Path(os.environ.get("SPARSIFY_HOME", str(Path.home() / ".sparsify")))
    home_real = Path(os.path.realpath(home))

    # Refuse obviously-catastrophic homes: SPARSIFY_HOME=$HOME or /
    # produces a working install whose uninstall would wipe everything.
    if home_real in (Path("/"), Path(os.path.realpath(Path.home()))) \
            or len(home_real.parts) <= 2:
        console.print(f"[red]Refusing: SPARSIFY_HOME resolves to {home_real} — "
                      f"removing it would destroy far more than Sparsify.[/red]")
        console.print("[dim]Delete the Sparsify files inside it by hand.[/dim]")
        raise SystemExit(1)

    dir_status = models_dir_status()
    models_real = Path(os.path.realpath(MODELS_DIR))
    models_inside_home = home_real == models_real or home_real in models_real.parents

    # what actually gets deleted: (label, path, kind)
    plan: list[tuple[str, Path, str]] = []
    notes: list[str] = []

    if _PLIST_PATH.exists():
        plan.append(("login service", _PLIST_PATH, "file"))

    spare_top: Path | None = None
    if home.exists():
        if keep_models and models_inside_home and dir_status == "ok":
            spare_top = home / models_real.relative_to(home_real).parts[0]
            plan.append(("install (venv, source, logs, config)", home, "home-except"))
            notes.append(f"models kept in {MODELS_DIR}")
        else:
            label = "install (venv, source, logs, config)"
            if models_inside_home and dir_status == "ok" and not keep_models:
                size = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
                label = f"install + models ({size / 1e9:.1f} GB, {MODELS_DIR})"
            plan.append((label, home, "tree"))

    if not keep_models and not models_inside_home:
        if dir_status == "ok":
            model_dirs = _sparsify_model_dirs()
            if model_dirs:
                size = sum(f.stat().st_size for d in model_dirs
                           for f in d.rglob("*") if f.is_file())
                plan.append((f"{len(model_dirs)} models ({size / 1e9:.1f} GB) "
                             f"in {MODELS_DIR}", MODELS_DIR, "models"))
        else:
            notes.append(f"models at {MODELS_DIR} are NOT reachable right now "
                         f"({dir_status}) and will NOT be removed — delete them "
                         f"from that disk yourself")
    elif keep_models and not models_inside_home:
        notes.append(f"models kept in {MODELS_DIR}")

    for launcher in _our_launchers(home):
        plan.append(("launcher", launcher, "file"))

    if not plan:
        console.print("[dim]Nothing to remove — Sparsify is not installed.[/dim]")
        return

    console.print("[bold]This will remove:[/bold]")
    for label, path, _kind in plan:
        console.print(f"  · {label}:  [dim]{path}[/dim]")
    for note in notes:
        console.print(f"  [yellow]! {note}[/yellow]")
    if not yes:
        if not sys.stdin.isatty():
            console.print("[red]Refusing to uninstall without confirmation — "
                          "pass --yes in scripts.[/red]")
            raise SystemExit(1)
        if not click.confirm("Continue?", default=False):
            console.print("[dim]Nothing removed.[/dim]")
            return

    subprocess.run(["launchctl", "unload", str(_PLIST_PATH)], capture_output=True)
    removed, failed = [], []
    for label, path, kind in plan:
        try:
            if kind == "home-except":
                _remove_tree_except(path, spare_top)
            elif kind == "models":
                for d in _sparsify_model_dirs():
                    shutil.rmtree(d)
                (path / ".registry.json").unlink(missing_ok=True)
                leftovers = list(path.iterdir())
                if leftovers:
                    notes.append(f"left non-Sparsify files in {path}")
                elif path.is_symlink():
                    path.unlink()
                else:
                    path.rmdir()
            elif path.is_symlink() or path.is_file():
                path.unlink()
            else:
                shutil.rmtree(path)
            removed.append(label)
        except OSError as exc:
            failed.append(label)
            console.print(f"[red]Could not remove {path}: {exc}[/red]")

    if removed:
        console.print(f"[bold green]Removed:[/bold green] {', '.join(removed)}.")
    for note in notes:
        console.print(f"  [dim]{note}[/dim]")
    if failed:
        console.print(f"[red]Failed to remove:[/red] {', '.join(failed)} — "
                      "see errors above.")
        raise SystemExit(1)
    if not removed:
        console.print("[yellow]Nothing was removed.[/yellow]")
        raise SystemExit(1)
    console.print("[dim]Goodbye. Reinstall any time:\n  curl -fsSL "
                  "https://github.com/daylinkltd/sparsify/releases/latest/"
                  "download/install.sh | sh[/dim]")


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
