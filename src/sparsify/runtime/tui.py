"""Interactive chat TUI for `sparsify run`.

Claude-Code-style surface: the input box is always live at the bottom —
messages typed while the model is generating queue up and run in order.
The Sparsify logo (sparse expert cells) animates in the toolbar while the
model thinks, and every answer ends with measured telemetry.

Threading model: MLX GPU streams are bound to the thread that creates
them, so the engine is loaded AND run on one dedicated worker thread;
the main thread only owns the prompt.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

# the logo as animation frames: a 2x2 expert grid, sparse cells lit
_LOGO_FRAMES = "▖▘▝▗▚▞"

try:  # register a branded spinner for rich status displays
    from rich._spinners import SPINNERS
    SPINNERS.setdefault("sparsify", {"interval": 140,
                                     "frames": list(_LOGO_FRAMES)})
except ImportError:
    pass

COMMANDS = {
    "/help":   "show available commands",
    "/stats":  "runtime + paging statistics for this session",
    "/model":  "details about the loaded model",
    "/budget": "set the expert-cache budget, e.g. /budget 3",
    "/clear":  "clear the conversation history",
    "/exit":   "leave the chat",
}


class ChatUI:
    """Persistent-prompt chat surface over a SparsifyEngine."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.session_tokens = 0
        self._model_tag = ""
        self._busy = ""            # "", "thinking", "streaming"
        self._jobs: "queue.Queue[str | None]" = queue.Queue()
        self._engine = None
        self._load_error: Exception | None = None
        self._loaded = threading.Event()

        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event):
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def _(event):
            event.current_buffer.insert_text("\n")

        self.session = PromptSession(
            message=HTML("<b><ansiyellow>❯</ansiyellow></b> "),
            style=Style.from_dict({"prompt": "ansiyellow bold"}),
            multiline=True,
            key_bindings=bindings,
            completer=WordCompleter(list(COMMANDS), sentence=True),
            complete_while_typing=True,
            prompt_continuation=HTML("<ansiyellow>  </ansiyellow>"),
            bottom_toolbar=self._toolbar,
            refresh_interval=0.25,   # animates the toolbar logo
        )

    # ── chrome ─────────────────────────────────────────────────────────

    def _toolbar(self):
        parts = []
        if self._busy:
            frame = _LOGO_FRAMES[int(time.monotonic() * 7) % len(_LOGO_FRAMES)]
            parts.append(f"<ansiyellow><b>{frame}</b></ansiyellow> {self._busy}…")
        pending = self._jobs.qsize()
        if pending:
            parts.append(f"<b>{pending}</b> queued")
        parts.append(f"<b>{self._model_tag}</b>")
        parts.append(f"session {self.session_tokens} tok")
        parts.append("<b>enter</b> send · <b>esc+enter</b> newline · /help")
        return HTML("  ·  ".join(parts))

    def banner(self, hf_id: str, model_path: Path, device: str,
               memory_limit) -> None:
        self._model_tag = hf_id.split("/")[-1]
        cap = f"{memory_limit} GB" if memory_limit is not None else "auto"
        self.console.print()
        self.console.print(Rule("[bold]sparsify[/bold]", style="yellow"))
        self.console.print(f"  [bold white]{hf_id}[/bold white]")
        self.console.print(f"  [dim]{device} · expert cache {cap} · {model_path}[/dim]")

    def _ready_line(self) -> None:
        engine = self._engine
        p = engine.paging.stats() if engine.paging else None
        if p and p.get("resident_blocks") == p.get("moe_blocks"):
            line = (f"  [green]●[/green] ready — all "
                    f"[bold]{p['paged_gb']:.1f} GB[/bold] of experts resident "
                    f"(fits the budget) — native speed")
        elif p:
            line = (f"  [green]●[/green] ready — backbone "
                    f"[bold]{engine.model_memory_gb:.2f} GB[/bold] in RAM, "
                    f"[bold]{p['paged_gb']:.1f} GB[/bold] of experts on SSD, "
                    f"budget [bold]{engine.memory_limit_gb:.1f} GB[/bold] "
                    f"({p['moe_blocks']} MoE blocks)")
        else:
            line = (f"  [green]●[/green] ready — dense model, "
                    f"[bold]{engine.model_memory_gb:.2f} GB[/bold] resident "
                    f"(no paging needed)")
        self.console.print(line)
        self.console.print(Rule(style="bright_black"))
        self.console.print()

    # ── engine worker (owns all MLX state) ─────────────────────────────

    def _worker(self, engine_factory) -> None:
        try:
            self._engine = engine_factory()
        except Exception as exc:  # surface to main thread
            self._load_error = exc
            self._loaded.set()
            return
        self._loaded.set()
        while True:
            item = self._jobs.get()
            if item is None:
                return
            try:
                if item.startswith("/"):
                    self._command(item)
                else:
                    self._respond(item)
            except Exception as exc:
                self._busy = ""
                self.console.print(f"[red]error: {exc}[/red]")
            finally:
                self._jobs.task_done()

    def run(self, engine_factory) -> None:
        """Load the engine on the worker thread, then run the prompt loop."""
        self._worker_thread = threading.Thread(
            target=self._worker, args=(engine_factory,),
            daemon=True, name="sparsify-engine")
        worker = self._worker_thread
        worker.start()
        with self.console.status("[dim]loading weights…[/dim]",
                                 spinner="sparsify", spinner_style="yellow"):
            self._loaded.wait()
        if self._load_error is not None:
            self.console.print(f"[red]Failed to load model: {self._load_error}[/red]")
            raise SystemExit(1)
        self._ready_line()

        with patch_stdout(raw=True):
            while True:
                try:
                    text = self.session.prompt()
                except (KeyboardInterrupt, EOFError):
                    break
                text = (text or "").strip()
                if not text:
                    continue
                if text.lower() in ("/exit", "/quit", "exit", "quit"):
                    break
                self._jobs.put(text)
        self._jobs.put(None)
        if self._busy or self._jobs.qsize() > 1:
            self.console.print("[dim]finishing queued messages… "
                               "(Ctrl-C to abort)[/dim]")
        try:
            self._worker_thread.join()
        except KeyboardInterrupt:
            pass
        self.console.print("\n[dim]session ended.[/dim]")

    # ── worker-side handlers ───────────────────────────────────────────

    def _respond(self, prompt: str) -> None:
        engine = self._engine
        self._busy = "thinking"
        last = None
        out = sys.stdout
        out.write("\n")
        try:
            for text, tel in engine.generate_stream(prompt):
                if self._busy != "streaming":
                    self._busy = "streaming"
                out.write(text)
                out.flush()
                last = tel
        finally:
            self._busy = ""
        out.write("\n")
        out.flush()

        if last:
            self.session_tokens += last["n_tokens"]
            parts = [
                f"{last['n_tokens']} tokens",
                f"{last['throughput']:.1f} tok/s",
                f"rss {last.get('rss_gb', 0):.2f} GB",
                f"peak {last['peak_gb']:.2f} GB",
            ]
            if "paging" in last:
                parts.append(f"cache {last['paging']['hit_rate']*100:.0f}% hit")
            self.console.print(Rule(" · ".join(parts), style="bright_black",
                                    align="right"))
        self.console.print()

    def _command(self, prompt: str) -> None:
        engine = self._engine
        cmd, _, arg = prompt.partition(" ")
        cmd = cmd.lower()
        if cmd == "/help":
            t = Table(show_header=False, box=None, padding=(0, 2))
            for name, desc in COMMANDS.items():
                t.add_row(f"[yellow]{name}[/yellow]", f"[dim]{desc}[/dim]")
            self.console.print(t)
        elif cmd == "/clear":
            engine.messages.clear()
            self.console.print("[dim]conversation cleared.[/dim]")
        elif cmd == "/model":
            self._print_model(engine)
        elif cmd == "/stats":
            self._print_stats(engine)
        elif cmd == "/budget":
            self._set_budget(engine, arg)
        else:
            self.console.print(f"[dim]unknown command {cmd} — try /help[/dim]")

    def _print_model(self, engine) -> None:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_row("model", str(engine.model_path.name))
        t.add_row("backbone", f"{engine.model_memory_gb:.2f} GB resident")
        if engine.paging:
            p = engine.paging.stats()
            t.add_row("experts", f"{p['paged_gb']:.1f} GB "
                                 f"({p['moe_blocks']} MoE blocks, "
                                 f"{p['resident_blocks']} fully resident)")
            t.add_row("budget", f"{engine.memory_limit_gb:.2f} GB")
        t.add_row("history", f"{len(engine.messages)} messages")
        self.console.print(Panel(t, border_style="bright_black",
                                 title="model", title_align="left"))

    def _print_stats(self, engine) -> None:
        if not engine.paging:
            self.console.print("[dim]dense model — no paging stats.[/dim]")
            return
        p = engine.paging.stats()
        t = Table(show_header=False, box=None, padding=(0, 2))
        if p["resident_blocks"]:
            t.add_row("resident", f"{p['resident_blocks']}/{p['moe_blocks']} blocks "
                                  f"({p['resident_full_bytes']/1e9:.2f} GB) — native path")
        t.add_row("expert cache", f"{p['resident_bytes']/1e9:.2f} / "
                                  f"{p['budget_bytes']/1e9:.2f} GB "
                                  f"({p['resident_experts']} experts resident)")
        t.add_row("hit rate", f"{p['hit_rate']*100:.1f}%  "
                              f"(hits {p['hits']:,} · misses {p['misses']:,} "
                              f"· evictions {p['evictions']:,})")
        t.add_row("SSD traffic", f"{p['bytes_read']/1e9:.2f} GB in "
                                 f"{p['reads']:,} reads "
                                 f"({p['read_seconds']:.1f}s summed I/O)")
        self.console.print(Panel(t, border_style="bright_black",
                                 title="paging · all values measured",
                                 title_align="left"))

    def _set_budget(self, engine, arg: str) -> None:
        if not engine.paging:
            self.console.print("[dim]dense model — there is no expert cache.[/dim]")
            return
        try:
            gb = float(arg)
            if gb <= 0:
                raise ValueError
        except ValueError:
            self.console.print("[dim]usage: /budget <gigabytes>, e.g. /budget 3[/dim]")
            return
        engine.paging.cache.budget_bytes = int(gb * 1024**3)
        engine.memory_limit_gb = gb
        self.console.print(f"[green]expert-cache budget set to {gb:.1f} GB[/green] "
                           "[dim](applies from the next token)[/dim]")


# Backwards-compatible alias (older callers imported ClaudeCodeUI)
ClaudeCodeUI = ChatUI
