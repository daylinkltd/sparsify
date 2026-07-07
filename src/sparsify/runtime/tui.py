"""Interactive chat TUI for `sparsify run`.

Design goals: the information density of a systems tool with the polish of
a modern coding agent — live token/memory/cache telemetry in the footer
while streaming, slash commands with completion, markdown answers.
"""
from __future__ import annotations

from pathlib import Path

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style

COMMANDS = {
    "/help":   "show available commands",
    "/stats":  "runtime + paging statistics for this session",
    "/model":  "details about the loaded model",
    "/budget": "set the expert-cache budget, e.g. /budget 3",
    "/clear":  "clear the conversation history",
    "/exit":   "leave the chat",
}


class ChatUI:
    """Claude-Code-style chat surface over a SparsifyEngine."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.session_tokens = 0
        self._model_tag = ""

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
        )

    # ── chrome ─────────────────────────────────────────────────────────

    def _toolbar(self):
        return HTML(
            f" <b>{self._model_tag}</b>"
            f"  ·  session {self.session_tokens} tok"
            "  ·  <b>enter</b> send · <b>esc+enter</b> newline · /help"
        )

    def banner(self, hf_id: str, model_path: Path, device: str,
               memory_limit) -> None:
        self._model_tag = hf_id.split("/")[-1]
        cap = f"{memory_limit} GB" if memory_limit is not None else "auto"
        self.console.print()
        self.console.print(Rule("[bold]sparsify[/bold]", style="yellow"))
        self.console.print(f"  [bold white]{hf_id}[/bold white]")
        self.console.print(f"  [dim]{device} · expert cache {cap} · {model_path}[/dim]")

    def ready(self, engine) -> None:
        p = engine.paging.stats() if engine.paging else None
        if p:
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

    # ── main loop ──────────────────────────────────────────────────────

    def chat_loop(self, engine) -> None:
        while True:
            try:
                prompt = self.session.prompt().strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not prompt:
                continue
            if prompt.startswith("/"):
                if not self._command(prompt, engine):
                    break
                continue
            self._respond(engine, prompt)
        self.console.print("\n[dim]session ended.[/dim]")

    def _command(self, prompt: str, engine) -> bool:
        """Handle a slash command; returns False to exit the loop."""
        cmd, _, arg = prompt.partition(" ")
        cmd = cmd.lower()
        if cmd in ("/exit", "/quit"):
            return False
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
        return True

    def _print_model(self, engine) -> None:
        t = Table(show_header=False, box=None, padding=(0, 2))
        t.add_row("model", str(engine.model_path.name))
        t.add_row("backbone", f"{engine.model_memory_gb:.2f} GB resident")
        if engine.paging:
            p = engine.paging.stats()
            t.add_row("experts", f"{p['paged_gb']:.1f} GB on SSD "
                                 f"({p['moe_blocks']} MoE blocks, "
                                 f"{p['replaced_modules']} paged projections)")
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
        t.add_row("expert cache", f"{p['resident_bytes']/1e9:.2f} / "
                                  f"{p['budget_bytes']/1e9:.2f} GB "
                                  f"({p['resident_experts']} experts resident)")
        t.add_row("hit rate", f"{p['hit_rate']*100:.1f}%  "
                              f"(hits {p['hits']:,} · misses {p['misses']:,} "
                              f"· evictions {p['evictions']:,})")
        t.add_row("SSD traffic", f"{p['bytes_read']/1e9:.2f} GB in "
                                 f"{p['reads']:,} reads "
                                 f"({p['read_seconds']:.1f}s I/O)")
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
                           "[dim](applies from the next token; shrinking evicts lazily)[/dim]")

    # ── streaming answer ───────────────────────────────────────────────

    def _respond(self, engine, prompt: str) -> None:
        self.console.print()
        full = ""
        last = None
        status = Text("thinking…", style="dim")
        with Live(status, console=self.console, refresh_per_second=16,
                  vertical_overflow="visible") as live:
            for text, tel in engine.generate_stream(prompt):
                full += text
                last = tel
                footer_parts = [
                    ("⏺ ", "yellow"),
                    (f"{tel['n_tokens']} tok", "bold"),
                    (f" · {tel['throughput']:.1f} tok/s", ""),
                    (f" · rss {tel.get('rss_gb', 0):.2f} GB", ""),
                ]
                if "paging" in tel:
                    footer_parts.append(
                        (f" · cache {tel['paging']['hit_rate']*100:.0f}%", ""))
                    footer_parts.append(
                        (f" · ssd {tel['paging']['bytes_read']/1e9:.1f} GB", ""))
                footer = Text.assemble(*footer_parts, style="dim")
                live.update(Group(Markdown(full), Text(), footer))
            live.update(Markdown(full))

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


# Backwards-compatible alias (older callers imported ClaudeCodeUI)
ClaudeCodeUI = ChatUI
