"""Interactive chat TUI for `sparsify run`.

A real terminal application (prompt_toolkit full-screen layout), not a
scrolling prompt: a scrollable transcript pane on top, the input field
always live above a status bar. Messages typed while the model is busy
queue up and run in order. The Sparsify logo (sparse expert cells)
animates inline while the model routes, and every answer ends with
measured telemetry.

Threading model: MLX GPU streams are bound to the thread that creates
them, so the engine is loaded AND run on one dedicated worker thread;
the UI thread never touches MLX.

When stdin is not a terminal (pipes, CI) a headless line-mode runs
instead so scripted sessions keep working.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

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
    """Full-screen chat application over a SparsifyEngine."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()
        self.session_tokens = 0
        self._model_tag = ""
        self._busy = ""            # "", "thinking", "streaming"
        self._jobs: "queue.Queue[str | None]" = queue.Queue()
        self._engine = None
        self._load_error: Exception | None = None
        self._loaded = threading.Event()
        self._app = None
        # transcript: list of [style, text] segments (mutable for streaming)
        self._lines: list[list] = []
        self._pinned = True        # auto-scroll unless the user scrolled up
        # messages accepted but not yet started — shown above the input
        # field; they move into the transcript when the model picks them up
        self._pending: list[str] = []

    # ── transcript helpers (safe from any thread) ──────────────────────

    def _append(self, style: str, text: str) -> list:
        seg = [style, text]
        self._lines.append(seg)
        self._refresh()
        return seg

    def _refresh(self) -> None:
        if self._app is not None:
            if self._pinned:
                self._scroll_to_bottom()
            self._app.invalidate()

    def _scroll_to_bottom(self) -> None:
        if getattr(self, "_transcript_window", None) is not None:
            self._transcript_window.vertical_scroll = 10 ** 6  # clamped

    def _transcript_fragments(self):
        if not self._lines:
            return [("class:dim", "\n  Say something — experts page in from "
                                  "SSD as the router selects them.\n")]
        frags = []
        for style, text in self._lines:
            if style == "think":
                frame = _LOGO_FRAMES[int(time.monotonic() * 7) % len(_LOGO_FRAMES)]
                frags.append(("class:think", f"{frame} {text}"))
            else:
                frags.append((f"class:{style}" if style else "", text))
        return frags

    # ── public lifecycle ───────────────────────────────────────────────

    def banner(self, hf_id: str, model_path: Path, device: str,
               memory_limit) -> None:
        self._model_tag = hf_id.split("/")[-1]
        self._hf_id = hf_id
        self._device = device

    def run(self, engine_factory) -> None:
        worker = threading.Thread(target=self._worker, args=(engine_factory,),
                                  daemon=True, name="sparsify-engine")
        worker.start()
        with self.console.status(f"[dim]loading {self._model_tag}…[/dim]",
                                 spinner="sparsify", spinner_style="yellow"):
            self._loaded.wait()
        if self._load_error is not None:
            self.console.print(f"[red]Failed to load model: {self._load_error}[/red]")
            raise SystemExit(1)

        if sys.stdin.isatty() and sys.stdout.isatty():
            self._run_app()
        else:
            self._run_headless()

        self._jobs.put(None)
        if self._busy or self._jobs.qsize() > 1:
            self.console.print("[dim]finishing queued messages… (Ctrl-C to abort)[/dim]")
        try:
            worker.join()
        except KeyboardInterrupt:
            pass
        self.console.print(f"[dim]session ended · {self.session_tokens} tokens generated.[/dim]")

    # ── the application ────────────────────────────────────────────────

    def _run_app(self) -> None:
        from prompt_toolkit.application import Application
        from prompt_toolkit.buffer import Buffer
        from prompt_toolkit.enums import EditingMode
        from prompt_toolkit.filters import Condition
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import (Dimension, FormattedTextControl,
                                           HSplit, Layout, VSplit, Window)
        from prompt_toolkit.layout.controls import BufferControl
        from prompt_toolkit.styles import Style

        input_buffer = Buffer(multiline=True)

        header = Window(
            FormattedTextControl(self._header_fragments),
            height=1, style="class:bar")
        self._transcript_window = Window(
            FormattedTextControl(self._transcript_fragments, show_cursor=False),
            wrap_lines=True, right_margins=[])
        divider = Window(height=1, char="─", style="class:line")
        prompt_label = Window(
            FormattedTextControl([("class:you", " ❯ ")]),
            width=3, dont_extend_width=True)
        input_window = Window(
            BufferControl(buffer=input_buffer),
            height=Dimension(min=1, max=6), wrap_lines=True)
        status = Window(
            FormattedTextControl(self._status_fragments),
            height=1, style="class:bar")

        from prompt_toolkit.layout import ConditionalContainer

        def _pending_fragments():
            frags = []
            for t in list(self._pending):
                shown = t if len(t) <= 72 else t[:69] + "…"
                frags.append(("class:queued", f"   ⏳ {shown}\n"))
            return frags

        queued_strip = ConditionalContainer(
            Window(FormattedTextControl(_pending_fragments),
                   height=lambda: Dimension.exact(min(len(self._pending), 3)),
                   style="class:queuedbg"),
            filter=Condition(lambda: bool(self._pending)),
        )

        root = HSplit([
            header,
            self._transcript_window,
            queued_strip,
            divider,
            VSplit([prompt_label, input_window]),
            status,
        ])

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event):
            text = input_buffer.text.strip()
            input_buffer.reset()
            if not text:
                return
            if text.lower() in ("/exit", "/quit", "exit", "quit"):
                event.app.exit()
                return
            # echoed into the transcript only when the model starts on it
            self._pending.append(text)
            self._jobs.put(text)
            self._refresh()

        @kb.add("escape", "enter")
        def _newline(event):
            input_buffer.insert_text("\n")

        @kb.add("c-c")
        @kb.add("c-d")
        def _quit(event):
            event.app.exit()

        @kb.add("pageup")
        def _pgup(event):
            self._pinned = False
            w = self._transcript_window
            w.vertical_scroll = max(0, w.vertical_scroll - 10)

        @kb.add("pagedown")
        def _pgdn(event):
            w = self._transcript_window
            w.vertical_scroll += 10
            info = w.render_info
            if info is None or w.vertical_scroll + getattr(info, "window_height", 0) \
                    >= getattr(info, "content_height", 0):
                self._pinned = True

        style = Style.from_dict({
            "bar":        "bg:#1a2130 #9aa5b8",
            "bar.model":  "bold #e7ebf2",
            "bar.accent": "bold #e8a33d",
            "bar.good":   "#3fb27f",
            "line":       "#2a3346",
            "you":        "bold #e8a33d",
            "bot":        "",
            "queued":     "#a08040",
            "queuedbg":   "",
            "stats":      "#5b6577",
            "think":      "#e8a33d",
            "dim":        "#5b6577",
            "err":        "#c0504a",
        })

        self._app = Application(
            layout=Layout(root, focused_element=input_window),
            key_bindings=kb,
            style=style,
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.25,   # animates the routing logo
            editing_mode=EditingMode.EMACS,
        )
        try:
            self._app.run()
        finally:
            self._app = None

    def _header_fragments(self):
        engine = self._engine
        parts = [("class:bar", "  "),
                 ("class:bar.accent", "▚ "),
                 ("class:bar.model", "sparsify"),
                 ("class:bar", "  ·  "),
                 ("class:bar.model", self._model_tag),
                 ("class:bar", f"  ·  {self._device}")]
        if engine and engine.paging:
            p = engine.paging.stats()
            if p.get("resident_blocks") == p.get("moe_blocks"):
                parts.append(("class:bar", "  ·  all experts resident — native speed"))
            else:
                parts.append(("class:bar",
                              f"  ·  {p['paged_gb']:.1f} GB experts on SSD · "
                              f"budget {engine.memory_limit_gb:.1f} GB"))
        parts.append(("class:bar.good", "   ● "))
        return parts

    def _status_fragments(self):
        parts = [("class:bar", "  ")]
        if self._busy:
            frame = _LOGO_FRAMES[int(time.monotonic() * 7) % len(_LOGO_FRAMES)]
            parts += [("class:bar.accent", f"{frame} {self._busy}… ")]
        pending = self._jobs.qsize()
        if pending:
            parts += [("class:bar", "· "), ("class:bar.model", f"{pending} queued "),
                      ("class:bar", "· ")]
        parts += [("class:bar", f"session {self.session_tokens} tok  ·  ")]
        parts += [("class:bar.model", "enter"), ("class:bar", " send · "),
                  ("class:bar.model", "esc+enter"), ("class:bar", " newline · "),
                  ("class:bar.model", "pgup/pgdn"), ("class:bar", " scroll · "),
                  ("class:bar.model", "/help"), ("class:bar", " commands  ")]
        return parts

    # ── headless line mode (pipes / CI) ────────────────────────────────

    def _run_headless(self) -> None:
        for line in sys.stdin:
            text = line.strip()
            if not text:
                continue
            if text.lower() in ("/exit", "/quit", "exit", "quit"):
                break
            self._jobs.put(text)

    # ── engine worker (owns all MLX state) ─────────────────────────────

    def _worker(self, engine_factory) -> None:
        try:
            self._engine = engine_factory()
        except Exception as exc:
            self._load_error = exc
            self._loaded.set()
            return
        self._loaded.set()
        while True:
            item = self._jobs.get()
            if item is None:
                return
            try:
                self._pending.remove(item)
            except ValueError:
                pass
            if self._app is not None:
                self._append("you", f"\n ❯ {item}\n")
            try:
                if item.startswith("/"):
                    self._command(item)
                else:
                    self._respond(item)
            except Exception as exc:
                self._busy = ""
                self._emit("err", f"error: {exc}\n")
            finally:
                self._jobs.task_done()

    def _emit(self, style: str, text: str) -> None:
        """Write a transcript segment (app) or plain stdout (headless)."""
        if self._app is not None or not self._loaded.is_set():
            self._append(style, text)
        else:
            sys.stdout.write(text)
            sys.stdout.flush()

    def _respond(self, prompt: str) -> None:
        engine = self._engine
        self._busy = "thinking"
        headless = self._app is None
        if headless:
            sys.stdout.write(f"\n❯ {prompt}\n")
        thinking = self._append("think", "routing experts…") if not headless else None
        answer = None
        last = None
        try:
            for text, tel in engine.generate_stream(prompt):
                last = tel
                if self._busy != "streaming":
                    self._busy = "streaming"
                    if thinking is not None:
                        self._lines.remove(thinking)
                        answer = self._append("bot", "")
                if headless:
                    sys.stdout.write(text)
                    sys.stdout.flush()
                else:
                    answer[1] += text
                    self._refresh()
        finally:
            self._busy = ""
            if thinking is not None and thinking in self._lines:
                self._lines.remove(thinking)  # no tokens arrived
        if headless:
            sys.stdout.write("\n")

        if last:
            self.session_tokens += last["n_tokens"]
            parts = [f"{last['n_tokens']} tokens",
                     f"{last['throughput']:.1f} tok/s",
                     f"rss {last.get('rss_gb', 0):.2f} GB"]
            if last.get("kv_reused_tokens"):
                parts.append(f"ctx {last['context_tokens']} "
                             f"({last['kv_reused_tokens']} reused)")
            if "paging" in last:
                parts.append(f"cache {last['paging']['hit_rate']*100:.0f}% hit")
            self._emit("stats", "\n  " + " · ".join(parts) + "\n")
        self._refresh()

    # ── slash commands ─────────────────────────────────────────────────

    def _command(self, prompt: str) -> None:
        engine = self._engine
        cmd, _, arg = prompt.partition(" ")
        cmd = cmd.lower()
        if cmd == "/help":
            out = "\n" + "".join(f"  {name:<9} {desc}\n"
                                 for name, desc in COMMANDS.items())
            self._emit("dim", out)
        elif cmd == "/clear":
            engine.messages.clear()
            self._lines.clear()
            self._emit("dim", "\n  conversation cleared.\n")
        elif cmd == "/model":
            p = engine.paging.stats() if engine.paging else None
            out = [f"\n  model     {engine.model_path.name}",
                   f"  backbone  {engine.model_memory_gb:.2f} GB resident"]
            if p:
                out.append(f"  experts   {p['paged_gb']:.1f} GB "
                           f"({p['moe_blocks']} MoE blocks, "
                           f"{p['resident_blocks']} fully resident)")
                out.append(f"  budget    {engine.memory_limit_gb:.2f} GB")
            out.append(f"  history   {len(engine.messages)} messages\n")
            self._emit("dim", "\n".join(out))
        elif cmd == "/stats":
            if not engine.paging:
                self._emit("dim", "\n  dense model — no paging stats.\n")
                return
            p = engine.paging.stats()
            out = ["\n  paging · all values measured"]
            if p["resident_blocks"]:
                out.append(f"  resident   {p['resident_blocks']}/{p['moe_blocks']} "
                           f"blocks ({p['resident_full_bytes']/1e9:.2f} GB) — native path")
            out.append(f"  cache      {p['resident_bytes']/1e9:.2f} / "
                       f"{p['budget_bytes']/1e9:.2f} GB "
                       f"({p['resident_experts']} experts)")
            out.append(f"  hit rate   {p['hit_rate']*100:.1f}%  "
                       f"(hits {p['hits']:,} · misses {p['misses']:,} · "
                       f"evictions {p['evictions']:,})")
            out.append(f"  SSD        {p['bytes_read']/1e9:.2f} GB in "
                       f"{p['reads']:,} reads ({p['read_seconds']:.1f}s summed I/O)\n")
            self._emit("dim", "\n".join(out))
        elif cmd == "/budget":
            if not engine.paging:
                self._emit("dim", "\n  dense model — there is no expert cache.\n")
                return
            try:
                gb = float(arg)
                if gb <= 0:
                    raise ValueError
            except ValueError:
                self._emit("dim", "\n  usage: /budget <gigabytes>, e.g. /budget 3\n")
                return
            engine.paging.cache.budget_bytes = int(gb * 1024**3)
            engine.memory_limit_gb = gb
            self._emit("dim", f"\n  expert-cache budget set to {gb:.1f} GB "
                              "(applies from the next token)\n")
        else:
            self._emit("dim", f"\n  unknown command {cmd} — try /help\n")


# Backwards-compatible alias (older callers imported ClaudeCodeUI)
ClaudeCodeUI = ChatUI
