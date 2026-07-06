"""Interactive Claude Code-style chat TUI for Sparsify."""
from __future__ import annotations

from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from rich.panel import Panel

from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.key_binding import KeyBindings

console = Console()

class ClaudeCodeUI:
    """A rich interactive chat UI matching the Claude Code terminal experience."""
    
    def __init__(self):
        bindings = KeyBindings()
        
        # Enter submits the query
        @bindings.add('enter')
        def _(event):
            event.current_buffer.validate_and_handle()
            
        # Alt+Enter / Esc+Enter inserts a newline
        @bindings.add('escape', 'enter')
        def _(event):
            event.current_buffer.insert_text('\n')

        self.session = PromptSession(
            message=HTML("<b><ansicyan>❯</ansicyan></b> "),
            style=Style.from_dict({
                'prompt': 'ansicyan bold',
            }),
            multiline=True,
            key_bindings=bindings,
            prompt_continuation=HTML('<ansicyan>  </ansicyan>')
        )

    def ask(self) -> str | None:
        """Prompt the user for input."""
        try:
            prompt = self.session.prompt()
            return prompt.strip()
        except (KeyboardInterrupt, EOFError):
            return None

    def stream_response(self, engine_generator):
        """Render markdown streamed from the generator."""
        full_text = ""
        console.print()
        
        from rich.console import Group
        from rich.text import Text
        
        status_text = Text("Thinking, processing...", style="cyan blink")
        
        with Live(
            Panel(status_text, border_style="cyan"), 
            console=console, 
            refresh_per_second=20, 
            vertical_overflow="visible"
        ) as live:
            is_thinking = True
            for text, telemetry in engine_generator:
                if is_thinking:
                    is_thinking = False
                    
                if text is not None:
                    full_text += text
                
                if telemetry is not None:
                    stats = (
                        f"Tokens: {telemetry['n_tokens']} | "
                        f"Speed: {telemetry['throughput']:.1f} tok/s | "
                        f"RAM: {telemetry['active_gb']:.2f} GB (Peak: {telemetry['peak_gb']:.2f} GB)"
                    )
                    header = Text(stats, style="dim")
                    content = Markdown(full_text)
                    
                    group = Group(
                        header,
                        "",
                        content
                    )
                    live.update(group)
                    
            # Make sure it prints at least the final text if no telemetry is yielded
            if is_thinking:
                live.update(Markdown(full_text))
