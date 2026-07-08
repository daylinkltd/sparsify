"""Browser control for the agent — DOM-driven, so text-only models can use it.

The model doesn't look at pixels (that needs a vision model — roadmap). It
reads the page as text + a numbered list of interactive elements, then acts
by index: "click 3", "type 'hi' into 2". A *persistent* browser profile
(~/.sparsify/browser-profile) means logins survive across runs — log into
Outlook once (headed) and scheduled tasks reuse the session.

Playwright's sync API is single-thread; the agent loop runs on one thread
(the engine worker or the CLI main thread), so one lazy session per process
is correct. Requires the optional browser extra:
    pip install 'sparsify[browser]' && python -m playwright install chromium
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

_PROFILE = Path(os.environ.get("SPARSIFY_HOME", str(Path.home() / ".sparsify"))) / "browser-profile"
_ELEM_SELECTOR = ("a, button, input, textarea, select, "
                  "[role=button], [role=link], [role=textbox], [contenteditable=true]")
_MAX_ELEMS = 60
_TEXT_LIMIT = 6000


def available() -> bool:
    try:
        import playwright.sync_api  # noqa: F401
        return True
    except ImportError:
        return False


class _Session:
    """One browser, lazily launched on first use, reused across tool calls."""

    def __init__(self) -> None:
        self._pw = None
        self._ctx = None
        self._page = None
        self._thread = None
        self._elements = []  # index -> ElementHandle from the last read

    def _ensure(self):
        import playwright.sync_api as pw

        if self._ctx is not None:
            if threading.get_ident() != self._thread:
                raise RuntimeError("browser was started on another thread")
            return
        _PROFILE.mkdir(parents=True, exist_ok=True)
        self._thread = threading.get_ident()
        self._pw = pw.sync_playwright().start()
        headed = os.environ.get("SPARSIFY_BROWSER_HEADED", "").lower() in ("1", "true", "yes")
        self._ctx = self._pw.chromium.launch_persistent_context(
            str(_PROFILE), headless=not headed,
            user_agent="Mozilla/5.0 (Macintosh; Apple Silicon) sparsify/0.2")
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()

    def close(self):
        try:
            if self._ctx: self._ctx.close()
            if self._pw: self._pw.stop()
        except Exception:
            pass
        self._ctx = self._pw = self._page = None

    # -- actions ---------------------------------------------------------

    def open(self, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        self._ensure()
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        self._page.wait_for_timeout(600)
        return self.read()

    def read(self) -> str:
        self._ensure()
        page = self._page
        title = page.title()
        try:
            body = page.inner_text("body")[:_TEXT_LIMIT]
        except Exception:
            body = ""
        self._elements = []
        lines = []
        for el in page.query_selector_all(_ELEM_SELECTOR):
            if len(self._elements) >= _MAX_ELEMS:
                break
            try:
                if not el.is_visible():
                    continue
                tag = el.evaluate("e => e.tagName.toLowerCase()")
                name = (el.get_attribute("aria-label") or el.inner_text().strip()
                        or el.get_attribute("placeholder")
                        or el.get_attribute("value") or el.get_attribute("name") or "")
                name = " ".join(name.split())[:60]
                kind = "input" if tag in ("input", "textarea", "select") else \
                       ("link" if tag == "a" else "button")
                self._elements.append(el)
                lines.append(f"[{len(self._elements)}] {kind}: {name or '(unlabeled)'}")
            except Exception:
                continue
        els = "\n".join(lines) or "(no interactive elements found)"
        return (f"URL: {page.url}\nTITLE: {title}\n\nTEXT:\n{body}\n\n"
                f"INTERACTIVE ELEMENTS (act by number):\n{els}")

    def _el(self, index):
        try:
            return self._elements[int(index) - 1]
        except (ValueError, IndexError):
            raise RuntimeError(f"no element [{index}] — call browser_read first")

    def click(self, index) -> str:
        self._ensure()
        self._el(index).click(timeout=10000)
        self._page.wait_for_timeout(700)
        return self.read()

    def type(self, index, text, submit=False) -> str:
        self._ensure()
        el = self._el(index)
        el.click(timeout=10000)
        el.fill(text)
        if submit:
            el.press("Enter")
            self._page.wait_for_timeout(800)
        return self.read()

    def screenshot(self, path: Path) -> str:
        self._ensure()
        self._page.screenshot(path=str(path), full_page=False)
        return str(path)


_SESSION: _Session | None = None


def session() -> _Session:
    global _SESSION
    if _SESSION is None:
        _SESSION = _Session()
    return _SESSION
