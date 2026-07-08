"""Tools the runtime can execute on the model's behalf — an agent
substrate like OpenHands / Claude Code, scaled to a *local, user-owned*
runtime.

Trust model, stated plainly. The server binds to localhost and acts for
the person at the machine, but the models driving it are small (Qwen3,
not Claude) and *will* misfire. So capability is tiered and opt-in:

  READ tier  (default): fetch_url, web_search, read_file, list_dir,
             current_time — cannot modify anything.
  WRITE tier (opt-in):  write_file — confined to a workspace directory.
  SHELL tier (opt-in):  run_shell — runs with YOUR privileges; NOT a
             sandbox. Enabling it means the model can do anything you can
             in a terminal. This is deliberate and documented, not hidden.

File tools are confined to a ``workspace`` root by real path resolution
(``..``, symlink, and absolute-path escapes are rejected). Shell is not
confinable in pure Python and is therefore all-or-nothing power, off
unless explicitly enabled.

Wire format is OpenAI function-calling; models emit Hermes/Qwen-style
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks that
``parse_tool_calls`` extracts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

_FETCH_LIMIT_BYTES = 512 * 1024
_FETCH_TIMEOUT_S = 20
_RESULT_LIMIT_CHARS = 24_000
_FILE_READ_LIMIT = 256 * 1024
_SHELL_TIMEOUT_S = 60
_SHELL_OUTPUT_LIMIT = 16_000

READ, WRITE, SHELL, BROWSER = "read", "write", "shell", "browser"


@dataclass
class ToolPolicy:
    """What the model is allowed to do this session."""

    workspace: Path
    allow_write: bool = False
    allow_shell: bool = False
    allow_browser: bool = False

    @classmethod
    def read_only(cls, workspace: Path | None = None) -> "ToolPolicy":
        return cls(workspace=_default_workspace(workspace))

    @classmethod
    def from_flags(cls, agent: bool, workspace: Path | None = None,
                   allow_shell: bool | None = None,
                   allow_browser: bool | None = None) -> "ToolPolicy":
        """`agent=True` enables write (+ shell unless disabled); browser is
        enabled when asked and the optional engine is installed."""
        from sparsify.runtime import browser as _b
        want_browser = agent if allow_browser is None else allow_browser
        return cls(
            workspace=_default_workspace(workspace),
            allow_write=agent,
            allow_shell=agent if allow_shell is None else allow_shell,
            allow_browser=bool(want_browser) and _b.available(),
        )

    def enabled_tiers(self) -> set:
        tiers = {READ}
        if self.allow_write:
            tiers.add(WRITE)
        if self.allow_shell:
            tiers.add(SHELL)
        if self.allow_browser:
            tiers.add(BROWSER)
        return tiers


def _default_workspace(ws: Path | None) -> Path:
    ws = Path(ws).expanduser() if ws else Path.home() / ".sparsify" / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws


# ── tool registry: (schema, tier, impl) ─────────────────────────────────

_REGISTRY: dict[str, tuple[dict, str, object]] = {}


def _tool(name: str, tier: str, description: str, params: dict):
    def deco(fn):
        _REGISTRY[name] = (
            {"type": "function", "function": {
                "name": name, "description": description,
                "parameters": {"type": "object", "properties": params,
                               "required": [k for k, v in params.items()
                                            if v.pop("_required", False)]}}},
            tier, fn)
        return fn
    return deco


def tools_for_policy(policy: ToolPolicy) -> list[dict]:
    """Schemas the model should see, given what the policy permits."""
    tiers = policy.enabled_tiers()
    return [schema for schema, tier, _ in _REGISTRY.values() if tier in tiers]


# Backwards-compatible read-only default (server /v1/tools, legacy callers).
BUILTIN_TOOLS: list[dict] = []  # filled after registrations below


# ── workspace confinement ───────────────────────────────────────────────

def _resolve_in_workspace(policy: ToolPolicy, path: str) -> Path:
    """Resolve *path* and guarantee it stays inside the workspace.

    Rejects ``..`` traversal, symlink escapes (including a *dangling*
    symlink as the final component — its target is followed and checked
    even though it doesn't exist yet) and absolute paths outside the root.

    The containment check runs on ``os.path.realpath`` of the full path,
    which dereferences every symlink and collapses ``..`` regardless of
    whether the leaf exists — so the checked path and the returned path
    are always the same real path (no parent-only fallback to slip past)."""
    root = policy.workspace.resolve()
    raw = Path(path).expanduser()
    candidate = raw if raw.is_absolute() else policy.workspace / raw
    resolved = Path(os.path.realpath(candidate))
    if resolved != root and root not in resolved.parents:
        raise PermissionError(
            f"path '{path}' is outside the workspace ({policy.workspace}). "
            f"Point the workspace elsewhere with --workspace to allow it.")
    return resolved


# ── READ tier ────────────────────────────────────────────────────────────

class _TextExtract(HTMLParser):
    _SKIP = {"script", "style", "noscript", "svg", "head"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._SKIP:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data):
        if not self._skip_depth and data.strip():
            self.parts.append(data.strip())


def _github_raw_candidates(url: str):
    m = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url)
    if m:
        owner, repo = m.groups()
        for branch in ("main", "master"):
            yield f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"


def _host_blocked(host: str) -> bool:
    """True if *host* resolves to any address the runtime must not reach:
    loopback, private (RFC1918/ULA), link-local (incl. 169.254 cloud
    metadata), reserved, multicast, unspecified. Blocks SSRF into the
    local network from a model-supplied URL. Opt out for local-dev with
    SPARSIFY_ALLOW_LOCAL_FETCH=1."""
    import ipaddress
    import socket

    if os.environ.get("SPARSIFY_ALLOW_LOCAL_FETCH", "").lower() in ("1", "true", "yes"):
        return False
    try:
        infos = socket.getaddrinfo(host, None)
    except (socket.gaierror, UnicodeError):
        return False  # let urlopen fail naturally with a clean network error
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            continue
        if (ip.is_private or ip.is_loopback or ip.is_link_local
                or ip.is_reserved or ip.is_multicast or ip.is_unspecified):
            return True
    return False


def _guarded_opener():
    """urllib opener that re-checks the destination on every redirect hop,
    so a public URL can't 302 into the internal network."""
    import urllib.error
    import urllib.parse
    import urllib.request

    class _Redirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
            host = urllib.parse.urlparse(newurl).hostname or ""
            if _host_blocked(host):
                raise urllib.error.HTTPError(
                    newurl, code, "blocked redirect to an internal address",
                    headers, fp)
            return super().redirect_request(req, fp, code, msg, headers, newurl)

    return urllib.request.build_opener(_Redirect)


@_tool("fetch_url", READ,
       "Fetch a web page or file over http(s) and return its text (HTML "
       "reduced to readable text). Use for any URL, repo, article or doc.",
       {"url": {"type": "string", "description": "http(s) URL", "_required": True}})
def _fetch_url(policy, args) -> str:
    import urllib.parse
    import urllib.request

    url = args.get("url", "")
    if not re.match(r"^https?://", url):
        return "error: only http(s) URLs can be fetched"
    opener = _guarded_opener()
    for candidate in list(_github_raw_candidates(url)) + [url]:
        host = urllib.parse.urlparse(candidate).hostname or ""
        if _host_blocked(host):
            return (f"error: refusing to fetch {candidate} — it resolves to a "
                    f"private/loopback address (SSRF guard). Set "
                    f"SPARSIFY_ALLOW_LOCAL_FETCH=1 to allow local addresses.")
        try:
            req = urllib.request.Request(
                candidate, headers={"User-Agent": "sparsify/0.1 (+local runtime)"})
            with opener.open(req, timeout=_FETCH_TIMEOUT_S) as r:
                ctype = r.headers.get("Content-Type", "")
                raw = r.read(_FETCH_LIMIT_BYTES)
            text = raw.decode("utf-8", errors="replace")
            if "html" in ctype:
                p = _TextExtract()
                p.feed(text)
                text = "\n".join(p.parts)
            return f"[fetched {candidate} · {len(raw)} bytes]\n{text[:_RESULT_LIMIT_CHARS]}"
        except OSError as exc:
            last = str(exc)
    return f"error: could not fetch {url}: {last}"


@_tool("web_search", READ,
       "Search the web and return the top results (title, URL, snippet). "
       "Use to find pages when you don't already have a URL.",
       {"query": {"type": "string", "description": "search query", "_required": True}})
def _web_search(policy, args) -> str:
    import urllib.parse
    import urllib.request

    q = (args.get("query") or "").strip()
    if not q:
        return "error: empty query"
    url = "https://html.duckduckgo.com/html/?q=" + urllib.parse.quote(q)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as r:
            html = r.read(_FETCH_LIMIT_BYTES).decode("utf-8", errors="replace")
    except OSError as exc:
        return f"error: search failed: {exc}"

    results = []
    for m in re.finditer(
            r'result__a[^>]*href="([^"]+)"[^>]*>(.*?)</a>.*?'
            r'result__snippet[^>]*>(.*?)</a>', html, re.S):
        href, title, snippet = m.groups()
        href = urllib.parse.unquote(re.sub(r"^.*?uddg=", "", href).split("&")[0]) \
            if "uddg=" in href else href
        title = re.sub(r"<[^>]+>", "", title).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet).strip()
        results.append(f"- {title}\n  {href}\n  {snippet}")
        if len(results) >= 6:
            break
    return "\n".join(results) if results else f"no results for '{q}'"


@_tool("read_file", READ,
       "Read a UTF-8 text file from the workspace and return its contents.",
       {"path": {"type": "string", "description": "path within the workspace",
                 "_required": True}})
def _read_file(policy, args) -> str:
    p = _resolve_in_workspace(policy, args.get("path", ""))
    if not p.is_file():
        return f"error: no such file: {args.get('path')}"
    data = p.read_bytes()[:_FILE_READ_LIMIT]
    text = data.decode("utf-8", errors="replace")
    trunc = "\n…[truncated]" if p.stat().st_size > _FILE_READ_LIMIT else ""
    return f"[{p}]\n{text}{trunc}"


@_tool("list_dir", READ,
       "List files and folders in a workspace directory.",
       {"path": {"type": "string", "description": "directory (default: workspace root)"}})
def _list_dir(policy, args) -> str:
    p = _resolve_in_workspace(policy, args.get("path", ".") or ".")
    if not p.is_dir():
        return f"error: not a directory: {args.get('path')}"
    rows = []
    for entry in sorted(p.iterdir())[:500]:
        kind = "dir " if entry.is_dir() else "file"
        size = entry.stat().st_size if entry.is_file() else 0
        rows.append(f"{kind}  {size:>10}  {entry.name}")
    return f"[{p}]\n" + ("\n".join(rows) if rows else "(empty)")


@_tool("current_time", READ, "Current date and time (UTC and local).", {})
def _current_time(policy, args) -> str:
    now = datetime.now(timezone.utc)
    return (f"UTC: {now:%Y-%m-%d %H:%M:%S} · "
            f"local: {now.astimezone():%Y-%m-%d %H:%M:%S %Z}")


# ── WRITE tier ────────────────────────────────────────────────────────────

@_tool("write_file", WRITE,
       "Create or overwrite a UTF-8 text file in the workspace.",
       {"path": {"type": "string", "description": "path within the workspace",
                 "_required": True},
        "content": {"type": "string", "description": "file contents",
                    "_required": True}})
def _write_file(policy, args) -> str:
    p = _resolve_in_workspace(policy, args.get("path", ""))
    p.parent.mkdir(parents=True, exist_ok=True)
    content = args.get("content", "")
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} chars to {p}"


# ── SHELL tier ────────────────────────────────────────────────────────────

@_tool("run_shell", SHELL,
       "Run a shell command in the workspace and return its output. Runs "
       "with the user's own privileges (not sandboxed).",
       {"command": {"type": "string", "description": "shell command",
                    "_required": True}})
def _run_shell(policy, args) -> str:
    cmd = args.get("command", "")
    if not cmd.strip():
        return "error: empty command"
    try:
        proc = subprocess.run(
            cmd, shell=True, cwd=str(policy.workspace),
            capture_output=True, text=True, timeout=_SHELL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {_SHELL_TIMEOUT_S}s"
    out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
    out = out[:_SHELL_OUTPUT_LIMIT]
    return f"[exit {proc.returncode}]\n{out}" if out.strip() else f"[exit {proc.returncode}] (no output)"


# ── BROWSER tier ──────────────────────────────────────────────────────────

@_tool("browser_open", BROWSER,
       "Open a URL in a real browser (persistent login session) and return "
       "the page text plus a numbered list of clickable/typeable elements.",
       {"url": {"type": "string", "description": "page URL", "_required": True}})
def _browser_open(policy, args) -> str:
    from sparsify.runtime import browser
    return browser.session().open(args.get("url", ""))


@_tool("browser_read", BROWSER,
       "Re-read the current browser page: text + numbered interactive elements.",
       {})
def _browser_read(policy, args) -> str:
    from sparsify.runtime import browser
    return browser.session().read()


@_tool("browser_click", BROWSER,
       "Click an element by its number from the last browser read.",
       {"index": {"type": "integer", "description": "element number", "_required": True}})
def _browser_click(policy, args) -> str:
    from sparsify.runtime import browser
    return browser.session().click(args.get("index"))


@_tool("browser_type", BROWSER,
       "Type text into an input by its number; set submit=true to press Enter.",
       {"index": {"type": "integer", "description": "element number", "_required": True},
        "text": {"type": "string", "description": "text to type", "_required": True},
        "submit": {"type": "boolean", "description": "press Enter after typing"}})
def _browser_type(policy, args) -> str:
    from sparsify.runtime import browser
    return browser.session().type(args.get("index"), args.get("text", ""),
                                  bool(args.get("submit")))


@_tool("browser_screenshot", BROWSER,
       "Save a screenshot of the current browser page to the workspace.",
       {})
def _browser_screenshot(policy, args) -> str:
    from sparsify.runtime import browser
    dest = policy.workspace / "screenshot.png"
    return f"saved {browser.session().screenshot(dest)}"


BUILTIN_TOOLS = tools_for_policy(ToolPolicy.read_only())


# ── execution + parsing ─────────────────────────────────────────────────

def execute(name: str, arguments: dict, policy: ToolPolicy | None = None) -> str:
    """Run a tool, enforcing the policy at execution time (defense in depth:
    even if a disallowed schema leaked into the prompt, it won't run)."""
    policy = policy or ToolPolicy.read_only()
    entry = _REGISTRY.get(name)
    if entry is None:
        return f"error: unknown tool '{name}'"
    _schema, tier, impl = entry
    if tier not in policy.enabled_tiers():
        return (f"error: tool '{name}' is not enabled. Start with --agent "
                f"(and, for shell, accept the risk) to allow it.")
    try:
        return impl(policy, arguments or {})
    except PermissionError as exc:
        return f"error: {exc}"
    except Exception as exc:  # a tool must never kill the generation loop
        return f"error: {exc}"


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split generated text into (visible text, [{name, arguments}, ...]).
    Malformed JSON in a block is dropped; the block leaves visible text."""
    calls = []
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict) and obj.get("name"):
                calls.append({"name": obj["name"],
                              "arguments": obj.get("arguments") or {}})
        except json.JSONDecodeError:
            continue
    visible = _TOOL_CALL_RE.sub("", text).strip()
    return visible, calls
