"""Built-in tools the runtime can execute on the model's behalf.

Deliberately small and local-trust: the server binds to localhost and
acts for the user sitting at the machine — the same trust model as their
browser. Network access is http(s)-only, size-capped and time-capped.

The wire format is OpenAI function-calling; models emit Hermes/Qwen-style
``<tool_call>{"name": ..., "arguments": {...}}</tool_call>`` blocks,
which ``parse_tool_calls`` extracts.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser

_FETCH_LIMIT_BYTES = 512 * 1024
_FETCH_TIMEOUT_S = 20
_RESULT_LIMIT_CHARS = 24_000

BUILTIN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a web page or file over http(s) and return its text "
                "content (HTML is reduced to readable text). Use for any "
                "question about a URL, repository, article or document the "
                "user mentions."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "http(s) URL to fetch"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "current_time",
            "description": "Current date and time (UTC and local).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


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
    """GitHub repo pages render mostly through JS; the raw README is the
    faithful source. Yield better URLs to try first for repo links."""
    m = re.match(r"https?://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$", url)
    if m:
        owner, repo = m.groups()
        for branch in ("main", "master"):
            yield f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/README.md"


def _fetch_url(url: str) -> str:
    import urllib.request

    if not re.match(r"^https?://", url or ""):
        return "error: only http(s) URLs can be fetched"

    candidates = list(_github_raw_candidates(url)) + [url]
    last_err = "unreachable"
    for candidate in candidates:
        try:
            req = urllib.request.Request(
                candidate, headers={"User-Agent": "sparsify/0.1 (+local runtime)"})
            with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT_S) as r:
                ctype = r.headers.get("Content-Type", "")
                raw = r.read(_FETCH_LIMIT_BYTES)
            text = raw.decode("utf-8", errors="replace")
            if "html" in ctype:
                p = _TextExtract()
                p.feed(text)
                text = "\n".join(p.parts)
            text = text[:_RESULT_LIMIT_CHARS]
            note = f"[fetched {candidate} · {len(raw)} bytes]"
            return f"{note}\n{text}"
        except OSError as exc:
            last_err = str(exc)
    return f"error: could not fetch {url}: {last_err}"


def _current_time() -> str:
    now = datetime.now(timezone.utc)
    return (f"UTC: {now:%Y-%m-%d %H:%M:%S} · "
            f"local: {now.astimezone():%Y-%m-%d %H:%M:%S %Z}")


_IMPL = {
    "fetch_url": lambda args: _fetch_url(args.get("url", "")),
    "current_time": lambda args: _current_time(),
}


def execute(name: str, arguments: dict) -> str:
    """Run a built-in tool; errors return as text for the model to see."""
    impl = _IMPL.get(name)
    if impl is None:
        return f"error: unknown tool '{name}'"
    try:
        return impl(arguments or {})
    except Exception as exc:  # tools must never kill the generation loop
        return f"error: {exc}"


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Split generated text into (visible text, [{name, arguments}, ...]).

    Malformed JSON inside a tool_call block is dropped (the model sees no
    result and can retry); the block is still removed from visible text.
    """
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
