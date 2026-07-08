"""Sparsify API server — OpenAI-compatible, Ollama-style.

Runs on localhost:7777 by default. Starts with no model loaded; the first
chat request naming a model loads it (paged), and it stays warm for
subsequent requests. Requesting a different model swaps engines — one
model is resident at a time, because the whole point is bounded RAM.

All inference runs on one dedicated worker thread: MLX GPU streams are
bound to the thread that created them, and a single Metal GPU serializes
generation anyway. HTTP handler threads talk to the worker via queues.

Endpoints:
    GET  /health               -> {"status": "ok", "loaded": <hf id | null>}
    GET  /v1/models            -> models available on this machine
    POST /v1/chat/completions  -> {"model", "messages", "stream"?, "max_tokens"?}
                                  streaming uses SSE chunks like OpenAI's API;
                                  non-stream responses carry measured Sparsify
                                  telemetry under "sparsify".
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_PORT = 7777


@dataclass
class _Job:
    model_tag: str
    messages: list
    max_tokens: int | None
    tools: list | None = None       # explicit OpenAI tool schemas (passthrough)
    auto_tools: bool = False        # run built-in tools server-side (agent loop)
    temperature: float = 0.0
    out: "queue.Queue" = field(default_factory=queue.Queue)


class EngineHost:
    """Single inference worker thread owning the resident engine."""

    def __init__(self, memory_limit_gb: float | None, max_tokens: int,
                 policy=None) -> None:
        self.memory_limit_gb = memory_limit_gb
        self.max_tokens = max_tokens
        self.policy = policy  # operator-set tool permissions (or None = read-only)
        self.engine = None
        self.loaded_hf_id: str | None = None
        self._jobs: "queue.Queue[_Job]" = queue.Queue()
        threading.Thread(target=self._worker, daemon=True,
                         name="sparsify-inference").start()

    # -- worker thread (owns all MLX state) -----------------------------
    def _worker(self) -> None:
        while True:
            job = self._jobs.get()
            try:
                engine, hf_id = self._get_engine(job.model_tag)
                job.out.put(("meta", hf_id))
                if job.auto_tools:
                    # server executes built-in tools between rounds
                    for kind, payload, tel in engine.agent_stream(
                            job.messages, max_tokens=job.max_tokens,
                            policy=self.policy, temperature=job.temperature):
                        if kind == "text":
                            job.out.put(("chunk", payload, tel))
                        else:  # tool ran
                            job.out.put(("tool", payload))
                else:
                    # passthrough: caller supplied its own tools (or none)
                    for text, tel in engine.chat_stream(
                            job.messages, max_tokens=job.max_tokens,
                            tools=job.tools, temperature=job.temperature):
                        job.out.put(("chunk", text, tel))
                job.out.put(("done",))
            except Exception as exc:  # deliver failures to the handler
                job.out.put(("error", exc))

    def _get_engine(self, model_tag: str):
        from sparsify.runtime.model_registry import resolve_local
        from sparsify.runtime.chat_generation import SparsifyEngine

        resolved = resolve_local(model_tag)
        if resolved is None:
            raise FileNotFoundError(
                f"model '{model_tag}' is not on this machine — run: sparsify pull {model_tag}"
            )
        hf_id, model_path = resolved
        if self.loaded_hf_id == hf_id and self.engine is not None:
            return self.engine, hf_id

        if self.engine is not None:
            import mlx.core as mx
            if self.engine.paging is not None:
                self.engine.paging.close()
            self.engine = None
            self.loaded_hf_id = None
            mx.clear_cache()

        self.engine = SparsifyEngine(
            model_path, max_tokens=self.max_tokens,
            memory_limit_gb=self.memory_limit_gb,
        )
        self.loaded_hf_id = hf_id
        return self.engine, hf_id

    # -- called from HTTP handler threads --------------------------------
    def submit(self, model_tag: str, messages: list, max_tokens: int | None,
               tools: list | None = None, auto_tools: bool = False,
               temperature: float = 0.0) -> "queue.Queue":
        job = _Job(model_tag, messages, max_tokens, tools, auto_tools, temperature)
        self._jobs.put(job)
        return job.out

    def warm(self, model_tag: str) -> None:
        """Load a model eagerly (blocks until loaded or failed)."""
        out = self.submit(model_tag, [{"role": "user", "content": "hello"}], max_tokens=1)
        while True:
            item = out.get()
            if item[0] == "error":
                raise item[1]
            if item[0] == "done":
                return


def _models_dir_accessible(timeout: float = 5.0) -> bool:
    """Probe the models directory in a side thread with a hard timeout.

    macOS blocks background (launchd) processes from removable volumes by
    *hanging* the file operation on a consent prompt that never renders.
    A probe that doesn't return within the timeout means blocked — the
    server then answers 503 with a remedy instead of hanging requests.
    """
    from sparsify.runtime.model_registry import MODELS_DIR

    result: dict = {}

    def probe() -> None:
        try:
            if MODELS_DIR.exists():
                list(MODELS_DIR.iterdir())
            result["ok"] = True
        except OSError:
            result["ok"] = False

    t = threading.Thread(target=probe, daemon=True)
    t.start()
    t.join(timeout)
    return result.get("ok", False)


_BLOCKED_MSG = (
    "the models directory is not readable from a background service — macOS "
    "blocks external volumes for login services. Either grant Full Disk "
    "Access to the sparsify python binary (System Settings > Privacy & "
    "Security), move models to an internal path (default ~/.sparsify/models), "
    "or run 'sparsify serve' from a terminal instead."
)


def serve(port: int = DEFAULT_PORT, model: str | None = None,
          memory_limit_gb: float | None = None, max_tokens: int = 0,
          policy=None, log=print) -> None:
    """Blocking server loop. ``model``, when given, is loaded eagerly.
    ``policy`` (tools.ToolPolicy) is the operator's tool grant — agent
    tools are a startup decision, never something a request grants itself."""
    host = EngineHost(memory_limit_gb, max_tokens, policy=policy)
    if policy is not None:
        tiers = ",".join(sorted(policy.enabled_tiers()))
        log(f"tools enabled: {tiers} · workspace {policy.workspace}")
    access = {"ok": _models_dir_accessible()}
    if not access["ok"]:
        log(f"WARNING: {_BLOCKED_MSG}")
    if model and access["ok"]:
        log(f"loading {model} …")
        host.warm(model)
        log(f"loaded {host.loaded_hf_id}")

    class Handler(BaseHTTPRequestHandler):
        server_version = "sparsify"

        # -- helpers ----------------------------------------------------
        def _json(self, status: int, obj: dict) -> None:
            payload = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _error(self, status: int, message: str) -> None:
            self._json(status, {"error": {"message": message, "type": "sparsify_error"}})

        # -- routes -----------------------------------------------------
        def _models_ready(self) -> bool:
            if not access["ok"]:
                access["ok"] = _models_dir_accessible()  # user may have granted
            if not access["ok"]:
                self._error(503, _BLOCKED_MSG)
                return False
            return True

        def do_GET(self):
            if self.path == "/":
                from sparsify.runtime.webui import PAGE
                payload = PAGE.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
            elif self.path == "/health":
                payload = {"status": "ok", "loaded": host.loaded_hf_id,
                           "models_dir_accessible": access["ok"],
                           "port": port, "runtime": "sparsify"}
                engine = host.engine
                if engine is not None:
                    payload["supports_tools"] = engine.supports_tools()
                    if engine.paging is not None:
                        payload["stats"] = engine.paging.stats()
                self._json(200, payload)
            elif self.path == "/v1/models":
                if not self._models_ready():
                    return
                from sparsify.runtime.model_registry import all_models
                data = [
                    {"id": m["hf_id"], "object": "model",
                     "owned_by": "sparsify", "size_gb": m["size_gb"]}
                    for m in all_models() if m["available"]
                ]
                self._json(200, {"object": "list", "data": data})
            elif self.path == "/v1/tools":
                from sparsify.runtime import tools as toolbox
                pol = host.policy or toolbox.ToolPolicy.read_only()
                self._json(200, {"object": "list",
                                 "data": toolbox.tools_for_policy(pol),
                                 "tiers_enabled": sorted(pol.enabled_tiers()),
                                 "workspace": str(pol.workspace)})
            elif self.path == "/version":
                from sparsify.runtime import updater
                from sparsify import __version__ as _v
                self._json(200, {"version": _v, **updater.check()})
            else:
                self._error(404, f"no route {self.path}")

        def do_POST(self):
            if self.path == "/admin/update":
                self._do_update()
                return
            if self.path != "/v1/chat/completions":
                self._error(404, f"no route {self.path}")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._error(400, "request body is not valid JSON")
                return

            messages = body.get("messages") or []
            if not messages:
                self._error(400, "'messages' is required")
                return
            model_tag = body.get("model") or host.loaded_hf_id
            if not model_tag:
                self._error(400, "'model' is required (no model loaded yet); "
                                 "see GET /v1/models for what's available")
                return
            if not self._models_ready():
                return

            # Tools: pass through caller-supplied schemas, OR run the
            # built-in tools server-side when the caller opts in with
            # {"tools": "auto"} (the web UI / CLI use this).
            tools = body.get("tools")
            auto_tools = tools == "auto" or body.get("auto_tools") is True
            passthrough = tools if isinstance(tools, list) else None
            try:
                temperature = float(body.get("temperature") or 0.0)
            except (TypeError, ValueError):
                temperature = 0.0
            out = host.submit(model_tag, messages, body.get("max_tokens"),
                              tools=passthrough, auto_tools=auto_tools,
                              temperature=temperature)
            first = out.get()
            if first[0] == "error":
                exc = first[1]
                status = 404 if isinstance(exc, FileNotFoundError) else 500
                self._error(status, str(exc))
                return
            hf_id = first[1]

            rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
            created = int(time.time())
            try:
                if body.get("stream"):
                    self._stream_completion(out, hf_id, rid, created)
                else:
                    self._full_completion(out, hf_id, rid, created)
            except (BrokenPipeError, ConnectionResetError):
                # client went away; drain so the worker isn't blocked
                self._drain(out)

        @staticmethod
        def _drain(out: "queue.Queue") -> None:
            while True:
                item = out.get()
                if item[0] in ("done", "error"):
                    return

        def _do_update(self):
            """Run 'sparsify update' detached; the login service (KeepAlive)
            restarts itself on the new version. Localhost-only, fixed command
            (no shell, no injection)."""
            import shutil
            import subprocess as _sp

            from sparsify.runtime import updater
            st = updater.check(force=True)
            if not st.get("update_available"):
                self._json(200, {"status": "up-to-date", **st})
                return
            home = os.environ.get("SPARSIFY_HOME", str(Path.home() / ".sparsify"))
            binary = str(Path(home) / "venv" / "bin" / "sparsify")
            if not Path(binary).exists():
                binary = shutil.which("sparsify") or "sparsify"
            try:
                _sp.Popen([binary, "update"], start_new_session=True,
                          stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
            except OSError as exc:
                self._error(500, f"could not start updater: {exc}")
                return
            self._json(202, {"status": "updating", **st,
                             "note": "the server restarts on the new version; "
                                     "reconnect in a few seconds"})

        def _full_completion(self, out, hf_id, rid, created):
            pieces, last_tel, tools_used = [], None, []
            while True:
                item = out.get()
                if item[0] == "chunk":
                    pieces.append(item[1])
                    last_tel = item[2]
                elif item[0] == "tool":
                    tools_used.append(item[1])
                elif item[0] == "done":
                    break
                else:  # error mid-generation
                    self._error(500, str(item[1]))
                    return
            resp = {
                "id": rid, "object": "chat.completion", "created": created,
                "model": hf_id,
                "choices": [{"index": 0,
                             "finish_reason": (last_tel or {}).get("finish_reason") or "stop",
                             "message": {"role": "assistant",
                                         "content": "".join(pieces)}}],
                "usage": {"completion_tokens": last_tel["n_tokens"] if last_tel else 0},
            }
            if tools_used:
                resp["sparsify_tools"] = tools_used
            if last_tel:
                resp["sparsify"] = {k: last_tel[k] for k in
                                    ("throughput", "active_gb", "peak_gb", "rss_gb")
                                    if k in last_tel}
                if "paging" in last_tel:
                    resp["sparsify"]["paging"] = last_tel["paging"]
            self._json(200, resp)

        def _stream_completion(self, out, hf_id, rid, created):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def chunk(delta: dict, finish=None, extra: dict | None = None):
                data = {"id": rid, "object": "chat.completion.chunk",
                        "created": created, "model": hf_id,
                        "choices": [{"index": 0, "delta": delta,
                                     "finish_reason": finish}]}
                if extra:
                    data.update(extra)
                self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()

            chunk({"role": "assistant", "content": ""})
            last_tel = None
            while True:
                item = out.get()
                if item[0] == "chunk":
                    last_tel = item[2]
                    if item[1]:
                        extra = None
                        if last_tel:
                            sparsify = {k: last_tel[k] for k in
                                        ("throughput", "active_gb", "peak_gb", "rss_gb")
                                        if k in last_tel}
                            if "paging" in last_tel:
                                sparsify["paging"] = last_tel["paging"]
                            extra = {"sparsify": sparsify}
                        chunk({"content": item[1]}, extra=extra)
                elif item[0] == "tool":
                    # surface each tool call as its own delta the UI can render
                    chunk({}, extra={"sparsify_tool": item[1]})
                elif item[0] == "done":
                    break
                else:
                    break  # error mid-stream: close with a clean stop below
            extra = None
            if last_tel:
                sparsify = {k: last_tel[k] for k in
                            ("throughput", "active_gb", "peak_gb", "rss_gb")
                            if k in last_tel}
                if "paging" in last_tel:
                    sparsify["paging"] = last_tel["paging"]
                extra = {"sparsify": sparsify}
            chunk({}, finish=(last_tel or {}).get("finish_reason") or "stop",
                  extra=extra)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *_):
            pass

    log(f"sparsify API listening on http://localhost:{port}  "
        f"(loaded: {host.loaded_hf_id or 'none — models load on first request'})")
    ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
