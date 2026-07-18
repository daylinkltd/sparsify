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

_AUDIO_LIMIT_BYTES = 32 * 1024**2   # ~17 min of 16 kHz mono 16-bit PCM
_whisper_lock = threading.Lock()


def _multipart_file(body: bytes, content_type: str) -> bytes:
    """Extract the first file part from a multipart/form-data body.
    Minimal by design — the web UI is the only expected caller; OpenAI
    SDK uploads parse identically (one boundary, one file part)."""
    import re as _re
    m = _re.search(r'boundary="?([^";,]+)"?', content_type)
    if not m:
        raise ValueError("multipart body without a boundary")
    boundary = b"--" + m.group(1).encode()
    for part in body.split(boundary):
        head, sep, payload = part.partition(b"\r\n\r\n")
        if sep and b"filename=" in head:
            return payload.rstrip(b"\r\n-")
    raise ValueError("no file part in multipart body")


def _decode_wav_16k_mono(data: bytes):
    """WAV bytes → float32 numpy array for whisper. Only 16 kHz mono
    16-bit PCM is accepted (what the web UI records); anything else gets
    a plain error instead of a silent bad transcription."""
    import io
    import wave

    import numpy as np

    try:
        with wave.open(io.BytesIO(data)) as w:
            if (w.getnchannels(), w.getsampwidth(), w.getframerate()) \
                    != (1, 2, 16000):
                raise ValueError(
                    f"expected 16 kHz mono 16-bit PCM WAV, got "
                    f"{w.getframerate()} Hz {w.getnchannels()}ch "
                    f"{w.getsampwidth() * 8}-bit")
            frames = w.readframes(w.getnframes())
    except wave.Error as exc:
        raise ValueError(f"not a valid WAV file: {exc}") from exc
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0


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
                engine, hf_id = self._get_engine(job.model_tag, job.out)
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
                elif job.tools:
                    # caller-supplied tool schemas: hold back <tool_call>
                    # blocks from the visible stream and hand them to the
                    # HTTP layer as structured calls (OpenAI wire format)
                    from sparsify.runtime import tools as toolbox
                    pieces: list[str] = []
                    emitted = 0
                    for text, tel in engine.chat_stream(
                            job.messages, max_tokens=job.max_tokens,
                            tools=job.tools, temperature=job.temperature):
                        pieces.append(text)
                        whole = "".join(pieces)
                        safe = toolbox.safe_visible_len(whole)
                        # forward even empty deltas: telemetry (token
                        # counts, paging stats) must flow while text is
                        # being held back inside a <tool_call>
                        job.out.put(("chunk", whole[emitted:safe], tel))
                        emitted = safe
                    visible, calls = toolbox.parse_tool_calls("".join(pieces))
                    if len(visible) > emitted:
                        job.out.put(("chunk", visible[emitted:], None))
                    if calls:
                        job.out.put(("calls", calls))
                else:
                    # plain chat: no tools in play, stream verbatim
                    for text, tel in engine.chat_stream(
                            job.messages, max_tokens=job.max_tokens,
                            temperature=job.temperature):
                        job.out.put(("chunk", text, tel))
                job.out.put(("done",))
            except Exception as exc:  # deliver failures to the handler
                job.out.put(("error", exc))

    def _get_engine(self, model_tag: str, out_queue: queue.Queue | None = None):
        from sparsify.runtime.model_registry import resolve_local, resolve_hf_id, KNOWN_ALIASES, register
        from sparsify.runtime.backend import detect
        import os
        import time
        import concurrent.futures
        from huggingface_hub import snapshot_download, HfApi
        import huggingface_hub.utils

        resolved = resolve_local(model_tag)
        if resolved is None:
            # Let's check if we can pull it!
            # It must be a known alias or look like a Hugging Face repo (e.g. contains '/')
            is_valid_repo = "/" in model_tag or model_tag.lower() in KNOWN_ALIASES
            if not is_valid_repo:
                raise FileNotFoundError(
                    f"model '{model_tag}' is not on this machine and is not a known alias — run: sparsify pull {model_tag}"
                )
            
            hf_id = resolve_hf_id(model_tag)
            safe_name = hf_id.replace("/", "--")
            from sparsify.runtime.model_registry import MODELS_DIR
            model_path = MODELS_DIR / safe_name
            
            if out_queue is not None:
                out_queue.put(("chunk", f"☁️ Model '{model_tag}' not found locally. Auto-downloading {hf_id}...\n", None))
            
            # Start download
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            huggingface_hub.utils.disable_progress_bars()
            model_path.mkdir(parents=True, exist_ok=True)
            
            # Fetch remote size
            remote_bytes = 0
            try:
                api = HfApi()
                info = api.repo_info(repo_id=hf_id, files_metadata=True)
                siblings = info.siblings or []
                remote_bytes = sum(getattr(s, "size", 0) or 0 for s in siblings)
            except Exception:
                pass

            def get_download_size(path_obj, repo_id):
                total = 0
                if path_obj.exists():
                    for dirpath, _, filenames in os.walk(path_obj):
                        for f in filenames:
                            fp = os.path.join(dirpath, f)
                            if not os.path.islink(fp):
                                try: total += os.path.getsize(fp)
                                except OSError: pass
                cache_dir = os.path.expanduser(f"~/.cache/huggingface/hub/models--{repo_id.replace('/', '--')}")
                if os.path.exists(cache_dir):
                    for dirpath, _, filenames in os.walk(cache_dir):
                        for f in filenames:
                            if f.endswith(".incomplete"):
                                fp = os.path.join(dirpath, f)
                                try: total += os.path.getsize(fp)
                                except OSError: pass
                return total

            # Download using thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    snapshot_download,
                    repo_id=hf_id, 
                    local_dir=str(model_path)
                )
                
                highest = 0
                last_bytes, last_t, speed = 0, time.monotonic(), 0.0
                last_report_t = 0.0
                
                while not future.done():
                    highest = max(highest, get_download_size(model_path, hf_id))
                    now = time.monotonic()
                    
                    if remote_bytes > 0:
                        done_b = min(highest, remote_bytes)
                        if now - last_t >= 0.5:
                            inst = (done_b - last_bytes) / (now - last_t)
                            speed = inst if speed == 0 else 0.7 * speed + 0.3 * inst
                            last_bytes, last_t = done_b, now
                        
                        if now - last_report_t >= 1.0: # report every 1s
                            pct = (done_b / remote_bytes) * 100
                            done_mb = done_b / 1e6
                            total_mb = remote_bytes / 1e6
                            speed_mb = speed / 1e6
                            if out_queue is not None:
                                out_queue.put(("chunk", f"📥 Download progress: {pct:.1f}% ({done_mb:.1f} / {total_mb:.1f} MB) at {speed_mb:.1f} MB/s...\n", None))
                            last_report_t = now
                    else:
                        if now - last_report_t >= 2.0:
                            if out_queue is not None:
                                out_queue.put(("chunk", f"📥 Downloading {hf_id} (retrieving files)... \n", None))
                            last_report_t = now
                    time.sleep(0.5)
                
                # Retrieve result to propagate any download error
                future.result()
            
            # Register newly downloaded model
            size_bytes = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
            register(hf_id, model_path, size_bytes)
            if out_queue is not None:
                out_queue.put(("chunk", f"✓ Download complete! ({size_bytes / 1e9:.2f} GB on disk). Booting model...\n", None))
            
            resolved = hf_id, model_path

        # Now continue with loading resolved model
        hf_id, model_path = resolved
        if self.loaded_hf_id == hf_id and self.engine is not None:
            return self.engine, hf_id

        if self.engine is not None:
            if self.engine.paging is not None:
                self.engine.paging.close()
            
            # Clear caches based on previous engine type
            if hasattr(self.engine, "backend") and self.engine.backend.name == "pytorch":
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            else:
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except ImportError:
                    pass

            self.engine = None
            self.loaded_hf_id = None

        backend_name = detect().name
        if backend_name == "pytorch":
            from sparsify.runtime.pytorch_chat_generation import PyTorchSparsifyEngine
            self.engine = PyTorchSparsifyEngine(
                model_path, max_tokens=self.max_tokens,
                memory_limit_gb=self.memory_limit_gb,
            )
        else:
            from sparsify.runtime.chat_generation import SparsifyEngine
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
                    # context_limit: the model's architectural max (from its
                    # own config). safe_context_tokens: what *this machine's*
                    # currently free RAM can hold in KV cache right now —
                    # can be 10x+ smaller. Clients (our web UI, OpenClaw,
                    # any agent framework) should size their context/
                    # compaction budget from the safe number, not the
                    # architectural one — using the architectural ceiling
                    # as if it were free capacity is how a context budget
                    # ends up bigger than the RAM that has to hold it.
                    payload["context_limit"] = engine.context_limit
                    payload["safe_context_tokens"] = engine.safe_context_tokens()
                    if engine.paging is not None:
                        payload["stats"] = engine.paging.stats()
                self._json(200, payload)
            elif self.path == "/v1/models":
                if not self._models_ready():
                    return
                from sparsify.runtime.model_registry import all_models
                data = [
                    {"id": m["hf_id"], "object": "model",
                     "owned_by": "sparsify", "size_gb": m["size_gb"],
                     "available": m["available"]}
                    for m in all_models()
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
            if self.path == "/v1/audio/transcriptions":
                self._do_transcribe()
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
            # {"tools": "auto"} (the web UI / CLI use this). Caller-supplied
            # schemas follow OpenAI function calling: the response carries
            # structured "tool_calls" and the caller sends role:"tool"
            # results back. tool_choice "none" disables tools for the turn;
            # other values behave as "auto" (a local model can't be forced).
            tools = body.get("tools")
            auto_tools = tools == "auto" or body.get("auto_tools") is True
            passthrough = tools if isinstance(tools, list) else None
            if body.get("tool_choice") == "none":
                passthrough = None
            from sparsify.runtime import tools as toolbox
            messages = toolbox.normalize_openai_messages(messages)
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

        def _do_transcribe(self):
            """POST /v1/audio/transcriptions — OpenAI-compatible, fully
            local speech-to-text via mlx-whisper. Accepts multipart with a
            16 kHz mono 16-bit PCM WAV under "file" (the web UI records
            exactly that — no ffmpeg dependency, no cloud STT). Optional
            deps and failures are reported plainly, never papered over."""
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            if not 0 < length <= _AUDIO_LIMIT_BYTES:
                self._error(413 if length else 400,
                            f"audio body must be 1..{_AUDIO_LIMIT_BYTES} bytes")
                return
            ctype = self.headers.get("Content-Type", "")
            body = self.rfile.read(length)
            try:
                wav = _multipart_file(body, ctype) if "multipart" in ctype \
                    else body  # raw WAV body also accepted
                audio = _decode_wav_16k_mono(wav)
            except ValueError as exc:
                self._error(400, str(exc))
                return
            try:
                import mlx_whisper
            except ImportError:
                self._error(501, "voice input needs mlx-whisper — install "
                                 "with: pip install mlx-whisper (fully "
                                 "local; ~40 MB model on first use)")
                return
            model = os.environ.get("SPARSIFY_WHISPER_MODEL",
                                   "mlx-community/whisper-tiny")
            try:
                # One transcription at a time: whisper shares the Metal GPU
                # with generation; serialize so neither starves.
                with _whisper_lock:
                    result = mlx_whisper.transcribe(
                        audio, path_or_hf_repo=model)
            except Exception as exc:
                self._error(500, f"transcription failed: {exc}")
                return
            self._json(200, {"text": (result.get("text") or "").strip(),
                             "model": model})

        def _full_completion(self, out, hf_id, rid, created):
            pieces, last_tel, tools_used, calls = [], None, [], None
            while True:
                item = out.get()
                if item[0] == "chunk":
                    pieces.append(item[1])
                    if item[2] is not None:
                        last_tel = item[2]
                elif item[0] == "tool":
                    tools_used.append(item[1])
                elif item[0] == "calls":
                    calls = item[1]
                elif item[0] == "done":
                    break
                else:  # error mid-generation
                    self._error(500, str(item[1]))
                    return
            message = {"role": "assistant", "content": "".join(pieces)}
            finish = (last_tel or {}).get("finish_reason") or "stop"
            if calls:
                from sparsify.runtime import tools as toolbox
                message["content"] = message["content"] or None
                message["tool_calls"] = toolbox.openai_tool_calls(calls)
                finish = "tool_calls"
            resp = {
                "id": rid, "object": "chat.completion", "created": created,
                "model": hf_id,
                "choices": [{"index": 0, "finish_reason": finish,
                             "message": message}],
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
            finish_override = None
            while True:
                item = out.get()
                if item[0] == "chunk":
                    if item[2] is not None:
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
                elif item[0] == "calls":
                    # OpenAI streaming tool calls: one delta carrying the
                    # complete calls (id + name + full arguments), then
                    # finish_reason "tool_calls" — spec-legal, and what
                    # OpenAI-SDK clients accumulate on.
                    from sparsify.runtime import tools as toolbox
                    chunk({"tool_calls": [
                        {"index": i, **tc} for i, tc in
                        enumerate(toolbox.openai_tool_calls(item[1]))]})
                    finish_override = "tool_calls"
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
            chunk({}, finish=finish_override
                  or (last_tel or {}).get("finish_reason") or "stop",
                  extra=extra)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def log_message(self, *_):
            pass

    log(f"sparsify API listening on http://localhost:{port}  "
        f"(loaded: {host.loaded_hf_id or 'none — models load on first request'})")
    ThreadingHTTPServer(("localhost", port), Handler).serve_forever()
