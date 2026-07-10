"""Sparsify inference engine.

Loads a model lazily, replaces its expert projections with SSD-paged
modules (sparsify.paging), and streams tokens with measured telemetry.

Memory model:
  - backbone (attention, norms, embeddings, router gates, shared experts)
    is materialized in RAM at startup;
  - expert weights stay on SSD and page in through a byte-budgeted LRU
    cache (``memory_limit_gb`` is that cache's budget);
  - dense models have no pageable experts and load fully, unmodified.

Every telemetry value is measured from the running process — never
estimated or simulated.
"""
from __future__ import annotations

import time
from pathlib import Path

import mlx.core as mx

from sparsify.paging import attach_paging

# Metal buffer-cache cap: bounds reuse-pool growth from transient
# per-token expert stacks without forcing reallocation every token.
_METAL_CACHE_LIMIT_BYTES = 2 * 1024**3


class SparsifyEngine:
    """Streaming chat engine over a (possibly storage-backed) MLX model."""

    def __init__(self, model_path: Path, max_tokens: int = 0,
                 memory_limit_gb: float | None = None) -> None:
        """``memory_limit_gb`` is the expert-cache budget. None (default)
        sizes it automatically from measured free system RAM at startup.
        ``max_tokens``: 0 (default) = unlimited — generate until the model
        finishes or its context window fills."""
        import json as _json
        import mlx_lm  # deferred: slow import

        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        self._mlx_lm = mlx_lm

        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json not found in {self.model_path}. "
                f"Run: sparsify pull <model>"
            )

        # The model's own context window is the only hard generation
        # ceiling; used when max_tokens is unlimited.
        cfg: dict = {}
        try:
            with open(self.model_path / "config.json") as f:
                cfg = _json.load(f)
            self.context_limit = int(
                cfg.get("max_position_embeddings")
                or cfg.get("model_max_length") or 32768)
        except (OSError, ValueError):
            self.context_limit = 32768

        # KV cache bytes/token — fp16, 2x for K+V, summed over layers.
        # Lets callers (our own API, external agent frameworks like
        # OpenClaw) size a context window their RAM can actually hold,
        # instead of hardcoding a guess. architectural_limit can be far
        # larger than what fits: at 262144 tokens this model alone would
        # need ~26 GB of KV cache, which no 16 GB machine has.
        kv_heads = cfg.get("num_key_value_heads") or cfg.get("num_attention_heads")
        head_dim = cfg.get("head_dim") or (
            cfg.get("hidden_size", 0) // cfg.get("num_attention_heads", 1)
            if cfg.get("num_attention_heads") else None)
        n_layers = cfg.get("num_hidden_layers")
        self.kv_bytes_per_token: int | None = (
            2 * kv_heads * head_dim * n_layers * 2
            if kv_heads and head_dim and n_layers else None)

        mx.set_cache_limit(_METAL_CACHE_LIMIT_BYTES)

        # Lazy load: no weight bytes are read yet, so expert tensors can be
        # dropped by attach_paging without ever touching RAM.
        self.model, self.tokenizer = mlx_lm.load(str(self.model_path), lazy=True)

        # Budget is finalized after the backbone is resident (see below);
        # nothing pages until the first generate call, so 0 is safe here.
        self.paging = attach_paging(self.model, self.model_path, budget_bytes=0)

        # Materialize whatever is left in the parameter tree. With paging
        # attached that is the backbone only; for dense models, everything.
        mx.eval(self.model.parameters())
        self.model_memory_gb: float = mx.get_active_memory() / 1e9
        mx.reset_peak_memory()

        if memory_limit_gb is not None:
            budget_bytes = int(memory_limit_gb * 1024**3)
        else:
            budget_bytes = self._auto_budget_bytes()
        if self.paging is not None:
            # Hybrid residency: blocks that fit the budget load fully and
            # run at native speed; the rest page per expert from SSD.
            self.paging.configure(budget_bytes)
        self.memory_limit_gb: float = budget_bytes / 1024**3

        self.messages: list[dict] = []

        # Persistent KV cache: token ids whose keys/values are already
        # computed. Each turn only the unseen suffix is prefilled — on a
        # paged model that avoids re-reading experts for the whole history.
        self._prompt_cache = None
        self._cached_tokens: list[int] = []

    @staticmethod
    def _auto_budget_bytes() -> int:
        """Expert-cache budget from *measured* free RAM: half of what the OS
        reports available right now (backbone is already resident at this
        point), floor 1 GiB. Half — not all — leaves room for activations,
        KV cache, the Metal buffer pool, and other processes."""
        floor = 1024**3
        try:
            import psutil
            available = psutil.virtual_memory().available
        except ImportError:
            return 4 * floor  # conservative fixed default without psutil
        return max(floor, int(available * 0.5))

    def safe_context_tokens(self) -> int:
        """Largest context this machine can hold *right now* without the
        KV cache alone exhausting RAM — measured free memory, not the
        model's architectural ceiling. The two can differ by 10x+: this
        model supports 262144 tokens architecturally, but that needs
        ~26 GB of KV cache no 16 GB Mac has. Callers (our own API,
        external agent frameworks like OpenClaw) should size their
        context budget from this, not from context_limit alone — using
        context_limit as if it were free capacity is exactly how an
        agent's own context-compaction settings end up undersized against
        reality, or oversized into a machine that can't hold it."""
        if not self.kv_bytes_per_token:
            return self.context_limit  # can't estimate; don't under-claim
        try:
            import psutil
            available = psutil.virtual_memory().available
        except ImportError:
            available = 4 * 1024**3
        # Reserve half of current free RAM for KV cache; the rest stays
        # free for the expert cache, activations, and everything else —
        # same 50% split _auto_budget_bytes uses for the expert cache.
        budget_bytes = available * 0.5
        return min(self.context_limit, int(budget_bytes / self.kv_bytes_per_token))

    # ------------------------------------------------------------------

    def supports_tools(self) -> bool:
        """True only if the model's chat template actually renders tool
        schemas. Small/older templates silently ignore the ``tools`` kwarg
        — detected by probing whether a known tool name reaches the prompt,
        so we can tell the user instead of hallucinating an answer."""
        if getattr(self, "_supports_tools", None) is not None:
            return self._supports_tools
        ok = False
        if getattr(self.tokenizer, "chat_template", None):
            probe = [{"type": "function", "function": {
                "name": "sparsify_probe_tool", "description": "probe",
                "parameters": {"type": "object", "properties": {}}}}]
            try:
                rendered = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": "hi"}], tools=probe,
                    add_generation_prompt=True, tokenize=False)
                ok = "sparsify_probe_tool" in rendered
            except Exception:
                ok = False
        self._supports_tools = ok
        return ok

    def _encode_messages(self, messages: list[dict], tools: list | None = None):
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tools=tools or None, add_generation_prompt=True
                )
            except Exception as exc:
                if tools:
                    raise RuntimeError(
                        f"this model's chat template does not support tools "
                        f"({exc}) — try a tool-capable model like qwen:30b"
                    ) from exc
                raise
        return self.tokenizer.encode(messages[-1]["content"] if messages else "")

    def _sync_prompt_cache(self, tokens: list[int]) -> list[int]:
        """Reconcile the persistent KV cache with *tokens*; return the
        suffix that still needs prefilling.

        Exactness: identical to vanilla mlx-lm's own KV-cached chat
        (verified in tests/test_e2e_golden.py). Versus a cold full
        prefill, outputs can differ only by floating-point kernel-shape
        effects — a property of every KV-cached runtime, not of paging.
        When the conversation diverges from what's cached, the cache is
        trimmed back to the common prefix (or rebuilt), never guessed.
        """
        from mlx_lm.models.cache import (can_trim_prompt_cache,
                                         make_prompt_cache,
                                         trim_prompt_cache)

        if self._prompt_cache is None:
            self._prompt_cache = make_prompt_cache(self.model)
            self._cached_tokens = []

        common = 0
        for a, b in zip(self._cached_tokens, tokens):
            if a != b:
                break
            common += 1
        # the model always needs at least one input token to step
        common = min(common, len(tokens) - 1)

        if common < len(self._cached_tokens):
            surplus = len(self._cached_tokens) - common
            if can_trim_prompt_cache(self._prompt_cache):
                trim_prompt_cache(self._prompt_cache, surplus)
                self._cached_tokens = self._cached_tokens[:common]
            else:
                self._prompt_cache = make_prompt_cache(self.model)
                self._cached_tokens = []
        return tokens[len(self._cached_tokens):]

    def chat_stream(self, messages: list[dict], max_tokens: int | None = None,
                    tools: list | None = None, temperature: float = 0.0):
        """Streaming chat over an explicit message list.

        Yields ``(text, telemetry)`` pairs; does not touch engine history.
        The KV cache persists across calls: only tokens not already cached
        are prefilled (prefix-matched, trimmed on divergence). ``tools``
        (OpenAI function schemas) are rendered into the chat template; the
        raw stream may then contain <tool_call> blocks for the caller to
        parse. ``temperature`` 0 = greedy (deterministic, the default and
        what the golden tests assert); >0 samples.
        """
        self._temperature = temperature
        tokens = list(self._encode_messages(messages, tools))
        suffix = self._sync_prompt_cache(tokens)
        reused = len(tokens) - len(suffix)
        generated: list[int] = []
        t_start = time.perf_counter()

        # 0 / None = unlimited: the context window is the only ceiling.
        window = self.context_limit - len(tokens) - 8
        if window <= 0:
            raise RuntimeError(
                f"context window is full ({len(tokens)} of "
                f"{self.context_limit} tokens) — /clear or start a new chat")
        requested = max_tokens if max_tokens and max_tokens > 0 \
            else (self.max_tokens if self.max_tokens > 0 else None)
        effective = min(requested, window) if requested else window

        try:
            yield from self._stream_with_cache(
                tokens, suffix, generated, reused,
                effective, t_start)
        finally:
            # Reconcile tracking with what the cache ACTUALLY holds — the
            # generator may be abandoned mid-stream, and mlx-lm pipelines
            # steps, so never infer the cache's contents arithmetically.
            full = tokens + generated
            try:
                offset = self._prompt_cache[0].offset
                self._cached_tokens = full[:offset]
            except (TypeError, IndexError, AttributeError):
                self._prompt_cache = None
                self._cached_tokens = []

    def _stream_with_cache(self, tokens, suffix, generated, reused,
                           max_tokens, t_start):
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            process = None
        n_tokens = 0

        # Greedy by default (temperature 0) — deterministic, and what the
        # golden tests assert. Only build a sampler when sampling is asked.
        extra = {}
        temp = getattr(self, "_temperature", 0.0) or 0.0
        if temp > 0:
            from mlx_lm.sample_utils import make_sampler
            extra["sampler"] = make_sampler(temp=temp)

        for response in self._mlx_lm.stream_generate(
            self.model, self.tokenizer, suffix,
            max_tokens=max_tokens,
            prompt_cache=self._prompt_cache,
            **extra,
        ):
            generated.append(response.token)
            n_tokens += 1
            elapsed = time.perf_counter() - t_start
            telemetry = {
                "n_tokens": n_tokens,
                "elapsed": elapsed,
                "throughput": n_tokens / elapsed if elapsed > 0 else 0.0,
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
                "footprint_gb": self.model_memory_gb,
                "context_tokens": len(tokens) + n_tokens,
                "kv_reused_tokens": reused,
                "finish_reason": getattr(response, "finish_reason", None),
            }
            if process is not None:
                telemetry["rss_gb"] = process.memory_info().rss / 1e9
            if self.paging is not None:
                telemetry["paging"] = self.paging.stats()
            yield response.text, telemetry

    def agent_stream(self, messages: list[dict], tools: list | None = None,
                     max_tokens: int | None = None, max_rounds: int = 6,
                     policy=None, temperature: float = 0.0):
        """Tool-using generation loop.

        Yields ("text", chunk, telemetry) for visible answer text and
        ("tool", {"name", "arguments", "result_preview"}, None) whenever a
        tool runs. The loop ends when the model answers without calling a
        tool, or after ``max_rounds`` (reported honestly as a note).

        ``policy`` (a tools.ToolPolicy) decides which tools the model both
        sees and may execute; without one, a read-only policy is used.
        """
        from sparsify.runtime import tools as toolbox

        if policy is None:
            policy = toolbox.ToolPolicy.read_only()
        schemas = tools if tools is not None else toolbox.tools_for_policy(policy)
        history = list(messages)

        for _round in range(max_rounds):
            pieces: list[str] = []
            emitted = 0
            for text, tel in self.chat_stream(history, max_tokens=max_tokens,
                                              tools=schemas, temperature=temperature):
                pieces.append(text)
                whole = "".join(pieces)
                # never emit inside (or a partial prefix of) a <tool_call>
                safe = toolbox.safe_visible_len(whole)
                if safe > emitted:
                    yield ("text", whole[emitted:safe], tel)
                    emitted = safe
            raw = "".join(pieces)
            visible, calls = toolbox.parse_tool_calls(raw)
            if len(visible) > emitted:
                yield ("text", visible[emitted:], None)
            if not calls:
                return
            history.append({"role": "assistant", "content": raw})
            for call in calls:
                result = toolbox.execute(call["name"], call["arguments"], policy)
                yield ("tool", {"name": call["name"],
                                "arguments": call["arguments"],
                                "result_preview": result[:160]}, None)
                history.append({"role": "tool", "name": call["name"],
                                "content": result})
        yield ("text", "\n\n*(stopped after "
               f"{max_rounds} tool rounds)*", None)

    def generate_stream(self, prompt: str):
        """Yield ``(text, telemetry)`` pairs, maintaining chat history."""
        self.messages.append({"role": "user", "content": prompt})
        pieces: list[str] = []
        for text, telemetry in self.chat_stream(self.messages):
            pieces.append(text)
            yield text, telemetry
        self.messages.append({"role": "assistant", "content": "".join(pieces)})

    def generate(self, prompt: str) -> str:
        """Blocking generate that prints the streamed text and a final report."""
        import sys

        full: list[str] = []
        last_telemetry = None
        for text, telemetry in self.generate_stream(prompt):
            sys.stdout.write(text)
            sys.stdout.flush()
            full.append(text)
            last_telemetry = telemetry

        if last_telemetry:
            t = last_telemetry
            print("\n")
            print("  ── Telemetry (measured) ──────────────────────────────")
            print(f"  Tokens generated : {t['n_tokens']}")
            print(f"  Throughput       : {t['throughput']:.2f} tok/s")
            print(f"  Elapsed          : {t['elapsed']:.2f}s")
            print(f"  Active memory    : {t['active_gb']:.2f} GB")
            print(f"  Peak memory      : {t['peak_gb']:.2f} GB")
            if "rss_gb" in t:
                print(f"  Process RSS      : {t['rss_gb']:.2f} GB")
            print(f"  Backbone         : {t['footprint_gb']:.2f} GB")
            if "paging" in t:
                p = t["paging"]
                print(f"  Paged to SSD     : {p['paged_gb']:.2f} GB "
                      f"({p['moe_blocks']} MoE blocks)")
                print(f"  Expert cache     : {p['resident_bytes']/1e9:.2f} / "
                      f"{p['budget_bytes']/1e9:.2f} GB budget")
                print(f"  Cache hit rate   : {p['hit_rate']*100:.1f}%  "
                      f"(hits {p['hits']}, misses {p['misses']}, "
                      f"evictions {p['evictions']})")
                print(f"  SSD reads        : {p['bytes_read']/1e9:.2f} GB "
                      f"in {p['read_seconds']:.1f}s")
            print("  ──────────────────────────────────────────────────────\n")
        return "".join(full)
