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
        try:
            with open(self.model_path / "config.json") as f:
                cfg = _json.load(f)
            self.context_limit = int(
                cfg.get("max_position_embeddings")
                or cfg.get("model_max_length") or 32768)
        except (OSError, ValueError):
            self.context_limit = 32768

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

    # ------------------------------------------------------------------

    def _encode_messages(self, messages: list[dict]):
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True
            )
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

    def chat_stream(self, messages: list[dict], max_tokens: int | None = None):
        """Streaming chat over an explicit message list.

        Yields ``(text, telemetry)`` pairs; does not touch engine history.
        The KV cache persists across calls: only tokens not already cached
        are prefilled (prefix-matched, trimmed on divergence).
        """
        tokens = list(self._encode_messages(messages))
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

        for response in self._mlx_lm.stream_generate(
            self.model, self.tokenizer, suffix,
            max_tokens=max_tokens,
            prompt_cache=self._prompt_cache,
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
