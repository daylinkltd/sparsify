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

    def __init__(self, model_path: Path, max_tokens: int = 512,
                 memory_limit_gb: float | None = None) -> None:
        """``memory_limit_gb`` is the expert-cache budget. None (default)
        sizes it automatically from measured free system RAM at startup."""
        import mlx_lm  # deferred: slow import

        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        self._mlx_lm = mlx_lm

        if not (self.model_path / "config.json").exists():
            raise FileNotFoundError(
                f"config.json not found in {self.model_path}. "
                f"Run: sparsify pull <model>"
            )

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

    def chat_stream(self, messages: list[dict], max_tokens: int | None = None):
        """Stateless streaming chat over an explicit message list.

        Yields ``(text, telemetry)`` pairs; does not touch engine history.
        This is the API the server uses — each request carries its own
        conversation.
        """
        try:
            import psutil
            process = psutil.Process()
        except ImportError:
            process = None

        tokens = self._encode_messages(messages)
        n_tokens = 0
        t_start = time.perf_counter()

        for response in self._mlx_lm.stream_generate(
            self.model, self.tokenizer, tokens,
            max_tokens=max_tokens or self.max_tokens,
        ):
            n_tokens += 1
            elapsed = time.perf_counter() - t_start
            telemetry = {
                "n_tokens": n_tokens,
                "elapsed": elapsed,
                "throughput": n_tokens / elapsed if elapsed > 0 else 0.0,
                "active_gb": mx.get_active_memory() / 1e9,
                "peak_gb": mx.get_peak_memory() / 1e9,
                "footprint_gb": self.model_memory_gb,
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
