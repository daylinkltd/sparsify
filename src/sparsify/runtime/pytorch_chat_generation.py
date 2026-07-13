"""SparsifyEngine implementation for PyTorch backend."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import torch
from transformers.cache_utils import DynamicCache

from sparsify.backends.pytorch_backend import PyTorchBackend


class PyTorchSparsifyEngine:
    """Streaming chat engine over a storage-backed PyTorch model."""

    def __init__(self, model_path: Path, max_tokens: int = 0,
                 memory_limit_gb: float | None = None) -> None:
        self.model_path = Path(model_path)
        self.max_tokens = max_tokens

        # Load backend
        self.backend = PyTorchBackend()
        self.model_info = self.backend.load_model(self.model_path, memory_limit_gb=memory_limit_gb)

        self.model = self.backend._model
        self.tokenizer = self.backend._tokenizer
        self.paging = self.backend._paging_runtime

        # Load configuration context limit
        try:
            with open(self.model_path / "config.json") as f:
                cfg = json.load(f)
            self.context_limit = int(
                cfg.get("max_position_embeddings")
                or cfg.get("model_max_length") or 32768)
        except (OSError, ValueError):
            self.context_limit = 32768

        # KV cache tracking
        self._prompt_cache = DynamicCache()
        self._cached_tokens: list[int] = []

        # Estimate parameter memory
        self.model_memory_gb = self.model_info.memory_bytes / 1e9 if self.model_info.memory_bytes else 0.0

    def supports_tools(self) -> bool:
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

    def _encode_messages(self, messages: list[dict], tools: list | None = None) -> list[int]:
        if getattr(self.tokenizer, "chat_template", None):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tools=tools or None, add_generation_prompt=True, tokenize=True
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
        common = 0
        for a, b in zip(self._cached_tokens, tokens):
            if a != b:
                break
            common += 1
        common = min(common, len(tokens) - 1)

        if common < len(self._cached_tokens):
            self._prompt_cache.crop(common)
            self._cached_tokens = self._cached_tokens[:common]

        return tokens[len(self._cached_tokens):]

    def chat_stream(self, messages: list[dict], max_tokens: int | None = None,
                    tools: list | None = None, temperature: float = 0.0):
        self._temperature = temperature
        tokens = list(self._encode_messages(messages, tools))
        suffix = self._sync_prompt_cache(tokens)
        reused = len(tokens) - len(suffix)
        t_start = time.perf_counter()

        window = self.context_limit - len(tokens) - 8
        if window <= 0:
            raise RuntimeError(
                f"context window is full ({len(tokens)} of {self.context_limit} tokens used)"
            )

        limit = max_tokens or self.max_tokens or window
        limit = min(limit, window)

        n_tokens = 0
        device = self.backend._device

        # 1. Prefill suffix (if any)
        if suffix:
            suffix_tensor = torch.tensor([suffix], device=device)
            with torch.no_grad():
                outputs = self.model(input_ids=suffix_tensor, past_key_values=self._prompt_cache, use_cache=True)
            self._cached_tokens.extend(suffix)
            logits = outputs.logits[:, -1, :]
        else:
            logits = None

        # Autoregressive generation
        while n_tokens < limit:
            if logits is None:
                last_token = torch.tensor([[self._cached_tokens[-1]]], device=device)
                with torch.no_grad():
                    outputs = self.model(input_ids=last_token, past_key_values=self._prompt_cache, use_cache=True)
                logits = outputs.logits[:, -1, :]

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            if next_token_id == self.tokenizer.eos_token_id:
                break

            self._cached_tokens.append(next_token_id)
            n_tokens += 1

            with torch.no_grad():
                outputs = self.model(input_ids=next_token, past_key_values=self._prompt_cache, use_cache=True)
            logits = outputs.logits[:, -1, :]

            elapsed = time.perf_counter() - t_start
            mem = self.backend.get_memory_usage()
            telemetry = {
                "n_tokens": n_tokens,
                "elapsed": elapsed,
                "throughput": n_tokens / elapsed if elapsed > 0 else 0.0,
                "active_gb": mem["active"] / 1e9,
                "peak_gb": mem["peak"] / 1e9,
                "footprint_gb": self.model_memory_gb,
                "context_tokens": len(self._cached_tokens),
                "kv_reused_tokens": reused,
                "finish_reason": "length" if n_tokens >= limit else None,
            }
            if self.paging is not None:
                telemetry["paging"] = self.paging.stats()

            yield self.tokenizer.decode([next_token_id]), telemetry

    def agent_stream(self, messages: list[dict], tools: list | None = None,
                      max_tokens: int | None = None, max_rounds: int = 6,
                      policy=None, temperature: float = 0.0):
        from sparsify.runtime import tools as toolbox

        if policy is None:
            policy = toolbox.ToolPolicy.read_only()

        rendered_tools = [t.schema for t in policy.tools.values()] if policy.tools else None

        for text, tel in self.chat_stream(messages, max_tokens=max_tokens,
                                         tools=rendered_tools, temperature=temperature):
            yield "text", text, tel
