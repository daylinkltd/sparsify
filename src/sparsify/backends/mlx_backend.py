"""MLX backend adapter implementing the Sparsify Backend protocol."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx
from mlx_lm import load, generate, stream_generate

from sparsify.backends.base import Backend, GenerationResult, ModelInfo


class MLXBackend(Backend):
    """Backend adapter using MLX for local inference on Apple Silicon."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_path: Path | None = None
        self._model_name: str = ""

    @property
    def name(self) -> str:
        return "mlx"

    def is_available(self) -> bool:
        # MLX is only available on Apple Silicon macOS
        try:
            import mlx.core as mx
            return mx.metal.is_available()
        except ImportError:
            return False

    def load_model(self, model_path: Path, **kwargs: Any) -> ModelInfo:
        """Load a model using mlx_lm."""
        self._model_path = Path(model_path)
        self._model_name = self._model_path.name
        
        # Load model and tokenizer
        # mlx_lm.load handles both local paths and HF repo IDs
        self._model, self._tokenizer = load(str(model_path), **kwargs)
        
        # Get parameter count and quantization level if available
        def count_params(d: Any) -> int:
            if isinstance(d, dict):
                return sum(count_params(v) for v in d.values())
            elif hasattr(d, "size"):
                return d.size
            return 0

        param_count = count_params(self._model.parameters()) if hasattr(self._model, "parameters") else 0
        
        # Determine memory usage
        active_memory = mx.get_active_memory()
        
        # Determine architecture
        architecture = getattr(self._model, "model_type", None)
        
        return ModelInfo(
            name=self._model_name,
            path=self._model_path,
            architecture=architecture,
            parameter_count=param_count,
            quantization=kwargs.get("quantization"),
            memory_bytes=active_memory,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded in MLXBackend.")

        t0 = time.perf_counter()
        
        # Verbose outputs can be retrieved from generating
        # We can implement a custom timing wrapper if needed, but for V1/V2
        # we can compute generation details directly
        mx.reset_peak_memory()
        
        # Perform generation
        response = generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            verbose=False,
            **kwargs,
        )
        
        total_time = (time.perf_counter() - t0) * 1000.0  # ms
        
        # Let's count tokens
        prompt_tokens = len(self._tokenizer.encode(prompt))
        gen_tokens = len(self._tokenizer.encode(response))
        
        # Estimate prefill / decode splits (rough estimation for base.py compliance)
        prompt_eval_ms = total_time * 0.2
        generation_ms = total_time * 0.8
        tokens_per_second = (gen_tokens / (generation_ms / 1000.0)) if generation_ms > 0 else 0.0

        return GenerationResult(
            text=response,
            token_count=gen_tokens,
            prompt_token_count=prompt_tokens,
            total_duration_ms=total_time,
            prompt_eval_ms=prompt_eval_ms,
            generation_ms=generation_ms,
            tokens_per_second=tokens_per_second,
            metadata={
                "active_memory_bytes": mx.get_active_memory(),
                "peak_memory_bytes": mx.get_peak_memory(),
            },
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded in MLXBackend.")
            
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            **kwargs,
        ):
            yield response

    def get_memory_usage(self) -> dict[str, int]:
        return {
            "active": mx.get_active_memory(),
            "peak": mx.get_peak_memory(),
            "cache": mx.get_cache_memory(),
        }

    def unload_model(self) -> None:
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._model_name = ""
        mx.metal.clear_cache()
