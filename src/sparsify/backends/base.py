"""Backend adapter protocol.

This module defines the abstract base for inference backend adapters.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Protocol, runtime_checkable


@dataclass
class GenerationResult:
    """Result from a text generation call.

    Attributes:
        text: The generated text.
        token_count: Number of tokens generated.
        prompt_token_count: Number of prompt tokens processed.
        total_duration_ms: Total wall-clock time in milliseconds.
        prompt_eval_ms: Time spent processing the prompt.
        generation_ms: Time spent generating tokens.
        tokens_per_second: Generation speed.
        metadata: Backend-specific additional data.
    """

    text: str
    token_count: int
    prompt_token_count: int
    total_duration_ms: float
    prompt_eval_ms: float
    generation_ms: float
    tokens_per_second: float
    metadata: dict[str, Any] | None = None


@dataclass
class ModelInfo:
    """Information about a loaded model.

    Attributes:
        name: Model identifier or filename.
        path: Path to the model file.
        architecture: Model architecture (e.g., 'llama').
        parameter_count: Total parameter count.
        quantization: Quantization type if applicable.
        memory_bytes: Memory used by the loaded model.
    """

    name: str
    path: Path | None
    architecture: str | None
    parameter_count: int | None
    quantization: str | None
    memory_bytes: int | None


@runtime_checkable
class Backend(Protocol):
    """Protocol for inference backend adapters.

    Each backend adapter wraps a specific inference engine (MLX, llama.cpp, etc.)
    and provides a uniform interface for loading models, generating text, and
    collecting metrics.
    """

    @property
    def name(self) -> str:
        """Backend name (e.g., 'mlx', 'llamacpp')."""
        ...

    def is_available(self) -> bool:
        """Check if this backend is available on the current system."""
        ...

    def load_model(self, model_path: Path, **kwargs: Any) -> ModelInfo:
        """Load a model and return its info."""
        ...

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate text from a prompt."""
        ...

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Stream generated tokens one at a time."""
        ...

    def get_memory_usage(self) -> dict[str, int]:
        """Get current memory usage in bytes.

        Returns dict with keys like 'active', 'peak', 'cache'.
        """
        ...

    def unload_model(self) -> None:
        """Unload the current model and free resources."""
        ...
