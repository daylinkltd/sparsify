"""PyTorch backend adapter implementing the Sparsify Backend protocol."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from sparsify.backends.base import Backend, GenerationResult, ModelInfo
from sparsify.paging_torch.surgery import attach_pytorch_paging, PyTorchPagingRuntime


class PyTorchBackend(Backend):
    """Backend adapter using PyTorch for inference and expert paging."""

    def __init__(self) -> None:
        self._model: Any = None
        self._tokenizer: Any = None
        self._model_path: Path | None = None
        self._model_name: str = ""
        self._device: torch.device = torch.device("cpu")
        self._paging_runtime: PyTorchPagingRuntime | None = None

    @property
    def name(self) -> str:
        return "pytorch"

    def is_available(self) -> bool:
        try:
            import torch
            return True
        except ImportError:
            return False

    def load_model(self, model_path: Path, **kwargs: Any) -> ModelInfo:
        """Load a PyTorch model and apply surgery."""
        self._model_path = Path(model_path)
        self._model_name = self._model_path.name

        # Detect the best accelerator device
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        # Determine budget in bytes
        # default to 4GB if not specified
        memory_limit_gb = kwargs.get("memory_limit_gb")
        budget_bytes = int(memory_limit_gb * 1024 * 1024 * 1024) if memory_limit_gb else 4 * 1024 * 1024 * 1024

        # Load model to CPU first to avoid VRAM spikes during surgery
        self._model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )

        # Count total params before surgery
        param_count = sum(p.numel() for p in self._model.parameters())

        # Apply paging surgery
        self._paging_runtime = attach_pytorch_paging(
            self._model, self._model_path, budget_bytes, device=self._device
        )

        # Move non-paged (resident) model parameters to the target device
        self._model.to(self._device)
        self._model.eval()

        return ModelInfo(
            name=self._model_name,
            path=self._model_path,
            architecture=getattr(self._model.config, "model_type", "Unknown"),
            parameter_count=param_count,
            quantization=None,
            memory_bytes=self.get_memory_usage()["active"],
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> GenerationResult:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded in PyTorchBackend.")

        t0 = time.perf_counter()

        # Generate tokens
        response_tokens = []
        for token_text in self.stream_generate(prompt, max_tokens, temperature, **kwargs):
            pass

        # Re-generate full response for exact token counting
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        prompt_tokens = inputs["input_ids"].shape[1]

        # Use native generate to compute full output text cleanly for non-streaming
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else None,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        response_ids = outputs[0, prompt_tokens:]
        response = self._tokenizer.decode(response_ids, skip_special_tokens=True)

        total_time = (time.perf_counter() - t0) * 1000.0  # ms
        prompt_eval_ms = total_time * 0.2
        generation_ms = total_time * 0.8
        tokens_per_second = (len(response_ids) / (generation_ms / 1000.0)) if generation_ms > 0 else 0.0

        return GenerationResult(
            text=response,
            token_count=len(response_ids),
            prompt_token_count=prompt_tokens,
            total_duration_ms=total_time,
            prompt_eval_ms=prompt_eval_ms,
            generation_ms=generation_ms,
            tokens_per_second=tokens_per_second,
            metadata=self.get_memory_usage(),
        )

    def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> Iterator[str]:
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("No model loaded in PyTorchBackend.")

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        past_key_values = None

        with torch.no_grad():
            for _ in range(max_tokens):
                if past_key_values is None:
                    outputs = self._model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
                else:
                    last_token = input_ids[:, -1:]
                    outputs = self._model(input_ids=last_token, use_cache=True, past_key_values=past_key_values)

                logits = outputs.logits[:, -1, :]
                past_key_values = outputs.past_key_values

                if temperature > 0:
                    probs = torch.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                next_token_id = next_token.item()
                if next_token_id == self._tokenizer.eos_token_id:
                    break

                # Update inputs for next steps
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=self._device)], dim=-1)

                yield self._tokenizer.decode([next_token_id])

    def get_memory_usage(self) -> dict[str, int]:
        active = 0
        peak = 0
        cache = 0
        if self._device.type == "cuda":
            active = torch.cuda.memory_allocated(self._device)
            peak = torch.cuda.max_memory_allocated(self._device)
            cache = torch.cuda.memory_reserved(self._device)
        elif self._device.type == "mps":
            active = torch.mps.current_allocated_memory()
            import psutil
            process = psutil.Process()
            peak = process.memory_info().rss
            cache = 0
        else:
            import psutil
            process = psutil.Process()
            active = process.memory_info().rss
            peak = active
            cache = 0

        return {"active": active, "peak": peak, "cache": cache}

    def unload_model(self) -> None:
        if self._paging_runtime:
            self._paging_runtime.close()
            self._paging_runtime = None
        self._model = None
        self._tokenizer = None
        self._model_path = None
        self._model_name = ""
        if self._device.type == "cuda":
            torch.cuda.empty_cache()
        elif self._device.type == "mps":
            torch.mps.empty_cache()
