"""End-to-end golden tests: SparsifyEngine output vs unmodified mlx-lm.

The MoE golden test uses OLMoE-1B-7B (4-bit, ~3.7 GB) because it is the
largest MoE that also fits fully in RAM on the 16 GB dev machine — the only
honest way to compare storage-backed output against full-RAM inference.
Greedy decoding, exact string equality.

Slow (loads models); run explicitly:
    pytest tests/test_e2e_golden.py -v -m e2e
"""
from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"
DENSE = MODELS / "mlx-community--Llama-3.2-1B-Instruct-4bit"
MOE = MODELS / "mlx-community--OLMoE-1B-7B-0125-Instruct-4bit"

PROMPT = "Name the planets of the solar system in order from the sun."
MAX_TOKENS = 80

pytestmark = pytest.mark.e2e


def _baseline(model_dir: Path) -> str:
    """Reference output from unmodified mlx-lm, full-RAM load."""
    import mlx_lm

    model, tokenizer = mlx_lm.load(str(model_dir))
    tokens = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], add_generation_prompt=True
    )
    return mlx_lm.generate(model, tokenizer, tokens, max_tokens=MAX_TOKENS)


def _sparsify(model_dir: Path, memory_limit_gb: float) -> tuple[str, dict | None]:
    from sparsify.runtime.chat_generation import SparsifyEngine

    engine = SparsifyEngine(model_dir, max_tokens=MAX_TOKENS,
                            memory_limit_gb=memory_limit_gb)
    text = "".join(t for t, _ in engine.generate_stream(PROMPT))
    stats = engine.paging.stats() if engine.paging else None
    del engine
    mx.clear_cache()
    return text, stats


@pytest.mark.skipif(not DENSE.exists(), reason="dense model not on disk")
def test_dense_output_identical_to_mlx_lm():
    """Dense models must pass through the engine completely unmodified."""
    want = _baseline(DENSE)
    mx.clear_cache()
    got, stats = _sparsify(DENSE, memory_limit_gb=8)
    assert stats is None, "dense model must not be paged"
    assert got == want


@pytest.mark.skipif(not MOE.exists(), reason="OLMoE model not on disk")
def test_moe_paged_output_identical_to_full_ram():
    """Storage-backed MoE inference must reproduce full-RAM output exactly."""
    want = _baseline(MOE)
    mx.clear_cache()
    got, stats = _sparsify(MOE, memory_limit_gb=1)  # 1 GB << 3.4 GB of experts
    assert stats is not None and stats["misses"] > 0, "paging did not engage"
    assert stats["evictions"] > 0, "budget never pressured — test proves nothing"
    assert got == want


@pytest.mark.skipif(not MOE.exists(), reason="OLMoE model not on disk")
def test_multi_turn_kv_cache_identical_to_mlx_lm_chat():
    """Multi-turn chat with Sparsify's persistent KV cache must match
    vanilla mlx-lm operating the same way (standard KV cache, suffix-only
    prefill) — the way every runtime actually serves multi-turn chat.

    (Cold full-prefill each turn is NOT the reference: different prefill
    batch shapes change Metal reduction order, and near-tie tokens can
    flip late in a generation — a floating-point property of any cached
    runtime, not a Sparsify behavior.)"""
    import mlx_lm
    from mlx_lm.models.cache import make_prompt_cache
    from sparsify.runtime.chat_generation import SparsifyEngine

    turns = ["Name two planets.", "Which of those is bigger?",
             "And how many moons does it have?"]

    # baseline: vanilla mlx-lm with its own persistent prompt cache
    model, tokenizer = mlx_lm.load(str(MOE))
    prompt_cache = make_prompt_cache(model)
    history, baseline, prev_tokens = [], [], []
    for q in turns:
        history.append({"role": "user", "content": q})
        tokens = list(tokenizer.apply_chat_template(history, add_generation_prompt=True))
        suffix = tokens[len(prev_tokens):]
        pieces, gen = [], []
        for r in mlx_lm.stream_generate(model, tokenizer, suffix,
                                        max_tokens=48, prompt_cache=prompt_cache):
            pieces.append(r.text)
            gen.append(r.token)
        text = "".join(pieces)
        baseline.append(text)
        history.append({"role": "assistant", "content": text})
        prev_tokens = (tokens + gen)[:prompt_cache[0].offset]
    del model
    mx.clear_cache()

    engine = SparsifyEngine(MOE, max_tokens=48, memory_limit_gb=1)
    got = []
    for i, q in enumerate(turns):
        text = "".join(t for t, _ in engine.generate_stream(q))
        got.append(text)
        assert text == baseline[i], f"turn {i+1} diverged"
    # reuse must actually have happened (turn 2+ prefills only the suffix)
    assert engine._cached_tokens, "KV cache tracking is empty"
