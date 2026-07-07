"""Verification suite for sparsify.paging.

Every assertion here compares against an independent reference:
  - store reads   vs MLX's own safetensors reader (mx.load)
  - paged modules vs the exact gather_qmm computation mlx-lm performs
  - cache bounds  vs its configured byte budget

Uses the real Qwen3-30B-A3B shards on disk (bfloat16 scales — the hard
dtype case). Skipped if the model is not present.
"""
from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from sparsify.paging.cache import ExpertCache
from sparsify.paging.modules import ExpertGroup, PagedSwitchLinear
from sparsify.paging.store import SafetensorsExpertStore

MODEL = Path(__file__).resolve().parents[1] / "models" / "mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit"
PREFIX = "model.layers.0.mlp.switch_mlp"

pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="Qwen3 model not on disk")


@pytest.fixture(scope="module")
def store():
    return SafetensorsExpertStore(MODEL)


@pytest.fixture(scope="module")
def reference_tensors():
    """Full stacked layer-0 tensors via MLX's native reader (independent path)."""
    with open(MODEL / "model.safetensors.index.json") as f:
        wm = json.load(f)["weight_map"]
    names = [f"{PREFIX}.{p}.{t}" for p in ("gate_proj", "up_proj", "down_proj")
             for t in ("weight", "scales", "biases")]
    shards = {wm[n] for n in names}
    tensors = {}
    for shard in shards:
        loaded = mx.load(str(MODEL / shard))
        tensors.update({n: loaded[n] for n in names if n in loaded})
    return tensors


def test_store_slice_matches_reference(store, reference_tensors):
    """A pread expert slice must be bit-identical to the reference tensor row."""
    for name, full in reference_tensors.items():
        for e in (0, 7, int(full.shape[0]) - 1):
            got = store.read_expert_slice(name, e)
            want = full[e : e + 1]
            assert got.dtype == want.dtype, name
            assert got.shape == want.shape, name
            assert mx.array_equal(got, want), f"{name}[{e}] mismatch"


def test_store_rejects_out_of_range(store):
    with pytest.raises(IndexError):
        store.read_expert_slice(f"{PREFIX}.gate_proj.weight", 128)


def _make_paged(store, budget_bytes=1 << 30):
    cache = ExpertCache(store, budget_bytes)
    proj_params = {p: ["weight", "scales", "biases"]
                   for p in ("gate_proj", "up_proj", "down_proj")}
    group = ExpertGroup(0, PREFIX, 128, proj_params, cache)
    mods = {
        p: PagedSwitchLinear(group, p, quantized=True, group_size=64, bits=4,
                             mode="affine", has_scalar_biases=True)
        for p in proj_params
    }
    return group, cache, mods


def test_paged_module_matches_reference_computation(store, reference_tensors):
    """Paged output must equal gather_qmm over the full stacked tensors —
    the exact computation mlx-lm's QuantizedSwitchLinear performs."""
    _, _, mods = _make_paged(store)
    mx.random.seed(0)

    for shape_note, indices in [
        ("decode top-8", mx.array(np.array([[[3, 17, 42, 99, 100, 101, 5, 77]]], dtype=np.uint32))),
        ("repeated experts", mx.array(np.array([[[7, 7, 7, 2, 2, 1, 0, 127]]], dtype=np.uint32))),
        ("prefill 4 tokens", mx.array(np.random.default_rng(1).integers(0, 128, (1, 4, 8)).astype(np.uint32))),
    ]:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            w = reference_tensors[f"{PREFIX}.{proj}.weight"]
            in_dim = w.shape[2] * 8  # uint32 packs 8 4-bit values
            # Same layout SwitchGLU feeds the projections: (B, L, 1, 1, D)
            x = mx.expand_dims(
                mx.random.normal((*indices.shape[:2], in_dim)).astype(mx.bfloat16),
                (-2, -3),
            )

            ref = mx.gather_qmm(
                x,
                w,
                reference_tensors[f"{PREFIX}.{proj}.scales"],
                reference_tensors[f"{PREFIX}.{proj}.biases"],
                rhs_indices=indices,
                transpose=True,
                group_size=64,
                bits=4,
            )
            got = mods[proj](x, indices)
            assert got.dtype == ref.dtype
            assert mx.array_equal(got, ref), (
                f"{proj} ({shape_note}): max|Δ| = "
                f"{mx.abs(got.astype(mx.float32) - ref.astype(mx.float32)).max().item()}"
            )


def test_cache_respects_budget_and_evicts_lru(store):
    """Resident bytes never exceed budget; eviction order is LRU."""
    group, cache, _ = _make_paged(store, budget_bytes=0)  # force size discovery
    one = cache.get_experts(group, (0,))
    expert_bytes = sum(t.nbytes for proj in one[0].values() for t in proj.values())

    budget = int(expert_bytes * 4.5)  # fits 4 experts
    group, cache, _ = _make_paged(store, budget_bytes=budget)

    cache.get_experts(group, (0, 1, 2, 3))
    assert cache.stats()["resident_experts"] == 4
    cache.get_experts(group, (0,))          # refresh 0 -> LRU order 1,2,3,0
    cache.get_experts(group, (4, 5))        # evicts 1 then 2
    stats = cache.stats()
    assert stats["resident_bytes"] <= budget
    assert stats["evictions"] == 2
    assert set(k[1] for k in cache._entries) == {3, 0, 4, 5}


def test_cache_protects_current_call_from_self_eviction(store):
    """A working set larger than the budget overshoots (measured) rather than
    evicting experts the in-flight call still needs."""
    group, cache, _ = _make_paged(store, budget_bytes=1)  # absurdly small
    got = cache.get_experts(group, (0, 1, 2))
    assert set(got) == {0, 1, 2}
    assert cache.stats()["resident_experts"] == 3  # none self-evicted mid-call
    cache.get_experts(group, (3,))
    assert cache.stats()["resident_experts"] == 1  # prior entries evictable again


def test_dense_model_is_left_untouched():
    """attach_paging must return None for a model with no expert structure."""
    import mlx_lm
    from sparsify.paging import attach_paging

    dense = Path(__file__).resolve().parents[1] / "models" / "mlx-community--Llama-3.2-1B-Instruct-4bit"
    if not dense.exists():
        pytest.skip("dense reference model not on disk")
    model, _ = mlx_lm.load(str(dense), lazy=True)
    assert attach_paging(model, dense, 1 << 30) is None
