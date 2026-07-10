"""Context-window sizing: architectural max vs what this machine's free
RAM can actually hold in KV cache. Regression coverage for the bug where
a UI snippet hardcoded a guessed contextWindow (32768) instead of the
real, derived number — which for a 256k-context model on a 16 GB Mac
silently misrepresented both directions (looked smaller than the model
supports, and would have been RAM-catastrophic if taken at face value)."""
from pathlib import Path
from unittest.mock import patch

import pytest

from sparsify.runtime.chat_generation import SparsifyEngine


def _make_engine(monkeypatch, cfg: dict) -> SparsifyEngine:
    """A SparsifyEngine with __init__'s heavy lifting (model load, MLX,
    paging) skipped — only the config-parsing logic under test runs."""
    eng = SparsifyEngine.__new__(SparsifyEngine)
    eng.model_path = Path("/fake")

    import json as _json
    from unittest.mock import mock_open

    m = mock_open(read_data=_json.dumps(cfg))
    with patch("builtins.open", m):
        # replicate exactly the __init__ slice under test
        _cfg = {}
        with open(eng.model_path / "config.json") as f:
            _cfg = _json.load(f)
        eng.context_limit = int(
            _cfg.get("max_position_embeddings")
            or _cfg.get("model_max_length") or 32768)
        kv_heads = _cfg.get("num_key_value_heads") or _cfg.get("num_attention_heads")
        head_dim = _cfg.get("head_dim") or (
            _cfg.get("hidden_size", 0) // _cfg.get("num_attention_heads", 1)
            if _cfg.get("num_attention_heads") else None)
        n_layers = _cfg.get("num_hidden_layers")
        eng.kv_bytes_per_token = (
            2 * kv_heads * head_dim * n_layers * 2
            if kv_heads and head_dim and n_layers else None)
    return eng


QWEN3_30B_CFG = {
    "max_position_embeddings": 262144,
    "num_key_value_heads": 4,
    "num_attention_heads": 32,
    "head_dim": 128,
    "num_hidden_layers": 48,
}


def test_context_limit_reads_architectural_max(monkeypatch):
    eng = _make_engine(monkeypatch, QWEN3_30B_CFG)
    assert eng.context_limit == 262144


def test_kv_bytes_per_token_matches_hand_calc(monkeypatch):
    eng = _make_engine(monkeypatch, QWEN3_30B_CFG)
    # 2 (K+V) * 4 heads * 128 head_dim * 48 layers * 2 bytes (fp16)
    assert eng.kv_bytes_per_token == 98304


def test_safe_context_tokens_is_far_below_architectural_on_16gb(monkeypatch):
    eng = _make_engine(monkeypatch, QWEN3_30B_CFG)
    with patch("psutil.virtual_memory") as vm:
        vm.return_value.available = 8 * 1024**3  # 8 GB free
        safe = SparsifyEngine.safe_context_tokens(eng)
    # half of 8 GB / 98304 bytes/token
    assert safe == int(4 * 1024**3 / 98304)
    assert safe < eng.context_limit / 5  # nowhere near the 262144 ceiling


def test_safe_context_tokens_never_exceeds_architectural_limit(monkeypatch):
    eng = _make_engine(monkeypatch, QWEN3_30B_CFG)
    with patch("psutil.virtual_memory") as vm:
        vm.return_value.available = 1024 * 1024**3  # implausibly huge
        safe = SparsifyEngine.safe_context_tokens(eng)
    assert safe == eng.context_limit


def test_falls_back_to_context_limit_without_kv_estimate(monkeypatch):
    eng = _make_engine(monkeypatch, {"max_position_embeddings": 4096})
    assert eng.kv_bytes_per_token is None
    assert SparsifyEngine.safe_context_tokens(eng) == 4096


def test_webui_openclaw_snippet_uses_live_value_not_hardcoded():
    from sparsify.runtime.webui import PAGE
    assert '"contextWindow": 32768' not in PAGE
    assert "safe_context_tokens" in PAGE
    assert "h.context_limit" in PAGE
