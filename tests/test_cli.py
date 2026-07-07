"""CLI smoke tests — every command must at least parse and execute its
non-model-loading paths. Guards against broken imports in command bodies
(click only resolves names when a command actually runs)."""
from __future__ import annotations

from click.testing import CliRunner

from sparsify.cli import main


def test_models_command_runs():
    result = CliRunner().invoke(main, ["models"])
    assert result.exit_code == 0, result.output
    assert "mixtral:8x7b" in result.output


def test_list_command_runs():
    result = CliRunner().invoke(main, ["list"])
    assert result.exit_code == 0, result.output


def test_run_missing_model_fails_cleanly():
    result = CliRunner().invoke(main, ["run", "does-not/exist"])
    assert result.exit_code != 0
    assert "no local model matches" in result.output.lower()
    assert "sparsify pull" in result.output


def test_pull_help():
    result = CliRunner().invoke(main, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output


def test_resolve_local_short_names():
    from sparsify.runtime.model_registry import MODELS_DIR, resolve_local

    if not (MODELS_DIR / "mlx-community--Qwen3-30B-A3B-Instruct-2507-4bit" / "config.json").exists():
        import pytest
        pytest.skip("Qwen3 not on disk")
    # bare HF name without org prefix, and a unique substring
    for tag in ("Qwen3-30B-A3B-Instruct-2507-4bit", "qwen3", "qwen:30b-a3b"):
        resolved = resolve_local(tag)
        assert resolved is not None, tag
        assert resolved[0] == "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
    # ambiguous or unknown stays None
    assert resolve_local("definitely-not-a-model") is None


def test_backend_detection_messages():
    import platform
    from unittest import mock
    from sparsify.runtime import backend

    assert backend.detect().name == "mlx"  # dev machine is Apple Silicon
    with mock.patch.object(platform, "system", return_value="Linux"), \
         mock.patch.object(platform, "machine", return_value="x86_64"):
        try:
            backend.detect()
            assert False, "should have raised"
        except RuntimeError as exc:
            assert "Linux" in str(exc) and "roadmap" in str(exc).lower() or "milestone" in str(exc)
