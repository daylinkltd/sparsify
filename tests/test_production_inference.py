"""Tests for the real inference engine — no mocks, no placeholders."""
from __future__ import annotations

from click.testing import CliRunner
from sparsify.cli import main


def _run(inputs: str, model: str = "mlx-community/Llama-3.2-1B-Instruct-4bit") -> str:
    r = CliRunner().invoke(main, ["run", model], input=inputs)
    return r.output


def test_list_shows_real_registry() -> None:
    """sparsify list reads the JSON registry and shows pulled models."""
    r = CliRunner().invoke(main, ["list"])
    assert r.exit_code == 0
    # Either shows a model or the 'no models' hint
    assert "Sparsify Local Models" in r.output or "sparsify pull" in r.output


def test_run_startup_banner() -> None:
    """Startup banner shows model id, backend, and memory footprint."""
    out = _run("/exit\n")
    assert "Sparsify Runtime" in out
    assert "mlx-community/Llama-3.2-1B-Instruct-4bit" in out
    assert "Apple MLX" in out
    assert "Model footprint" in out
    assert "GB" in out


def test_run_india_capital() -> None:
    """Real mlx-lm inference must answer 'capital of India' with New Delhi."""
    out = _run("What is the capital of India?\n/exit\n")
    assert "New Delhi" in out
    # No old placeholders
    assert "Sparsify Runtime is executing" not in out
    assert "[SIMULATED]" not in out
    assert "[PLACEHOLDER]" not in out


def test_run_japan_capital() -> None:
    """Real mlx-lm inference must answer 'capital of Japan' with Tokyo."""
    out = _run("What is the capital of Japan?\n/exit\n")
    assert "Tokyo" in out


def test_run_telemetry_is_real() -> None:
    """Telemetry block shows measured values — throughput, memory in GB."""
    out = _run("hey\n/exit\n")
    assert "Tokens generated" in out
    assert "Throughput" in out
    assert "tok/s" in out
    assert "Active memory" in out
    assert "Peak memory" in out
    # No simulation labels
    assert "[SIMULATED]" not in out
    assert "[PLACEHOLDER]" not in out


def test_stats_shows_hardware_memory() -> None:
    """sparsify stats shows real MLX hardware stats."""
    r = CliRunner().invoke(main, ["stats"])
    assert r.exit_code == 0
    assert "Active unified memory" in r.output
    assert "Peak unified memory" in r.output
    assert "GB" in r.output
