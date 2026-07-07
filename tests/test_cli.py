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
    assert "not found" in result.output.lower()


def test_pull_help():
    result = CliRunner().invoke(main, ["pull", "--help"])
    assert result.exit_code == 0
    assert "--force" in result.output
