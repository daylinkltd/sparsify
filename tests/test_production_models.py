"""Unit and integration tests for real production models and inspect CLI command."""
from __future__ import annotations

import click
from click.testing import CliRunner
import pytest
from pathlib import Path

from sparsify.cli import main


def test_production_pull_and_rejection_rules() -> None:
    """Validate pulling real MoE models, checking size bounds, and rejecting test models."""
    runner = CliRunner()

    # 1. Reject deprecated testing models
    res_reject = runner.invoke(main, ["pull", "mixtral:8x7b"])
    assert res_reject.exit_code != 0
    assert "rejected as test models" in res_reject.output

    # 2. Reject small HF repositories
    res_reject_small = runner.invoke(main, ["pull", "hf-internal-testing/tiny-random-MixtralForCausalLM"])
    assert res_reject_small.exit_code != 0
    assert "rejected as test models" in res_reject_small.output or "Rejected as test model" in res_reject_small.output

    # 3. Pull real model successfully
    res_pull = runner.invoke(main, ["pull", "mixtral:8x7b-instruct"])
    assert res_pull.exit_code == 0
    assert "Successfully pulled" in res_pull.output
    assert "Total Occupied Size: 45.0" in res_pull.output
    assert "Safetensor Shard Count: 19" in res_pull.output

    # Check file exists and reports correct virtual size
    local_path = Path("/Volumes/projects/sparsify/models/mixtral_8x7b-instruct")
    assert local_path.exists()
    assert (local_path / "config.json").exists()
    # Check one shard size
    shard = local_path / "model-00001-of-00019.safetensors"
    assert shard.exists()
    assert shard.stat().st_size > 1 * 1024 * 1024 * 1024


def test_production_inspect_subcommand() -> None:
    """Validate that sparsify inspect subcommand correctly prints parameter and memory estimation maps."""
    runner = CliRunner()

    res_inspect = runner.invoke(main, ["inspect", "mixtral:8x7b-instruct"])
    assert res_inspect.exit_code == 0
    assert "Sparsify Model Inspection: mixtral:8x7b-instruct" in res_inspect.output
    assert "MixtralForCausalLM" in res_inspect.output
    assert "Shared Parameters" in res_inspect.output
    assert "24.5 GB" in res_inspect.output
    assert "2.72 GB/s" in res_inspect.output
