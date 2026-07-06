"""Unit and integration tests for Ollama compatibility layer commands."""
from __future__ import annotations

import click
from click.testing import CliRunner
import pytest
import requests
import threading
import time

from sparsify.cli import main


def test_ollama_compatibility_cli_subcommands() -> None:
    """Validate pull, list, stats, and mock run execution loops."""
    runner = CliRunner()
    
    # 1. Test pull command
    pull_res = runner.invoke(main, ["pull", "mixtral:8x7b-instruct"])
    assert pull_res.exit_code == 0
    assert "Successfully pulled" in pull_res.output

    # 2. Test list command
    list_res = runner.invoke(main, ["list"])
    assert list_res.exit_code == 0
    assert "Sparsify Local Models" in list_res.output
    assert "mixtral:8x7b-instr" in list_res.output
    
    # 3. Test stats command
    stats_res = runner.invoke(main, ["stats"])
    assert stats_res.exit_code == 0
    assert "Sparsify Inference Historical Stats" in stats_res.output
    assert "Shannon Routing Entropy" in stats_res.output
    
    # 4. Test run command with /exit input
    run_res = runner.invoke(main, ["run", "mixtral:8x7b-instruct"], input="/exit\n")
    assert run_res.exit_code == 0
    assert "Sparsify Runtime v1.0" in run_res.output


def test_openai_compatibility_completions_server() -> None:
    """Spin up the REST serve endpoint in the background and query it."""
    runner = CliRunner()
    
    port = 18000
    # Start server in background thread
    server_thread = threading.Thread(
        target=lambda: runner.invoke(main, ["serve", "mixtral:8x7b-instruct", "--port", str(port)]),
        daemon=True
    )
    server_thread.start()
    
    # Give the HTTP server a brief moment to bind
    time.sleep(0.5)
    
    url = f"http://localhost:{port}/v1/chat/completions"
    payload = {
        "model": "mixtral:8x7b-instruct",
        "messages": [{"role": "user", "content": "Test completions prompt"}]
    }
    
    response = requests.post(url, json=payload, timeout=2.0)
    assert response.status_code == 200
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Hello from Sparsify OpenAI compatibility layer!"
    assert data["model"] == "mixtral:8x7b-instruct"
