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
    for tag in ("Qwen3-30B-A3B-Instruct-2507-4bit", "qwen3-30b", "qwen:30b-a3b"):
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
         mock.patch.object(platform, "machine", return_value="x86_64"), \
         mock.patch.dict("sys.modules", {"torch": None}):
        try:
            backend.detect()
            assert False, "should have raised"
        except RuntimeError as exc:
            assert "Linux" in str(exc) and (
                "roadmap" in str(exc).lower()
                or "milestone" in str(exc)
                or "install pytorch" in str(exc).lower()
                or "requires pytorch" in str(exc).lower()
            )


def test_pull_typo_alias_suggests():
    result = CliRunner().invoke(main, ["pull", "qwen3:30b"])
    assert result.exit_code != 0
    assert "Did you mean" in result.output and "qwen:30b" in result.output


def test_pull_no_arg_non_tty_shows_catalog_and_fails_cleanly():
    result = CliRunner().invoke(main, ["pull"])
    assert result.exit_code != 0
    assert "olmoe:1b-7b" in result.output      # catalog rendered
    assert "no terminal" in result.output.lower()


def _fake_install(tmp_path):
    """A sparsify install shaped like the real default layout."""
    home = tmp_path / "home"
    sp = home / ".sparsify"
    (sp / "venv").mkdir(parents=True)
    (sp / "venv" / "marker").write_text("x")
    (sp / "models" / "mlx-community--Fake-4bit").mkdir(parents=True)
    (sp / "models" / "mlx-community--Fake-4bit" / "config.json").write_text("{}")
    (sp / "models" / "mlx-community--Fake-4bit" / "w.safetensors").write_text("weights")
    (home / ".local" / "bin").mkdir(parents=True)
    launcher = home / ".local" / "bin" / "sparsify"
    launcher.symlink_to(sp / "venv" / "marker")  # resolves into home: ours
    plist = tmp_path / "com.daylink.sparsify.plist"
    plist.write_text("<plist/>")
    return home, sp, launcher, plist


def _patch_install(monkeypatch, tmp_path, home, sp, plist):
    import sparsify.cli as cli_mod
    import sparsify.runtime.model_registry as reg

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SPARSIFY_HOME", str(sp))
    monkeypatch.setattr(cli_mod, "_PLIST_PATH", plist)
    monkeypatch.setattr(cli_mod, "_SYSTEM_BIN_DIRS", ())   # never touch /usr/local
    monkeypatch.setattr(cli_mod, "MODELS_DIR", sp / "models")
    monkeypatch.setattr(reg, "MODELS_DIR", sp / "models")


def test_uninstall_removes_everything(tmp_path, monkeypatch):
    home, sp, launcher, plist = _fake_install(tmp_path)
    _patch_install(monkeypatch, tmp_path, home, sp, plist)

    result = CliRunner().invoke(main, ["uninstall", "--yes"])
    assert result.exit_code == 0, result.output
    assert not sp.exists()
    assert not launcher.exists()
    assert not plist.exists()
    # deleting models must be disclosed, not silent
    assert "models" in result.output.lower()


def test_uninstall_keep_models_actually_keeps_them(tmp_path, monkeypatch):
    """Regression: default layout has models INSIDE ~/.sparsify — --keep-models
    must spare them while removing the rest of the install."""
    home, sp, launcher, plist = _fake_install(tmp_path)
    _patch_install(monkeypatch, tmp_path, home, sp, plist)

    result = CliRunner().invoke(main, ["uninstall", "--yes", "--keep-models"])
    assert result.exit_code == 0, result.output
    assert (sp / "models" / "mlx-community--Fake-4bit" / "w.safetensors").exists(), \
        "model weights were deleted despite --keep-models"
    assert not (sp / "venv").exists()
    assert not launcher.exists()


def test_uninstall_refuses_catastrophic_home(tmp_path, monkeypatch):
    import sparsify.cli as cli_mod

    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SPARSIFY_HOME", str(home))   # SPARSIFY_HOME=$HOME
    monkeypatch.setattr(cli_mod, "_SYSTEM_BIN_DIRS", ())

    result = CliRunner().invoke(main, ["uninstall", "--yes"])
    assert result.exit_code != 0
    assert "Refusing" in result.output
    assert home.exists()


def test_uninstall_leaves_foreign_launcher(tmp_path, monkeypatch):
    """A non-symlink binary named 'sparsify' (e.g. Neural Magic's PyPI
    package) must never be deleted."""
    home, sp, launcher, plist = _fake_install(tmp_path)
    launcher.unlink()
    launcher.write_text("#!/bin/sh\necho not ours\n")   # real file, not ours
    _patch_install(monkeypatch, tmp_path, home, sp, plist)

    result = CliRunner().invoke(main, ["uninstall", "--yes"])
    assert result.exit_code == 0, result.output
    assert launcher.exists(), "foreign binary named sparsify was deleted"


def test_picker_rejects_zero_and_negatives(monkeypatch):
    import sys
    import sparsify.cli as cli_mod

    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    for bad in ("0", "-1", "999", "abc"):
        monkeypatch.setattr(cli_mod.click, "prompt", lambda *a, **k: bad)
        status, alias = cli_mod._pick_model_interactively()
        assert (status, alias) == ("invalid", None), bad
    monkeypatch.setattr(cli_mod.click, "prompt", lambda *a, **k: "")
    assert cli_mod._pick_model_interactively() == ("cancel", None)
    monkeypatch.setattr(cli_mod.click, "prompt", lambda *a, **k: "1")
    status, alias = cli_mod._pick_model_interactively()
    assert status == "ok" and alias in cli_mod.KNOWN_ALIASES


def test_ps_no_server():
    result = CliRunner().invoke(main, ["ps", "--port", "7799"])
    assert result.exit_code == 0
    assert "No Sparsify server" in result.output
