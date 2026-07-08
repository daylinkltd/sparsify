"""Update mechanism: version check and command wiring."""
from unittest import mock

from sparsify.runtime import updater


def test_check_no_git(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))  # no src/.git
    st = updater.check(force=True)
    assert st["source"] == "unknown"
    assert st["update_available"] is False


def test_check_update_available(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    monkeypatch.setattr(updater, "local_commit", lambda: "aaaaaa")
    monkeypatch.setattr(updater, "remote_commit", lambda timeout=4.0: "bbbbbb")
    st = updater.check(force=True)
    assert st["current"] == "aaaaaa" and st["latest"] == "bbbbbb"
    assert st["update_available"] is True


def test_check_up_to_date(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    monkeypatch.setattr(updater, "local_commit", lambda: "same123")
    monkeypatch.setattr(updater, "remote_commit", lambda timeout=4.0: "same123")
    assert updater.check(force=True)["update_available"] is False


def test_check_offline_reuses_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    monkeypatch.setattr(updater, "local_commit", lambda: "aaaaaa")
    monkeypatch.setattr(updater, "remote_commit", lambda timeout=4.0: "bbbbbb")
    updater.check(force=True)                      # populates cache
    monkeypatch.setattr(updater, "remote_commit", lambda timeout=4.0: None)  # offline
    st = updater.check(force=True)
    assert st["latest"] == "bbbbbb"               # last-known reused


def test_do_update_requires_git(monkeypatch, tmp_path):
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    try:
        updater.do_update(log=lambda m: None)
        assert False, "should refuse without a git checkout"
    except RuntimeError as exc:
        assert "not a git checkout" in str(exc)


def test_cli_version_command():
    from click.testing import CliRunner
    from sparsify.cli import main
    with mock.patch.object(updater, "check",
                           return_value={"current": "aaa", "latest": "bbb",
                                         "update_available": True, "source": "git"}):
        r = CliRunner().invoke(main, ["version"])
    assert r.exit_code == 0 and "Update available" in r.output


def test_server_version_endpoint_shape():
    # /version merges updater.check() with the semantic version
    from sparsify.runtime import updater as u
    st = u.check(force=False)
    for k in ("current", "latest", "update_available", "source"):
        assert k in st
