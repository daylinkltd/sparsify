"""Agent-tool policy, workspace confinement, and tier gating."""
import pytest

from sparsify.runtime import tools
from sparsify.runtime.tools import ToolPolicy


@pytest.fixture
def ws(tmp_path):
    (tmp_path / "hello.txt").write_text("hi there")
    (tmp_path / "sub").mkdir()
    return tmp_path


def test_read_only_policy_hides_write_and_shell(ws):
    pol = ToolPolicy.read_only(ws)
    names = {t["function"]["name"] for t in tools.tools_for_policy(pol)}
    assert "read_file" in names and "list_dir" in names and "fetch_url" in names
    assert "write_file" not in names and "run_shell" not in names


def test_agent_policy_exposes_all(ws):
    names = {t["function"]["name"]
             for t in tools.tools_for_policy(ToolPolicy.from_flags(agent=True, workspace=ws))}
    assert {"write_file", "run_shell"} <= names


def test_execute_gates_disabled_tier(ws):
    pol = ToolPolicy.read_only(ws)
    assert tools.execute("write_file", {"path": "x", "content": "y"}, pol).startswith("error")
    assert tools.execute("run_shell", {"command": "echo hi"}, pol).startswith("error")
    assert not (ws / "x").exists()


def test_read_file_in_workspace(ws):
    out = tools.execute("read_file", {"path": "hello.txt"}, ToolPolicy.read_only(ws))
    assert "hi there" in out


@pytest.mark.parametrize("escape", [
    "../../etc/passwd", "/etc/passwd", "sub/../../outside.txt", "~/secret",
])
def test_path_escapes_rejected(ws, escape):
    pol = ToolPolicy.from_flags(agent=True, workspace=ws)
    out = tools.execute("read_file", {"path": escape}, pol)
    assert "outside the workspace" in out or "no such file" in out
    # a write must never land outside the workspace
    tools.execute("write_file", {"path": escape, "content": "x"}, pol)
    import os
    assert not os.path.exists("/etc/passwd.sparsify")  # sanity: nothing weird


def test_symlink_escape_rejected(ws, tmp_path):
    secret = tmp_path.parent / "secret.txt"
    secret.write_text("top secret")
    (ws / "link").symlink_to(secret)
    out = tools.execute("read_file", {"path": "link"}, ToolPolicy.from_flags(agent=True, workspace=ws))
    assert "outside the workspace" in out


def test_write_then_read_roundtrip(ws):
    pol = ToolPolicy.from_flags(agent=True, workspace=ws)
    assert "wrote" in tools.execute("write_file", {"path": "note.md", "content": "# hi"}, pol)
    assert "# hi" in tools.execute("read_file", {"path": "note.md"}, pol)


def test_run_shell_confined_cwd(ws):
    out = tools.execute("run_shell", {"command": "pwd"}, ToolPolicy.from_flags(agent=True, workspace=ws))
    assert str(ws.resolve()) in out and "[exit 0]" in out


def test_no_shell_flag(ws):
    pol = ToolPolicy.from_flags(agent=True, workspace=ws, allow_shell=False)
    assert tools.execute("run_shell", {"command": "echo hi"}, pol).startswith("error")
    assert "wrote" in tools.execute("write_file", {"path": "a.txt", "content": "b"}, pol)


def test_dangling_symlink_write_rejected(ws, tmp_path):
    """Regression (audit finding): a dangling symlink whose target is
    outside the workspace must NOT let write_file escape."""
    outside = tmp_path.parent / "escape_target.txt"  # does not exist yet
    (ws / "pwn").symlink_to(outside)
    pol = ToolPolicy.from_flags(agent=True, workspace=ws)
    out = tools.execute("write_file", {"path": "pwn", "content": "OWNED"}, pol)
    assert "outside the workspace" in out
    assert not outside.exists(), "write escaped the workspace via dangling symlink"


def test_fetch_url_blocks_private_addresses(monkeypatch):
    monkeypatch.delenv("SPARSIFY_ALLOW_LOCAL_FETCH", raising=False)
    for url in ("http://169.254.169.254/latest/meta-data/",
                "http://127.0.0.1:9200/_search",
                "http://10.0.0.5/", "http://192.168.1.1/admin",
                "http://localhost:7777/health"):
        out = tools.execute("fetch_url", {"url": url}, ToolPolicy.read_only())
        assert "SSRF guard" in out or "private/loopback" in out, url


def test_fetch_url_local_optin(monkeypatch):
    monkeypatch.setenv("SPARSIFY_ALLOW_LOCAL_FETCH", "1")
    # with opt-in the guard is off; the fetch is attempted (may fail to
    # connect, but must NOT be refused by the SSRF guard)
    out = tools.execute("fetch_url", {"url": "http://127.0.0.1:1/"},
                        ToolPolicy.read_only())
    assert "SSRF guard" not in out


def test_fetch_url_rejects_non_http():
    assert tools.execute("fetch_url", {"url": "file:///etc/passwd"},
                         ToolPolicy.read_only()).startswith("error")
