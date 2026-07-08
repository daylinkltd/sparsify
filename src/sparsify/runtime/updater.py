"""Self-update: check GitHub for newer commits and pull+reinstall in place.

The install created by install.sh is a git checkout at ``$SPARSIFY_HOME/src``
with its own venv at ``$SPARSIFY_HOME/venv``. Updating = fast-forward that
checkout, reinstall it into the venv, re-brand the runtime, and restart the
login service. Version identity is the git commit; the GitHub API gives the
latest ``main`` commit for the "update available" check.

Every network call is short-timeout and best-effort — the check must never
block or crash the CLI/UI; a failed check just reports "unknown".
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

REPO = "daylinkltd/sparsify"
BRANCH = "main"
_CHECK_TTL = 6 * 3600  # re-check the remote at most every 6 hours


def _home() -> Path:
    return Path(os.environ.get("SPARSIFY_HOME", str(Path.home() / ".sparsify")))


def src_dir() -> Path:
    return _home() / "src"


def _cache_path() -> Path:
    return _home() / "update_cache.json"


def local_commit() -> str | None:
    """Short SHA of the installed checkout, or None if not a git install."""
    src = src_dir()
    if not (src / ".git").exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip()[:12] or None
    except (OSError, subprocess.SubprocessError):
        return None


def remote_commit(timeout: float = 4.0) -> str | None:
    """Latest ``main`` commit on GitHub, or None if unreachable."""
    import urllib.request

    url = f"https://api.github.com/repos/{REPO}/commits/{BRANCH}"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "sparsify-updater",
                          "Accept": "application/vnd.github.sha"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read(200).decode("utf-8", errors="replace").strip()
        # With the .sha Accept header GitHub returns the raw SHA; fall back
        # to JSON if a proxy stripped it.
        if len(body) >= 7 and all(c in "0123456789abcdef" for c in body[:40]):
            return body[:12]
        return json.loads(body).get("sha", "")[:12] or None
    except (OSError, ValueError):
        return None


def check(force: bool = False) -> dict:
    """Cached update status: {current, latest, update_available, checked_at,
    source}. Refreshes from GitHub at most every _CHECK_TTL unless *force*."""
    cur = local_commit()
    cache = {}
    cp = _cache_path()
    if cp.exists():
        try:
            cache = json.loads(cp.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    fresh = (not force and cache.get("checked_at")
             and (time.time() - cache["checked_at"]) < _CHECK_TTL
             and cache.get("current") == cur)
    if fresh:
        latest = cache.get("latest")
    else:
        latest = remote_commit()
        if latest is not None:
            try:
                cp.write_text(json.dumps(
                    {"current": cur, "latest": latest, "checked_at": time.time()}))
            except OSError:
                pass
        elif cache.get("current") == cur:
            latest = cache.get("latest")  # reuse last known if offline

    return {
        "current": cur,
        "latest": latest,
        "update_available": bool(cur and latest and cur != latest),
        "source": "git" if (src_dir() / ".git").exists() else "unknown",
    }


def do_update(log=print, restart: bool = True) -> tuple[str | None, str | None]:
    """Fast-forward the checkout, reinstall into the venv, re-brand, restart.
    Returns (old_commit, new_commit). Raises RuntimeError on failure."""
    home = _home()
    src = src_dir()
    venv_pip = home / "venv" / "bin" / "pip"
    if not (src / ".git").exists():
        raise RuntimeError(
            f"{src} is not a git checkout — reinstall with:\n"
            "  curl -fsSL https://github.com/daylinkltd/sparsify/releases/"
            "latest/download/install.sh | sh")
    if not venv_pip.exists():
        raise RuntimeError(f"venv not found at {home}/venv — reinstall Sparsify")

    old = local_commit()
    log("fetching latest…")
    _run(["git", "-C", str(src), "fetch", "--depth", "1", "origin", BRANCH])
    _run(["git", "-C", str(src), "reset", "--hard", f"origin/{BRANCH}"])
    new = local_commit()
    if new == old:
        log("already up to date.")
        return old, new

    log(f"installing {old} → {new}…")
    _run([str(venv_pip), "install", "--quiet", "--upgrade", str(src)])
    _rebrand(home)

    if restart:
        _restart_service(home, log)
    return old, new


def _run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}…: {r.stderr.strip() or r.stdout.strip()}")


def _rebrand(home: Path) -> None:
    """Reapply the 'sparsify-runtime' process name (pip reset the shebang)."""
    import shutil

    venv = home / "venv" / "bin"
    script, python = venv / "sparsify", venv / "python3"
    runtime = venv / "sparsify-runtime"
    try:
        if python.exists() and not runtime.exists():
            shutil.copy(python.resolve(), runtime)
            runtime.chmod(0o755)
        if script.exists() and runtime.exists():
            lines = script.read_text().splitlines(keepends=True)
            if lines and lines[0].startswith("#!") and "sparsify-runtime" not in lines[0]:
                lines[0] = f"#!{runtime}\n"
                script.write_text("".join(lines))
    except OSError:
        pass


def _restart_service(home: Path, log) -> None:
    plist = Path.home() / "Library" / "LaunchAgents" / "com.daylink.sparsify.plist"
    if not plist.exists():
        log("update complete. Restart any running 'sparsify serve' to load it.")
        return
    try:
        subprocess.run(["launchctl", "kickstart", "-k",
                        f"gui/{os.getuid()}/com.daylink.sparsify"],
                       capture_output=True, timeout=15)
        log("service restarted on the new version.")
    except (OSError, subprocess.SubprocessError):
        # older macOS without kickstart: unload/load
        subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
        subprocess.run(["launchctl", "load", str(plist)], capture_output=True)
        log("service reloaded on the new version.")
