"""Scheduled autonomous agent tasks — the generic engine behind
"do <anything> every day at <time>".

A task is a plain-English instruction + a schedule + a tool grant. At its
time the runtime runs the instruction through the model's agent loop
(``SparsifyEngine.agent_stream``) with whatever tools are enabled, and
writes a timestamped report to the workspace. Nothing is domain-specific:
"pull leads and email them", "summarize my inbox", "back up my notes" are
all just instructions + tools. Delivery (email, a sheet, a file) is
whatever the instruction asks and the enabled tools can do.

Storage: ~/.sparsify/tasks.json. Scheduling: a launchd job calls
``sparsify task run-due`` periodically; each task fires once per day in
its own timezone. Unattended runs default to read+write tools (no shell)
for safety; --allow-shell opts in per task.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo


def _home() -> Path:
    return Path(os.environ.get("SPARSIFY_HOME", str(Path.home() / ".sparsify")))


def _tasks_file() -> Path:
    return _home() / "tasks.json"


def _reports_dir() -> Path:
    d = _home() / "reports"
    d.mkdir(parents=True, exist_ok=True)
    return d


@dataclass
class Task:
    prompt: str
    model: str
    hour: int = 10
    minute: int = 0
    tz: str = "Asia/Kolkata"
    days: str = "daily"                 # "daily" or comma list: "mon,wed,fri"
    allow_shell: bool = False
    workspace: str | None = None
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: str = ""
    last_run: str = ""                  # ISO date (in tz) of last completed run

    def schedule_str(self) -> str:
        return f"{self.hour:02d}:{self.minute:02d} {self.tz} ({self.days})"


def _load() -> list[Task]:
    f = _tasks_file()
    if not f.exists():
        return []
    try:
        return [Task(**t) for t in json.loads(f.read_text())]
    except (json.JSONDecodeError, TypeError, OSError):
        return []


def _save(tasks: list[Task]) -> None:
    _home().mkdir(parents=True, exist_ok=True)
    _tasks_file().write_text(json.dumps([asdict(t) for t in tasks], indent=2))


def list_tasks() -> list[Task]:
    return _load()


def add_task(task: Task) -> Task:
    if not task.created:
        pass  # timestamp stamped by the caller (no clock in this module for tests)
    tasks = _load()
    tasks.append(task)
    _save(tasks)
    return task


def remove_task(task_id: str) -> bool:
    tasks = _load()
    kept = [t for t in tasks if t.id != task_id and not t.id.startswith(task_id)]
    if len(kept) == len(tasks):
        return False
    _save(kept)
    return True


def get_task(task_id: str) -> Task | None:
    for t in _load():
        if t.id == task_id or t.id.startswith(task_id):
            return t
    return None


_DAYS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def is_due(task: Task, now_utc: datetime) -> bool:
    """True if *task* should run at *now_utc* and hasn't run yet today (in
    its own timezone). Designed for a coarse poll (e.g. every 15 min): it
    fires on the first poll at/after the scheduled time each day."""
    try:
        local = now_utc.astimezone(ZoneInfo(task.tz))
    except Exception:
        local = now_utc
    if task.days != "daily":
        wanted = {d.strip().lower()[:3] for d in task.days.split(",")}
        if _DAYS[local.weekday()] not in wanted:
            return False
    today = local.date().isoformat()
    if task.last_run == today:
        return False
    scheduled = local.replace(hour=task.hour, minute=task.minute,
                              second=0, microsecond=0)
    return local >= scheduled


def _mark_run(task: Task) -> None:
    try:
        local = datetime.now(timezone.utc).astimezone(ZoneInfo(task.tz))
    except Exception:
        local = datetime.now(timezone.utc)
    task.last_run = local.date().isoformat()
    tasks = _load()
    for i, t in enumerate(tasks):
        if t.id == task.id:
            tasks[i] = task
    _save(tasks)


def run_task(task: Task, log=print, stamp: str | None = None) -> Path:
    """Execute a task's instruction through the agent loop and write a
    report. Returns the report path. Loads the model fresh (scheduled runs
    are independent processes)."""
    from sparsify.runtime.model_registry import resolve_local
    from sparsify.runtime.chat_generation import SparsifyEngine
    from sparsify.runtime.tools import ToolPolicy

    resolved = resolve_local(task.model)
    if resolved is None:
        raise RuntimeError(f"model '{task.model}' is not on this machine "
                           f"(sparsify pull {task.model})")
    hf_id, model_path = resolved
    ws = Path(task.workspace) if task.workspace else None
    policy = ToolPolicy.from_flags(agent=True, workspace=ws,
                                   allow_shell=task.allow_shell)

    log(f"running task {task.id}: {task.prompt[:60]}…")
    engine = SparsifyEngine(model_path, memory_limit_gb=None)
    messages = [{"role": "user", "content": task.prompt}]
    parts, tools_used = [], []
    for kind, payload, _tel in engine.agent_stream(messages, policy=policy,
                                                   max_rounds=12):
        if kind == "text":
            parts.append(payload)
        elif kind == "tool":
            tools_used.append(payload["name"])
            log(f"  tool: {payload['name']}")
    answer = "".join(parts).strip()

    stamp = stamp or datetime.now(timezone.utc).astimezone(
        _safe_zone(task.tz)).strftime("%Y-%m-%d_%H%M")
    report = _reports_dir() / f"{stamp}_{task.id}.md"
    report.write_text(
        f"# Sparsify task report\n\n"
        f"- task: {task.id}\n- model: {hf_id}\n- when: {stamp} ({task.tz})\n"
        f"- tools used: {', '.join(tools_used) or 'none'}\n\n"
        f"## Instruction\n\n{task.prompt}\n\n## Result\n\n{answer}\n")
    _mark_run(task)
    log(f"  report: {report}")
    return report


def run_due(log=print) -> list[Path]:
    now = datetime.now(timezone.utc)
    done = []
    for task in _load():
        if is_due(task, now):
            try:
                done.append(run_task(task, log=log))
            except Exception as exc:  # one bad task must not stop the rest
                log(f"  task {task.id} failed: {exc}")
    return done


def _safe_zone(tz: str):
    try:
        return ZoneInfo(tz)
    except Exception:
        return timezone.utc
