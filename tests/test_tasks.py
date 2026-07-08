"""Scheduled-task logic: timezone-aware due computation, once-per-day, days."""
from datetime import datetime, timezone

from sparsify.runtime import tasks as T
from sparsify.runtime.tasks import Task, is_due


def _utc(y, mo, d, h, mi):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


def test_due_at_scheduled_time_in_tz():
    # 10:00 Asia/Kolkata == 04:30 UTC
    t = Task(prompt="x", model="m", hour=10, minute=0, tz="Asia/Kolkata")
    assert is_due(t, _utc(2026, 7, 9, 4, 30))          # exactly 10:00 IST
    assert is_due(t, _utc(2026, 7, 9, 5, 0))           # 10:30 IST (poll after)
    assert not is_due(t, _utc(2026, 7, 9, 4, 0))       # 09:30 IST (before)


def test_not_due_twice_same_day():
    t = Task(prompt="x", model="m", hour=10, minute=0, tz="Asia/Kolkata")
    assert is_due(t, _utc(2026, 7, 9, 4, 30))
    t.last_run = "2026-07-09"                            # already ran today (IST)
    assert not is_due(t, _utc(2026, 7, 9, 5, 0))
    # ...but due again the next day
    t2 = Task(prompt="x", model="m", hour=10, minute=0, tz="Asia/Kolkata",
              last_run="2026-07-09")
    assert is_due(t2, _utc(2026, 7, 10, 4, 30))


def test_day_of_week_filter():
    # 2026-07-09 is a Thursday; task only on mon/wed/fri
    t = Task(prompt="x", model="m", hour=10, minute=0, tz="Asia/Kolkata",
             days="mon,wed,fri")
    assert not is_due(t, _utc(2026, 7, 9, 4, 30))       # Thursday
    assert is_due(t, _utc(2026, 7, 10, 4, 30))          # Friday


def test_bad_timezone_falls_back():
    t = Task(prompt="x", model="m", hour=10, minute=0, tz="Not/AZone")
    # must not raise; treats now as UTC
    is_due(t, _utc(2026, 7, 9, 10, 0))


def test_crud_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    t = Task(prompt="pull leads and email them", model="qwen:30b",
             hour=10, minute=0, tz="Asia/Kolkata")
    T.add_task(t)
    assert [x.id for x in T.list_tasks()] == [t.id]
    assert T.get_task(t.id[:4]).prompt.startswith("pull leads")
    assert T.remove_task(t.id) and T.list_tasks() == []


def test_cli_task_add_list(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from sparsify.cli import main
    import sparsify.runtime.model_registry as reg

    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    monkeypatch.setattr("sparsify.cli.all_models",
                        lambda: [{"hf_id": "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit",
                                  "available": True, "size_gb": 16.3}])
    r = CliRunner().invoke(main, ["task", "add", "do a thing", "--at", "10:00",
                                  "--tz", "Asia/Kolkata"])
    assert r.exit_code == 0, r.output
    assert "scheduled" in r.output
    r2 = CliRunner().invoke(main, ["task", "list"])
    assert "do a thing" in r2.output


def test_cli_task_add_rejects_bad_time(tmp_path, monkeypatch):
    from click.testing import CliRunner
    from sparsify.cli import main
    monkeypatch.setenv("SPARSIFY_HOME", str(tmp_path))
    r = CliRunner().invoke(main, ["task", "add", "x", "--at", "25:99"])
    assert r.exit_code != 0 and "HH:MM" in r.output
