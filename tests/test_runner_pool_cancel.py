from __future__ import annotations

from gui import storage
from service.runner_pool import ManagedRun, RunnerPool


class FakeHandle:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class EventlessHandle:
    def __init__(self, return_code: int, stderr: list[str] | None = None) -> None:
        self.return_code = return_code
        self.stderr_buf = stderr or []

    def poll_events(self) -> list[dict]:
        return []

    def is_running(self) -> bool:
        return False


def test_cancel_persists_run_error(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "gui.db")
    storage.init_db()
    storage.create_run(
        run_id="run-cancel",
        ticker="NVDA",
        trade_date="2026-05-10",
        provider="openai",
        deep_model="gpt-5.4",
        quick_model="gpt-5.4-mini",
        debate_rounds=1,
        risk_rounds=1,
        vendors={"core_stock_apis": "yfinance"},
    )

    handle = FakeHandle()
    pool = RunnerPool()
    managed = ManagedRun(run_id="run-cancel", handle=handle)  # type: ignore[arg-type]
    pool._runs["run-cancel"] = managed

    assert pool.cancel("run-cancel") is True

    row = storage.get_run("run-cancel")
    assert handle.cancelled is True
    assert row is not None
    assert row["status"] == "error"
    assert row["error_message"] == "Cancelled by user."


def test_reader_loop_finalizes_nonzero_exit_without_terminal_event(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "gui.db")
    storage.init_db()
    storage.create_run(
        run_id="run-eventless-error",
        ticker="NVDA",
        trade_date="2026-05-10",
        provider="openai",
        deep_model="gpt-5.4",
        quick_model="gpt-5.4-mini",
        debate_rounds=1,
        risk_rounds=1,
        vendors={"core_stock_apis": "yfinance"},
    )

    handle = EventlessHandle(return_code=7, stderr=["Traceback line\n", "fatal problem\n"])
    pool = RunnerPool()
    managed = ManagedRun(run_id="run-eventless-error", handle=handle)  # type: ignore[arg-type]

    pool._reader_loop(managed)

    row = storage.get_run("run-eventless-error")
    assert row is not None
    assert row["status"] == "error"
    assert "worker exited with code 7" in row["error_message"]
    assert "fatal problem" in row["error_message"]
    assert managed.finished is True
    assert managed.error == row["error_message"]


def test_reader_loop_finalizes_zero_exit_without_terminal_event(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "DB_PATH", tmp_path / "gui.db")
    storage.init_db()
    storage.create_run(
        run_id="run-eventless-zero",
        ticker="NVDA",
        trade_date="2026-05-10",
        provider="openai",
        deep_model="gpt-5.4",
        quick_model="gpt-5.4-mini",
        debate_rounds=1,
        risk_rounds=1,
        vendors={"core_stock_apis": "yfinance"},
    )

    handle = EventlessHandle(return_code=0)
    pool = RunnerPool()
    managed = ManagedRun(run_id="run-eventless-zero", handle=handle)  # type: ignore[arg-type]

    pool._reader_loop(managed)

    row = storage.get_run("run-eventless-zero")
    assert row is not None
    assert row["status"] == "error"
    assert row["error_message"] == "worker exited without a terminal event"
    assert managed.finished is True
