"""Screener queue manager: dedup-by-today + cap-by-max accounting."""

from datetime import datetime
from pathlib import Path

import pytest

from screener.queue_manager import already_run_today, build_queue


@pytest.fixture
def results_dir(tmp_path: Path) -> Path:
    (tmp_path / "by_ticker").mkdir()
    return tmp_path


def _mark_run_today(results_dir: Path, ticker: str) -> None:
    """Synthesise a today-stamped result file for ``ticker``."""
    today = datetime.now().strftime("%Y%m%d")
    d = results_dir / "by_ticker" / ticker
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{today}_120000_{ticker}.json").write_text("{}")


def test_already_run_today_detects_today_stamped_file(results_dir):
    _mark_run_today(results_dir, "AAPL")
    assert already_run_today("AAPL", results_dir) is True
    assert already_run_today("MSFT", results_dir) is False


def test_already_run_today_ignores_other_dates(results_dir):
    d = results_dir / "by_ticker" / "AAPL"
    d.mkdir(parents=True)
    (d / "20200101_120000_AAPL.json").write_text("{}")
    assert already_run_today("AAPL", results_dir) is False


def test_build_queue_returns_queue_and_already_run_count(results_dir):
    _mark_run_today(results_dir, "AAPL")
    _mark_run_today(results_dir, "MSFT")
    queue, already_run = build_queue(
        ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"], results_dir, max=10,
    )
    assert already_run == 2
    assert set(queue) == {"NVDA", "TSLA", "AMZN"}
    # max not hit when remaining ≤ max
    assert len(queue) == 3


def test_build_queue_caps_at_max_independent_of_dedup(results_dir):
    queue, already_run = build_queue(
        ["AAPL", "MSFT", "NVDA", "TSLA"], results_dir, max=2,
    )
    assert already_run == 0
    assert len(queue) == 2
    # Caller can derive deferred = 4 - 2 - 0 = 2
    deferred = 4 - len(queue) - already_run
    assert deferred == 2


def test_build_queue_with_all_already_run_returns_empty_queue(results_dir):
    for t in ("AAPL", "MSFT", "NVDA"):
        _mark_run_today(results_dir, t)
    queue, already_run = build_queue(
        ["AAPL", "MSFT", "NVDA"], results_dir, max=10,
    )
    assert queue == []
    assert already_run == 3


def test_build_queue_summary_math_adds_up(results_dir):
    """The pipeline summary's three buckets should partition the candidate list."""
    _mark_run_today(results_dir, "AAPL")        # 1 already run
    candidates = ["AAPL", "MSFT", "NVDA", "TSLA", "AMZN"]
    queue, already_run = build_queue(candidates, results_dir, max=2)

    deferred = len(candidates) - len(queue) - already_run
    assert already_run == 1
    assert len(queue) == 2
    assert deferred == 2  # 5 - 2 - 1

    # Whole partition must equal the candidate count (no double-counting).
    assert len(queue) + already_run + deferred == len(candidates)
