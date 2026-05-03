"""Per-day deduplication and run-queue construction."""

from __future__ import annotations

import logging
import random
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def already_run_today(ticker: str, results_dir: str | Path) -> bool:
    """Has ``ticker`` been analysed today (any wallclock time)?

    Looks for ``by_ticker/{TICKER}/{YYYYMMDD}_*_{TICKER}.json``.
    """
    today = datetime.now().strftime("%Y%m%d")
    ticker_dir = Path(results_dir) / "by_ticker" / ticker
    if not ticker_dir.is_dir():
        return False
    suffix = f"_{ticker}.json"
    return any(
        p.name.startswith(f"{today}_") and p.name.endswith(suffix)
        for p in ticker_dir.iterdir()
    )


def build_queue(
    tickers: list[str], results_dir: str | Path, max: int
) -> tuple[list[str], int]:
    """Filter today-already-run tickers, shuffle the remainder, cap at ``max``.

    Returns ``(queue, already_run_today_count)``. The caller can derive
    "deferred to next run" as ``len(tickers) - len(queue) - already_run``.
    """
    remaining = [t for t in tickers if not already_run_today(t, results_dir)]
    already_run = len(tickers) - len(remaining)
    random.shuffle(remaining)
    queue = remaining[:max]
    logger.info(
        "build_queue: %d total → %d after dedup (already-run %d) → %d after cap",
        len(tickers), len(remaining), already_run, len(queue),
    )
    return queue, already_run


def mark_complete(ticker: str, results_dir: str | Path) -> None:
    """Placeholder. Completion is inferred from result-file existence."""
    return None
