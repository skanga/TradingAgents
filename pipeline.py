"""Screener pipeline entry point.

Pulls a candidate list from Finviz, filters out tickers already analysed
today, runs each through TradingAgentsGraph.propagate, and persists each
result both by ticker and (via symlink) by date.

Run with:  ``python pipeline.py``
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# config.py loads .env BEFORE importing DEFAULT_CONFIG. Importing CONFIG
# here is enough to get the env-aware fork config built.
from config import CONFIG  # noqa: E402
from screener.finviz_filter import get_candidates  # noqa: E402
from screener.queue_manager import build_queue, mark_complete  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir: Path = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    tickers = get_candidates(CONFIG["finviz_filters"])
    if not tickers:
        logger.error("No tickers returned from Finviz; aborting run.")
        sys.exit(1)

    queue, already_run = build_queue(tickers, output_dir, CONFIG["max_tickers_per_run"])
    deferred = len(tickers) - len(queue) - already_run
    if not queue:
        logger.info("Nothing to do — every candidate was already run today.")
        print(
            f"\nSummary: 0 analyzed, 0 failed, "
            f"{already_run} already run today, {deferred} deferred (over cap)"
        )
        return

    # Lazy import so the screener can fail early on Finviz before paying
    # the LangGraph compile cost.
    from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

    today = datetime.now().strftime("%Y-%m-%d")
    analyzed = failed = 0

    for i, ticker in enumerate(queue, start=1):
        print(f"[{i}/{len(queue)}] Analyzing {ticker}...")
        try:
            ta = TradingAgentsGraph(config=CONFIG["tradingagents_config"])
            state, decision = ta.propagate(ticker, today)
            save_result(
                ticker=ticker,
                state=state,
                decision=decision,
                results_dir=output_dir,
                ta=ta,
                trade_date=today,
            )
            mark_complete(ticker, output_dir)
            analyzed += 1
        except Exception as e:
            logger.exception("Analysis failed for %s: %s", ticker, e)
            failed += 1
            continue

    print(
        f"\nSummary: {analyzed} analyzed, {failed} failed, "
        f"{already_run} already run today, {deferred} deferred (over cap)"
    )


def save_result(
    *,
    ticker: str,
    state: Any,
    decision: Any,
    results_dir: Path,
    ta: Any = None,
    trade_date: str | None = None,
) -> Path:
    """Persist one analysis to ``results/by_ticker/`` and link from ``results/by_date/``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    date_str = timestamp.split("_")[0]

    by_ticker_dir = results_dir / "by_ticker" / ticker
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    by_ticker_file = by_ticker_dir / f"{timestamp}_{ticker}.json"

    # Prefer the fork's pre-flattened JSON-safe state if available; otherwise
    # coerce the raw propagation state by stringifying any non-JSON values.
    log_states = getattr(ta, "log_states_dict", None)
    if log_states and trade_date and str(trade_date) in log_states:
        clean_state = log_states[str(trade_date)]
    else:
        clean_state = _coerce_jsonable(state)

    payload = {
        "ticker": ticker,
        "trade_date": trade_date,
        "decision": decision,
        "state": clean_state,
    }
    by_ticker_file.write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )

    by_date_dir = results_dir / "by_date" / date_str
    by_date_dir.mkdir(parents=True, exist_ok=True)
    by_date_link = by_date_dir / f"{ticker}_{timestamp}.json"
    _link_or_stub(target=by_ticker_file, link=by_date_link)

    return by_ticker_file


def _coerce_jsonable(obj: Any) -> Any:
    """Best-effort JSON coercion. Falls back to ``default=str`` for opaque values."""
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return json.loads(json.dumps(obj, default=str))


def _link_or_stub(*, target: Path, link: Path) -> None:
    """Create a relative symlink at ``link`` pointing to ``target``.

    On systems where symlinks fail (Windows without dev-mode/admin), writes
    a small JSON stub naming the relative target instead, with a warning.
    """
    if link.exists() or link.is_symlink():
        link.unlink()
    relative = os.path.relpath(target, link.parent)
    try:
        os.symlink(relative, link)
    except OSError as e:
        logger.warning(
            "symlink %s -> %s failed (%s); writing stub JSON instead",
            link, relative, e,
        )
        link.write_text(
            json.dumps({"_alias_of": relative}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
