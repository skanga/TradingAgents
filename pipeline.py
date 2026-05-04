"""Screener pipeline entry point.

Pulls a candidate list from Finviz (or a hand-supplied list), filters out
tickers already analysed today, runs each through TradingAgentsGraph, and
persists each result both by ticker and (via symlink) by date.

Run with:

    python pipeline.py                          # screen Finviz, analyse top N
    python pipeline.py --tickers AAPL,MSFT      # bypass Finviz
    python pipeline.py --ticker-file watch.txt  # one ticker per line
    python pipeline.py --screen-only watch.txt  # write Finviz candidates to a file, no LLM calls
    python pipeline.py --dry-run                # preview the queue, no LLM calls
    python pipeline.py --rerun-today            # bypass already-run dedup
    python pipeline.py --max-tickers 3          # override config cap
    python pipeline.py --filter-overrides "Sector=Technology"
"""

from __future__ import annotations

import argparse
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
from screener.markdown_writer import render_markdown_report  # noqa: E402
from screener.queue_manager import build_queue, mark_complete  # noqa: E402

logger = logging.getLogger(__name__)


# --- CLI -------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description=(
            "Screener pipeline: Finviz candidates → TradingAgentsGraph batch. "
            "Results land under results/by_ticker/{TICKER}/ with date-symlinks."
        ),
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--tickers",
        metavar="LIST",
        help="Comma-separated ticker list, bypasses Finviz (e.g. AAPL,MSFT,NVDA).",
    )
    src.add_argument(
        "--ticker-file",
        metavar="PATH",
        help=(
            "Path to file with one ticker per line; lines starting with '#' "
            "are comments. Comma-separated lines are also OK. Bypasses Finviz."
        ),
    )
    src.add_argument(
        "--screen-only",
        metavar="PATH",
        help=(
            "Run the Finviz screener and write candidates to PATH (one ticker "
            "per line, with a header comment naming the filter), then exit. "
            "Honours --filter-overrides. Re-feed via --ticker-file after editing."
        ),
    )

    parser.add_argument(
        "--max-tickers",
        type=int,
        metavar="N",
        help=(
            "Override max_tickers_per_run (config default: "
            f"{CONFIG['max_tickers_per_run']}). Caps the queue after dedup."
        ),
    )
    parser.add_argument(
        "--run-id",
        metavar="ID",
        help=(
            "Identifier for this analysis run (default: a YYYY_MM_DD_HH_mm_ss "
            "timestamp). Names the folder under results/by_run/ where "
            "per-ticker symlinks and the run log land."
        ),
    )
    parser.add_argument(
        "--rerun-today",
        action="store_true",
        help="Bypass the today-already-run dedup; useful for retrying a partially failed batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve the queue and print it, then exit. No LLM calls, no writes.",
    )
    parser.add_argument(
        "--filter-overrides",
        metavar="K=V,K=V",
        help=(
            "Patch the Finviz filter dict from the command line "
            "(e.g. \"Sector=Technology,Price=Over $20\"). "
            "Honoured for the default Finviz run and --screen-only; "
            "ignored when --tickers / --ticker-file is set."
        ),
    )

    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    verbosity.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Reduce log output to WARNING and above.",
    )

    return parser


def _configure_logging(args: argparse.Namespace) -> None:
    if args.verbose:
        level = logging.DEBUG
    elif args.quiet:
        level = logging.WARNING
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _banner(title: str, *, char: str = "=", width: int = 70) -> None:
    """Print a section header to stdout."""
    line = char * width
    print(f"\n{line}\n  {title}\n{line}")


def _parse_ticker_list_arg(raw: str) -> list[str]:
    """Parse a comma-separated ticker string. Validates each via safe_ticker_component."""
    from tradingagents.dataflows.utils import safe_ticker_component
    out: list[str] = []
    seen: set[str] = set()
    for chunk in raw.split(","):
        sym = chunk.strip().upper()
        if not sym or sym in seen:
            continue
        try:
            safe_ticker_component(sym)
        except ValueError as e:
            raise SystemExit(f"--tickers: rejected {sym!r}: {e}")
        out.append(sym)
        seen.add(sym)
    logger.info("Parsed %d tickers from --tickers", len(out))
    return out


def _read_ticker_file(path: Path) -> list[str]:
    """Read one-per-line (or comma-on-line) ticker file. '#' starts a comment."""
    from tradingagents.dataflows.utils import safe_ticker_component
    if not path.is_file():
        raise SystemExit(f"--ticker-file: not a file: {path}")
    raw_lines = path.read_text(encoding="utf-8").splitlines()
    out: list[str] = []
    seen: set[str] = set()
    skipped_dupes = 0
    for raw_line in raw_lines:
        # Strip inline '#' comments and whitespace, then split on commas.
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        for chunk in line.split(","):
            sym = chunk.strip().upper()
            if not sym:
                continue
            if sym in seen:
                skipped_dupes += 1
                continue
            try:
                safe_ticker_component(sym)
            except ValueError as e:
                raise SystemExit(f"--ticker-file: rejected {sym!r} in {path}: {e}")
            out.append(sym)
            seen.add(sym)
    logger.info(
        "Read %d tickers from %s (%d raw lines, %d duplicates skipped)",
        len(out), path, len(raw_lines), skipped_dupes,
    )
    return out


def _parse_filter_overrides(raw: str) -> dict:
    """Parse ``KEY=VAL,KEY=VAL`` into a dict. Values may contain spaces."""
    out: dict = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise SystemExit(
                f"--filter-overrides: expected KEY=VAL, got {chunk!r}"
            )
        key, _, value = chunk.partition("=")
        key, value = key.strip(), value.strip()
        if not key:
            raise SystemExit(f"--filter-overrides: empty key in {chunk!r}")
        out[key] = value
    return out


def _resolve_candidates(args: argparse.Namespace) -> tuple[list[str], str]:
    """Return (candidates, source_label)."""
    if args.tickers:
        tickers = _parse_ticker_list_arg(args.tickers)
        return tickers, f"--tickers ({len(tickers)} candidates)"
    if args.ticker_file:
        path = Path(args.ticker_file)
        tickers = _read_ticker_file(path)
        return tickers, f"--ticker-file {path} ({len(tickers)} candidates)"
    filters = dict(CONFIG["finviz_filters"])
    if args.filter_overrides:
        filters.update(_parse_filter_overrides(args.filter_overrides))
    tickers = get_candidates(filters)
    return tickers, f"Finviz ({len(tickers)} candidates)"


def _do_screen_only(args: argparse.Namespace) -> None:
    """Resolve Finviz candidates, write them to ``args.screen_only``, exit."""
    filters = dict(CONFIG["finviz_filters"])
    if args.filter_overrides:
        filters.update(_parse_filter_overrides(args.filter_overrides))

    _banner("Finviz Screener (--screen-only)")
    print(f"  Filters    : {filters}")

    candidates = get_candidates(filters)
    if not candidates:
        logger.error("No candidates from Finviz; nothing written.")
        sys.exit(1)

    out_path = Path(args.screen_only).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# Finviz screener candidates ({len(candidates)})",
        f"# Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"# Filters: {filters}",
        "# Edit this list, then run:",
        f"#   python pipeline.py --ticker-file {out_path}",
        "",
    ]
    content = "\n".join(header + candidates) + "\n"
    out_path.write_text(content, encoding="utf-8")

    print(f"  Candidates : {len(candidates)}")
    print(f"  Output file: {out_path}")
    logger.info("Wrote %d candidates to %s", len(candidates), out_path)

    _banner("File contents", char="-")
    print(content, end="")
    _banner("Done", char="-")




# --- Main ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    _configure_logging(args)

    output_dir: Path = CONFIG["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.screen_only:
        _do_screen_only(args)
        return

    today = datetime.now().strftime("%Y-%m-%d")
    ta_cfg = CONFIG["tradingagents_config"]
    run_id = args.run_id or datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    run_dir = output_dir / "by_run" / run_id

    _banner(f"TradingAgents pipeline — {today}")

    candidates, source_label = _resolve_candidates(args)
    if not candidates:
        logger.error("No candidate tickers (source: %s); aborting.", source_label)
        sys.exit(1)

    max_tickers = (
        args.max_tickers
        if args.max_tickers is not None
        else CONFIG["max_tickers_per_run"]
    )
    queue, already_run = build_queue(
        candidates, output_dir, max=max_tickers, rerun_today=args.rerun_today,
    )
    deferred = len(candidates) - len(queue) - already_run

    print(f"  Source       : {source_label}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Provider     : {ta_cfg.get('llm_provider')}")
    print(f"  Deep think   : {ta_cfg.get('deep_think_llm')}")
    print(f"  Quick think  : {ta_cfg.get('quick_think_llm')}")
    print(f"  Max tickers  : {max_tickers}")
    print(f"  Queue size   : {len(queue)}")
    print(f"  Already run  : {already_run}")
    print(f"  Deferred(cap): {deferred}")
    if queue:
        print(f"  Tickers      : {', '.join(queue)}")

    if args.dry_run:
        _banner("Dry run — no LLM calls, no writes", char="-")
        return

    if not queue:
        logger.info("Nothing to do — queue is empty after dedup + cap.")
        _banner("Summary", char="-")
        print(
            f"  Analyzed         : 0\n"
            f"  Failed           : 0\n"
            f"  Already run today: {already_run}\n"
            f"  Deferred (cap)   : {deferred}"
        )
        return

    # Lazy import so the screener can fail early on Finviz before paying
    # the LangGraph compile cost.
    from tradingagents.graph.trading_graph import TradingAgentsGraph  # noqa: E402

    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Run dir      : {run_dir}")

    _banner("Analyzing tickers", char="-")
    analyzed = failed = 0

    for i, ticker in enumerate(queue, start=1):
        print(f"\n[{i}/{len(queue)}] Analyzing {ticker}...", flush=True)
        try:
            ta = TradingAgentsGraph(config=ta_cfg)
            state, decision = ta.propagate(ticker, today)
            json_path = save_result(
                ticker=ticker,
                state=state,
                decision=decision,
                results_dir=output_dir,
                run_dir=run_dir,
                ta=ta,
                trade_date=today,
            )
            md_path = json_path.with_suffix(".md")
            mark_complete(ticker, output_dir)
            analyzed += 1
            print(f"  → Decision : {decision}")
            print(f"  → JSON     : {json_path}")
            print(f"  → Markdown : {md_path}")
        except Exception as e:
            logger.exception("Analysis failed for %s: %s", ticker, e)
            failed += 1
            print(f"  → FAILED   : {e}")
            continue

    _banner("Summary", char="-")
    print(
        f"  Analyzed         : {analyzed}\n"
        f"  Failed           : {failed}\n"
        f"  Already run today: {already_run}\n"
        f"  Deferred (cap)   : {deferred}"
    )


def save_result(
    *,
    ticker: str,
    state: Any,
    decision: Any,
    results_dir: Path,
    run_dir: Path,
    ta: Any = None,
    trade_date: str | None = None,
) -> Path:
    """Persist one analysis to ``results/by_ticker/`` and link from ``run_dir``."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    by_ticker_dir = results_dir / "by_ticker" / ticker
    by_ticker_dir.mkdir(parents=True, exist_ok=True)
    by_ticker_file = by_ticker_dir / f"{ticker}_{timestamp}.json"

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

    # Markdown report: BLUF up top, every pipeline step rendered as a section.
    # Use clean_state (which now includes options_report / macro_snapshot /
    # iv_snapshot from the updated _log_state).
    md_text = render_markdown_report(
        ticker=ticker,
        trade_date=trade_date or "",
        state=clean_state if isinstance(clean_state, dict) else {},
        decision=decision,
    )
    by_ticker_md = by_ticker_dir / f"{ticker}_{timestamp}.md"
    by_ticker_md.write_text(md_text, encoding="utf-8")

    run_dir.mkdir(parents=True, exist_ok=True)
    by_run_link = run_dir / f"{ticker}_{timestamp}.json"
    by_run_md_link = run_dir / f"{ticker}_{timestamp}.md"
    _link_or_stub(target=by_ticker_file, link=by_run_link)
    _link_or_stub(target=by_ticker_md,   link=by_run_md_link)

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
