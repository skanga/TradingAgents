"""CLI flag plumbing for the screener pipeline.

Targets the pure helpers (parsers, file readers) and the queue-manager
``rerun_today`` path. The full main() entry point is covered indirectly by
the existing queue_manager + markdown_writer tests; we keep this file
focused on the new surface.
"""

from datetime import datetime
from pathlib import Path

import pytest

import pipeline
from screener.queue_manager import build_queue


# --- _parse_ticker_list_arg ------------------------------------------------


def test_parse_ticker_list_arg_uppercases_and_strips():
    assert pipeline._parse_ticker_list_arg(" aapl, msft ,nvda") == ["AAPL", "MSFT", "NVDA"]


def test_parse_ticker_list_arg_dedupes_within_one_run():
    assert pipeline._parse_ticker_list_arg("AAPL,aapl,AAPL") == ["AAPL"]


def test_parse_ticker_list_arg_skips_blanks():
    assert pipeline._parse_ticker_list_arg(",,AAPL,,MSFT,") == ["AAPL", "MSFT"]


def test_parse_ticker_list_arg_rejects_path_traversal():
    with pytest.raises(SystemExit) as exc:
        pipeline._parse_ticker_list_arg("AAPL,../etc/passwd")
    assert "rejected" in str(exc.value)


def test_parse_ticker_list_arg_accepts_exchange_suffixes():
    assert pipeline._parse_ticker_list_arg("CNC.TO,7203.T,0700.HK") == [
        "CNC.TO", "7203.T", "0700.HK",
    ]


# --- _read_ticker_file -----------------------------------------------------


def test_read_ticker_file_one_per_line(tmp_path):
    p = tmp_path / "watch.txt"
    p.write_text("AAPL\nMSFT\nNVDA\n")
    assert pipeline._read_ticker_file(p) == ["AAPL", "MSFT", "NVDA"]


def test_read_ticker_file_strips_comments_and_blanks(tmp_path):
    p = tmp_path / "watch.txt"
    p.write_text(
        "# my watchlist\n"
        "AAPL  # the obvious one\n"
        "\n"
        "  # leading-space comment\n"
        "MSFT\n"
        "#NVDA  ← commented out, should NOT appear\n"
    )
    assert pipeline._read_ticker_file(p) == ["AAPL", "MSFT"]


def test_read_ticker_file_supports_comma_separated_lines(tmp_path):
    p = tmp_path / "watch.txt"
    p.write_text("AAPL, MSFT\nNVDA,GOOGL\n")
    assert pipeline._read_ticker_file(p) == ["AAPL", "MSFT", "NVDA", "GOOGL"]


def test_read_ticker_file_rejects_missing_path():
    with pytest.raises(SystemExit) as exc:
        pipeline._read_ticker_file(Path("/no/such/file.txt"))
    assert "not a file" in str(exc.value)


def test_read_ticker_file_rejects_invalid_ticker(tmp_path):
    p = tmp_path / "watch.txt"
    p.write_text("AAPL\n../etc/passwd\n")
    with pytest.raises(SystemExit) as exc:
        pipeline._read_ticker_file(p)
    assert "rejected" in str(exc.value)


# --- _parse_filter_overrides -----------------------------------------------


def test_parse_filter_overrides_basic():
    out = pipeline._parse_filter_overrides("Sector=Technology,Price=Over $20")
    assert out == {"Sector": "Technology", "Price": "Over $20"}


def test_parse_filter_overrides_strips_whitespace():
    out = pipeline._parse_filter_overrides("  Sector = Technology , Price = Over $20 ")
    assert out == {"Sector": "Technology", "Price": "Over $20"}


def test_parse_filter_overrides_skips_empty_chunks():
    out = pipeline._parse_filter_overrides("Sector=Technology,,Price=Over $20,")
    assert out == {"Sector": "Technology", "Price": "Over $20"}


def test_parse_filter_overrides_rejects_missing_equals():
    with pytest.raises(SystemExit) as exc:
        pipeline._parse_filter_overrides("Sector=Technology,GarbageNoEquals")
    assert "expected KEY=VAL" in str(exc.value)


def test_parse_filter_overrides_rejects_empty_key():
    with pytest.raises(SystemExit) as exc:
        pipeline._parse_filter_overrides("=value")
    assert "empty key" in str(exc.value)


# --- argparse wiring -------------------------------------------------------


def test_parser_accepts_all_documented_flags():
    parser = pipeline._build_parser()
    args = parser.parse_args([
        "--tickers", "AAPL,MSFT",
        "--max-tickers", "3",
        "--rerun-today",
        "--dry-run",
        "-v",
    ])
    assert args.tickers == "AAPL,MSFT"
    assert args.max_tickers == 3
    assert args.rerun_today is True
    assert args.dry_run is True
    assert args.verbose is True


def test_parser_rejects_tickers_and_ticker_file_together():
    parser = pipeline._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--tickers", "AAPL", "--ticker-file", "x.txt"])


def test_parser_rejects_verbose_and_quiet_together():
    parser = pipeline._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["-v", "-q"])


# --- _resolve_candidates routing -------------------------------------------


def test_resolve_candidates_uses_tickers_arg(monkeypatch):
    parser = pipeline._build_parser()
    args = parser.parse_args(["--tickers", "AAPL,MSFT"])
    # Ensure get_candidates is NOT called when --tickers is supplied.
    monkeypatch.setattr(
        pipeline, "get_candidates",
        lambda *a, **kw: pytest.fail("Finviz should not be queried with --tickers"),
    )
    cands, label = pipeline._resolve_candidates(args)
    assert cands == ["AAPL", "MSFT"]
    assert "--tickers" in label and "2 candidates" in label


def test_resolve_candidates_uses_ticker_file(monkeypatch, tmp_path):
    p = tmp_path / "w.txt"
    p.write_text("AAPL\nMSFT\n")
    parser = pipeline._build_parser()
    args = parser.parse_args(["--ticker-file", str(p)])
    monkeypatch.setattr(
        pipeline, "get_candidates",
        lambda *a, **kw: pytest.fail("Finviz should not be queried with --ticker-file"),
    )
    cands, label = pipeline._resolve_candidates(args)
    assert cands == ["AAPL", "MSFT"]
    assert "--ticker-file" in label


def test_resolve_candidates_calls_finviz_with_overrides(monkeypatch):
    captured = {}

    def fake_get(filters, *_, **__):
        captured["filters"] = filters
        return ["AAPL"]

    monkeypatch.setattr(pipeline, "get_candidates", fake_get)
    parser = pipeline._build_parser()
    args = parser.parse_args(["--filter-overrides", "Sector=Technology"])
    cands, label = pipeline._resolve_candidates(args)

    assert cands == ["AAPL"]
    assert captured["filters"]["Sector"] == "Technology"
    # Untouched defaults remain
    assert captured["filters"]["Country"] == "USA"
    assert "Finviz" in label


# --- queue_manager rerun_today bypass --------------------------------------


def _mark_run_today(results_dir: Path, ticker: str) -> None:
    today = datetime.now().strftime("%Y%m%d")
    d = results_dir / "by_ticker" / ticker
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{today}_120000_{ticker}.json").write_text("{}")


def test_build_queue_rerun_today_bypasses_dedup(tmp_path):
    _mark_run_today(tmp_path, "AAPL")
    _mark_run_today(tmp_path, "MSFT")
    queue, already_run = build_queue(
        ["AAPL", "MSFT", "NVDA"], tmp_path, max=10, rerun_today=True,
    )
    # AAPL + MSFT would normally be filtered out, but rerun_today=True keeps them.
    assert set(queue) == {"AAPL", "MSFT", "NVDA"}
    assert already_run == 0


def test_build_queue_default_keeps_dedup_behavior(tmp_path):
    _mark_run_today(tmp_path, "AAPL")
    queue, already_run = build_queue(["AAPL", "MSFT"], tmp_path, max=10)
    assert queue == ["MSFT"]
    assert already_run == 1
