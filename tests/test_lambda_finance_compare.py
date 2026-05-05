"""Tests for the Lambda Finance peer-comparison adapter.

The live API returns a flat top-level array — one dict per ticker with
metric names as keys. Tickers Lambda has no data for are silently dropped
from the response, so the adapter has to surface that explicitly.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from tradingagents.dataflows import lambda_finance_compare
from tradingagents.dataflows.config import set_config


# --- Envelope extraction ---------------------------------------------------


def test_extract_rows_handles_top_level_list():
    rows = [
        {"ticker": "AAPL", "revenue": 1.0},
        {"ticker": "MSFT", "revenue": 2.0},
    ]
    assert lambda_finance_compare._extract_rows(rows) == rows


def test_extract_rows_handles_data_envelope():
    rows = [{"ticker": "AAPL"}]
    assert lambda_finance_compare._extract_rows({"data": rows}) == rows
    assert lambda_finance_compare._extract_rows({"data": {"results": rows}}) == rows


def test_extract_rows_returns_empty_on_garbage():
    assert lambda_finance_compare._extract_rows("nope") == []
    assert lambda_finance_compare._extract_rows({"status": "err"}) == []
    assert lambda_finance_compare._extract_rows(None) == []


# --- Input validation -----------------------------------------------------


def test_parse_ticker_csv_dedups_and_uppercases():
    assert lambda_finance_compare._parse_ticker_csv("msft, GOOGL ,msft,AMZN") == [
        "MSFT", "GOOGL", "AMZN",
    ]


def test_parse_ticker_csv_rejects_unsafe_components():
    with pytest.raises(ValueError):
        lambda_finance_compare._parse_ticker_csv("MSFT,../../etc/passwd")


def test_parse_ticker_csv_caps_at_max_peers():
    too_many = ",".join(f"T{i}" for i in range(lambda_finance_compare._MAX_PEERS + 5))
    with pytest.raises(ValueError, match="too many peers"):
        lambda_finance_compare._parse_ticker_csv(too_many)


def test_parse_metrics_falls_back_to_defaults_when_empty():
    out = lambda_finance_compare._parse_metrics("")
    assert out == list(lambda_finance_compare._DEFAULT_METRICS)
    assert lambda_finance_compare._parse_metrics(None) == list(lambda_finance_compare._DEFAULT_METRICS)


def test_parse_metrics_dedups_and_lowers():
    assert lambda_finance_compare._parse_metrics("Revenue, NET_INCOME, revenue") == [
        "revenue", "net_income",
    ]


def test_parse_metrics_caps_at_max():
    raw = ",".join(f"m{i}" for i in range(lambda_finance_compare._MAX_METRICS + 5))
    out = lambda_finance_compare._parse_metrics(raw)
    assert len(out) == lambda_finance_compare._MAX_METRICS


# --- Formatting -----------------------------------------------------------


def test_metric_label_uses_known_pretty_names():
    assert lambda_finance_compare._metric_label("net_income") == "Net Income"
    assert lambda_finance_compare._metric_label("eps_diluted") == "EPS (Diluted)"


def test_metric_label_falls_back_to_title_case_for_unknown():
    assert lambda_finance_compare._metric_label("free_cash_flow") == "Free Cash Flow"


def test_format_table_renders_live_shape():
    """Snapshot of the actual /api/sec/compare response shape we probed."""
    rows = [
        {
            "ticker": "AAPL", "company_name": "Apple Inc.",
            "revenue": 391_035_000_000.0, "net_income": 93_736_000_000.0,
            "gross_profit": 180_683_000_000.0,
        },
        {
            "ticker": "MSFT", "company_name": "MICROSOFT CORPORATION",
            "revenue": 245_122_000_000.0, "net_income": 88_136_000_000.0,
            "gross_profit": 171_008_000_000.0,
        },
    ]
    out = lambda_finance_compare._format_table(
        primary="AAPL",
        requested=["AAPL", "MSFT", "GOOGL"],
        rows=rows,
        metric_keys=["revenue", "net_income", "gross_profit"],
        fiscal_year=2024,
    )
    assert "## Peer Comparison — AAPL vs MSFT, GOOGL (FY2024)" in out
    assert "Lambda Finance (SEC)" in out
    # Header preserves requested order
    header = next(l for l in out.splitlines() if l.startswith("| Metric"))
    assert header.index("AAPL") < header.index("MSFT") < header.index("GOOGL")
    # Cells render in scaled money
    assert "$391.04B" in out and "$93.74B" in out and "$245.12B" in out
    # GOOGL was missing — em-dash + footer note
    assert "— |" in out
    assert "no FY2024 data for GOOGL" in out


def test_format_table_emits_all_present_note_when_no_missing():
    rows = [
        {"ticker": "AAPL", "revenue": 1_000_000_000.0, "net_income": 100_000_000.0},
        {"ticker": "MSFT", "revenue": 2_000_000_000.0, "net_income": 200_000_000.0},
    ]
    out = lambda_finance_compare._format_table(
        primary="AAPL",
        requested=["AAPL", "MSFT"],
        rows=rows,
        metric_keys=["revenue", "net_income"],
        fiscal_year=2024,
    )
    assert "All 2 requested tickers had data" in out
    assert "no FY2024 data for" not in out


def test_format_table_preserves_requested_order_regardless_of_response():
    """Lambda may return rows in any order — the table must still show
    tickers in the order the caller asked for."""
    rows = [
        {"ticker": "MSFT", "revenue": 2.0e9},
        {"ticker": "AAPL", "revenue": 1.0e9},
    ]
    out = lambda_finance_compare._format_table(
        primary="AAPL",
        requested=["AAPL", "MSFT"],
        rows=rows,
        metric_keys=["revenue"],
        fiscal_year=2024,
    )
    header = next(l for l in out.splitlines() if l.startswith("| Metric"))
    assert header.index("AAPL") < header.index("MSFT")


# --- Public-API behaviour --------------------------------------------------


def test_returns_bracketed_string_when_no_api_key(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": ""})
    out = lambda_finance_compare.get_peer_comparison(
        "AAPL", "MSFT,GOOGL", "revenue,net_income", 2024
    )
    assert out.startswith("[Peer comparison unavailable")
    assert "LAMBDA_FINANCE_API_KEY not set" in out


def test_returns_bracketed_string_when_no_peers(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test"})
    out = lambda_finance_compare.get_peer_comparison("AAPL", "", "revenue", 2024)
    assert out.startswith("[Peer comparison unavailable")
    assert "no peers supplied" in out


def test_rejects_unsafe_primary_ticker(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test"})
    out = lambda_finance_compare.get_peer_comparison(
        "../../etc/passwd", "MSFT", "revenue", 2024
    )
    assert out.startswith("[Peer comparison unavailable")


def test_returns_bracketed_string_on_http_error(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test"})

    def fake_get(*args, **kwargs):
        raise RuntimeError("boom 503")

    monkeypatch.setattr(lambda_finance_compare.requests, "get", fake_get)
    out = lambda_finance_compare.get_peer_comparison("AAPL", "MSFT", "revenue", 2024)
    assert out.startswith("[Peer comparison unavailable")
    assert "boom 503" in out


def test_year_zero_defaults_to_last_calendar_year(monkeypatch, tmp_path):
    """year=0 must be treated as 'use sensible default' — never sent verbatim."""
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test"})
    captured = {}

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = [
        {"ticker": "AAPL", "revenue": 1.0e9},
    ]

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["params"] = params or {}
        return fake_response

    monkeypatch.setattr(lambda_finance_compare.requests, "get", fake_get)
    lambda_finance_compare.get_peer_comparison("AAPL", "MSFT", "revenue", 0)

    from datetime import datetime
    expected_year = datetime.utcnow().year - 1
    assert captured["params"]["year"] == expected_year


def test_renders_when_api_returns_rows(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test"})
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = [
        {"ticker": "AAPL", "revenue": 391_035_000_000.0, "net_income": 93_736_000_000.0},
        {"ticker": "MSFT", "revenue": 245_122_000_000.0, "net_income": 88_136_000_000.0},
    ]

    def fake_get(url, params=None, headers=None, timeout=None):
        assert url == lambda_finance_compare._URL
        assert headers and headers.get("X-API-Key") == "test"
        assert params and params["tickers"] == "AAPL,MSFT"
        assert params["year"] == 2024
        return fake_response

    monkeypatch.setattr(lambda_finance_compare.requests, "get", fake_get)
    out = lambda_finance_compare.get_peer_comparison(
        "AAPL", "MSFT", "revenue,net_income", 2024
    )
    assert "## Peer Comparison — AAPL vs MSFT (FY2024)" in out
    assert "$391.04B" in out


def test_returns_bracketed_string_when_lambda_returns_no_rows(monkeypatch, tmp_path):
    """If Lambda silently drops every requested ticker, we must surface that
    rather than render an empty table."""
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test"})
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = []

    monkeypatch.setattr(
        lambda_finance_compare.requests, "get",
        lambda *a, **kw: fake_response,
    )
    out = lambda_finance_compare.get_peer_comparison(
        "OBSCURE", "ALSOOBSCURE", "revenue", 2024
    )
    assert out.startswith("[Peer comparison unavailable")
    assert "no rows for any of" in out


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("LAMBDA_FINANCE_API_KEY"), reason="LAMBDA_FINANCE_API_KEY unset")
def test_get_peer_comparison_live_aapl_vs_msft(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "lambda_finance_api_key": os.environ["LAMBDA_FINANCE_API_KEY"],
    })
    out = lambda_finance_compare.get_peer_comparison(
        "AAPL", "MSFT,GOOGL", "revenue,net_income,gross_profit", 2024
    )
    assert isinstance(out, str) and out
    assert out.startswith("##") or out.startswith("[Peer comparison unavailable")
