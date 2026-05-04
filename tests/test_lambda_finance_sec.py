"""Unit tests for the Lambda Finance SEC fundamentals adapter.

Live response shape isn't pinned in Lambda's public docs, so the parser is
field-tolerant; these tests cover the documented envelope, several
plausible alternates, missing-key fallback, and the look-ahead-bias filter.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from tradingagents.dataflows import lambda_finance_sec
from tradingagents.dataflows.config import set_config


# --- Envelope extraction ---------------------------------------------------


def test_extract_rows_handles_top_level_list():
    rows = [{"fiscalDateEnding": "2024-12-31"}]
    assert lambda_finance_sec._extract_rows(rows) == rows


def test_extract_rows_handles_data_list_envelope():
    rows = [{"fiscalDateEnding": "2024-12-31"}]
    assert lambda_finance_sec._extract_rows({"status": "ok", "data": rows}) == rows


def test_extract_rows_handles_data_dict_with_reports_key():
    rows = [{"fiscalDateEnding": "2024-12-31"}]
    body = {"status": "ok", "data": {"reports": rows}}
    assert lambda_finance_sec._extract_rows(body) == rows


def test_extract_rows_merges_annual_and_quarterly_alpha_vantage_shape():
    annual = {"fiscalDateEnding": "2024-12-31", "totalRevenue": 1}
    quarterly = {"fiscalDateEnding": "2024-09-30", "totalRevenue": 2}
    body = {"data": {"annualReports": [annual], "quarterlyReports": [quarterly]}}
    rows = lambda_finance_sec._extract_rows(body)
    assert annual in rows and quarterly in rows


def test_extract_rows_returns_empty_on_garbage():
    assert lambda_finance_sec._extract_rows("nope") == []
    assert lambda_finance_sec._extract_rows({"status": "err"}) == []
    assert lambda_finance_sec._extract_rows(None) == []


# --- Look-ahead bias filter ------------------------------------------------


def test_filter_by_curr_date_drops_periods_after_cutoff():
    rows = [
        {"fiscalDateEnding": "2024-03-31"},
        {"fiscalDateEnding": "2024-09-30"},
        {"fiscalDateEnding": "2025-03-31"},
    ]
    out = lambda_finance_sec._filter_by_curr_date(rows, "2024-12-31")
    assert {r["fiscalDateEnding"] for r in out} == {"2024-03-31", "2024-09-30"}


def test_filter_by_curr_date_no_op_when_curr_date_missing():
    rows = [{"fiscalDateEnding": "2025-03-31"}]
    assert lambda_finance_sec._filter_by_curr_date(rows, None) == rows


def test_row_period_end_prefers_filing_date_then_falls_back():
    """``filing_date`` is preferred for Lambda Finance because their date
    fields (end_date / report_date) carry the prior-period comparative
    instead of the current period. Other aliases remain as fallbacks."""
    # filing_date wins even when end_date is present
    assert lambda_finance_sec._row_period_end({
        "filing_date": "2025-10-31", "end_date": "2023-09-30",
    }) == "2025-10-31"
    # Fallbacks for non-Lambda data sources
    assert lambda_finance_sec._row_period_end({"fiscalDateEnding": "2024-12-31"}) == "2024-12-31"
    assert lambda_finance_sec._row_period_end({"periodEndDate": "2024-09-30"}) == "2024-09-30"
    assert lambda_finance_sec._row_period_end({"end_date": "2024-06-30"}) == "2024-06-30"
    assert lambda_finance_sec._row_period_end({"report_date": "2024-03-31"}) == "2024-03-31"
    assert lambda_finance_sec._row_period_end({}) == ""


def test_row_period_label_uses_fiscal_year_and_period():
    assert lambda_finance_sec._row_period_label(
        {"fiscal_year": 2025, "fiscal_period": "FY"}
    ) == "FY2025"
    assert lambda_finance_sec._row_period_label(
        {"fiscal_year": 2026, "fiscal_period": "Q1"}
    ) == "Q1 FY2026"
    # Falls back to date when fiscal_year/period absent
    assert lambda_finance_sec._row_period_label(
        {"fiscalDateEnding": "2024-12-31"}
    ) == "2024-12-31"
    assert lambda_finance_sec._row_period_label({}) == "—"


# --- Money formatter -------------------------------------------------------


def test_money_chooses_scale_by_magnitude():
    assert lambda_finance_sec._money(2_500_000_000) == "$2.50B"
    assert lambda_finance_sec._money(750_000_000) == "$750.00M"
    assert lambda_finance_sec._money(500_000) == "$500.00k"
    assert lambda_finance_sec._money(-12_500_000) == "-$12.50M"
    # Per-share / small values keep two decimals
    assert lambda_finance_sec._money(2.34) == "$2.34"


def test_format_cell_handles_strings_and_none():
    assert lambda_finance_sec._format_cell(None) == "—"
    assert lambda_finance_sec._format_cell("") == "—"
    assert lambda_finance_sec._format_cell("None") == "—"
    # Numeric string gets formatted; non-numeric string passes through
    assert lambda_finance_sec._format_cell("1500000") == "$1.50M"
    assert lambda_finance_sec._format_cell("USD") == "USD"


# --- End-to-end formatting --------------------------------------------------


def test_format_statement_renders_income_statement_table():
    """Live Lambda shape: snake_case fields + fiscal_year/fiscal_period labels."""
    rows = [
        {
            "fiscal_year": 2025, "fiscal_period": "FY",
            "filing_date": "2025-10-31",
            "revenue": 416_161_000_000.0,
            "gross_profit": 195_201_000_000.0,
            "operating_income": 133_050_000_000.0,
            "net_income": 112_010_000_000.0,
            "eps_diluted": 7.46,
            "eps_basic": 7.49,
        },
        {
            "fiscal_year": 2024, "fiscal_period": "FY",
            "filing_date": "2024-11-01",
            "revenue": 391_035_000_000.0,
            "gross_profit": 180_683_000_000.0,
            "operating_income": 123_216_000_000.0,
            "net_income": 93_736_000_000.0,
            "eps_diluted": 6.08,
            "eps_basic": 6.11,
        },
    ]
    out = lambda_finance_sec._format_statement("AAPL", "Income Statement", "annual", rows)
    assert "## Income Statement for AAPL (annual)" in out
    assert "Lambda Finance (SEC)" in out
    # Most recent period first; column headers use fiscal-period labels
    header_line = next(l for l in out.splitlines() if l.startswith("| Line Item"))
    assert "FY2025" in header_line and "FY2024" in header_line
    assert header_line.index("FY2025") < header_line.index("FY2024")
    assert "$416.16B" in out and "$7.46" in out


def test_format_statement_renders_balance_sheet_with_live_lambda_shape():
    """Live Lambda balance-sheet shape: snake_case ``cash_and_equivalents``
    and ``stockholders_equity``, no separate cashAndCashEquivalents key."""
    rows = [
        {
            "fiscal_year": 2026, "fiscal_period": "Q1",
            "filing_date": "2026-01-30",
            "cash_and_equivalents": 45_317_000_000.0,
            "current_assets": 158_104_000_000.0,
            "current_liabilities": 162_367_000_000.0,
            "total_assets": 379_297_000_000.0,
            "total_liabilities": 291_107_000_000.0,
            "stockholders_equity": 88_190_000_000.0,
            "long_term_debt": 88_500_000_000.0,
            "short_term_debt": None,
        },
    ]
    out = lambda_finance_sec._format_statement("AAPL", "Balance Sheet", "quarterly", rows)
    assert "Q1 FY2026" in out
    assert "$45.32B" in out          # cash_and_equivalents
    assert "$88.19B" in out          # stockholders_equity (resolved by snake_case alias)
    assert "$158.10B" in out         # current_assets
    assert "$88.50B" in out          # long_term_debt
    # short_term_debt was null → emdash
    assert "| Short-Term Debt | — |" in out


def test_format_statement_handles_missing_fields_with_emdash():
    rows = [{"fiscalDateEnding": "2024-12-31", "totalRevenue": 1_000_000_000}]
    out = lambda_finance_sec._format_statement("XYZ", "Income Statement", "annual", rows)
    # Missing line items show "—"
    assert "| Net Income | — |" in out
    assert "| EPS (Diluted) | — |" in out


# --- Public-API behaviour ---------------------------------------------------


def test_get_income_statement_returns_bracketed_string_when_no_api_key(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": ""})
    out = lambda_finance_sec.get_income_statement("AAPL", freq="annual", curr_date="2024-12-31")
    assert out.startswith("[Income Statement unavailable")
    assert "LAMBDA_FINANCE_API_KEY not set" in out


def test_get_income_statement_rejects_unsafe_ticker(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test-key"})
    out = lambda_finance_sec.get_income_statement("../../../etc/passwd")
    assert out.startswith("[Income Statement unavailable")


def test_get_income_statement_returns_bracketed_string_on_http_error(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test-key"})

    def fake_get(*args, **kwargs):
        raise RuntimeError("boom 503")

    monkeypatch.setattr(lambda_finance_sec.requests, "get", fake_get)
    out = lambda_finance_sec.get_income_statement("AAPL")
    assert out.startswith("[Income Statement unavailable")
    assert "boom 503" in out


def test_get_income_statement_renders_when_api_returns_rows(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test-key"})

    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "status": "ok",
        "data": {
            "reports": [
                {
                    "fiscalDateEnding": "2024-12-31",
                    "totalRevenue": 100_000_000_000,
                    "netIncome": 25_000_000_000,
                    "dilutedEPS": 1.50,
                }
            ]
        },
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        # Make sure the request shape is right
        assert url.endswith("/income-statement/AAPL")
        assert headers and headers.get("X-API-Key") == "test-key"
        return fake_response

    monkeypatch.setattr(lambda_finance_sec.requests, "get", fake_get)
    out = lambda_finance_sec.get_income_statement("AAPL", freq="annual", curr_date="2025-01-01")
    assert "## Income Statement for AAPL (annual)" in out
    assert "$100.00B" in out


def test_get_income_statement_filters_lookahead_periods(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test-key"})
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "data": [
            {"fiscalDateEnding": "2024-09-30", "totalRevenue": 1, "netIncome": 1, "dilutedEPS": 0.1},
            {"fiscalDateEnding": "2025-09-30", "totalRevenue": 2, "netIncome": 2, "dilutedEPS": 0.2},
        ]
    }
    monkeypatch.setattr(
        lambda_finance_sec.requests, "get",
        lambda *a, **kw: fake_response,
    )
    out = lambda_finance_sec.get_income_statement("AAPL", freq="quarterly", curr_date="2024-12-31")
    # Only the in-range period should appear
    assert "2024-09-30" in out
    assert "2025-09-30" not in out


def test_get_balance_sheet_uses_correct_endpoint_and_omits_period(monkeypatch, tmp_path):
    """Balance-sheet endpoint doesn't accept ``period``; we must not send it."""
    set_config({"data_cache_dir": str(tmp_path), "lambda_finance_api_key": "test-key"})
    captured = {}
    fake_response = MagicMock()
    fake_response.raise_for_status = MagicMock()
    fake_response.json.return_value = {
        "data": [
            {"fiscalDateEnding": "2024-12-31", "totalAssets": 500_000_000_000,
             "totalLiabilities": 300_000_000_000, "totalEquity": 200_000_000_000}
        ]
    }

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params or {}
        return fake_response

    monkeypatch.setattr(lambda_finance_sec.requests, "get", fake_get)
    out = lambda_finance_sec.get_balance_sheet("AAPL", freq="annual")
    assert captured["url"].endswith("/balance-sheet/AAPL")
    assert "period" not in captured["params"]
    assert "## Balance Sheet for AAPL (annual)" in out
    assert "$500.00B" in out


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("LAMBDA_FINANCE_API_KEY"), reason="LAMBDA_FINANCE_API_KEY unset")
def test_get_income_statement_live_aapl(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "lambda_finance_api_key": os.environ["LAMBDA_FINANCE_API_KEY"],
    })
    out = lambda_finance_sec.get_income_statement("AAPL", freq="annual", curr_date="2025-12-31")
    assert isinstance(out, str) and out
    # Either real data or graceful no-data — never an unhandled crash
    assert out.startswith("##") or out.startswith("[Income Statement unavailable")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("LAMBDA_FINANCE_API_KEY"), reason="LAMBDA_FINANCE_API_KEY unset")
def test_get_balance_sheet_live_aapl(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "lambda_finance_api_key": os.environ["LAMBDA_FINANCE_API_KEY"],
    })
    out = lambda_finance_sec.get_balance_sheet("AAPL", freq="annual", curr_date="2025-12-31")
    assert isinstance(out, str) and out
    assert out.startswith("##") or out.startswith("[Balance Sheet unavailable")
