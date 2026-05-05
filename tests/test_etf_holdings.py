"""Tests for the yfinance ETF-holdings adapter.

The motivating gap: SPY's Fundamentals report had no sector weights, top
holdings, or concentration metric — so the analyst had to write about an
ETF as if it were a single company. This module's adapter pulls those
through ``yfinance.Ticker(symbol).funds_data`` and renders a Markdown
report with the same failure-string contract as every other vendor.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tradingagents.dataflows import etf_holdings
from tradingagents.dataflows.config import set_config


# --- Fixtures --------------------------------------------------------------


def _spy_holdings_df() -> pd.DataFrame:
    """Mirrors yfinance's actual top_holdings shape for SPY (probed live)."""
    return pd.DataFrame(
        {
            "Name": [
                "NVIDIA Corp", "Apple Inc", "Microsoft Corp", "Amazon.com Inc",
                "Alphabet Inc Class A", "Broadcom Inc", "Alphabet Inc Class C",
                "Meta Platforms Inc Class A", "Tesla Inc",
                "Berkshire Hathaway Inc Class B",
            ],
            "Holding Percent": [
                0.075581, 0.066450, 0.049022, 0.036286, 0.029857, 0.026172,
                0.023927, 0.022318, 0.018645, 0.015670,
            ],
        },
        index=pd.Index(
            ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
             "AVGO", "GOOG", "META", "TSLA", "BRK-B"],
            name="Symbol",
        ),
    )


def _spy_funds_data_snapshot() -> dict:
    return {
        "asset_classes": {
            "cashPosition": 0.003,
            "stockPosition": 0.997,
            "bondPosition": 0.0,
            "preferredPosition": 0.0,
        },
        "sector_weightings": {
            "realestate": 0.0195,
            "consumer_cyclical": 0.1,
            "basic_materials": 0.019,
            "consumer_defensive": 0.0525,
            "technology": 0.336,
            "communication_services": 0.105,
            "financial_services": 0.124,
            "utilities": 0.0254,
            "industrials": 0.0847,
            "energy": 0.0402,
            "healthcare": 0.0947,
        },
        "top_holdings": _spy_holdings_df(),
        "fund_overview": {
            "categoryName": "Large Blend",
            "family": "State Street",
            "legalType": "Exchange Traded Fund",
        },
    }


# --- looks_like_etf heuristic ----------------------------------------------


def test_looks_like_etf_true_when_sectors_present():
    snap = {"sector_weightings": {"technology": 0.3}, "top_holdings": None}
    assert etf_holdings._looks_like_etf(snap) is True


def test_looks_like_etf_true_when_holdings_present():
    snap = {"sector_weightings": None, "top_holdings": _spy_holdings_df()}
    assert etf_holdings._looks_like_etf(snap) is True


def test_looks_like_etf_false_for_empty_data():
    """yfinance returns empty dicts/DataFrames for non-fund tickers."""
    snap = {
        "sector_weightings": {},
        "top_holdings": pd.DataFrame(),
        "asset_classes": {"stockPosition": 1.0},
    }
    assert etf_holdings._looks_like_etf(snap) is False


# --- Sector / asset-class formatting ---------------------------------------


def test_sector_label_handles_known_quirks():
    assert etf_holdings._sector_label("realestate") == "Real Estate"
    assert etf_holdings._sector_label("consumer_cyclical") == "Consumer Cyclical"
    # Unknown keys fall through to title-case
    assert etf_holdings._sector_label("foo_bar") == "Foo Bar"


def test_format_sector_weights_sorts_descending_and_formats_pct():
    sw = {"technology": 0.336, "energy": 0.04, "realestate": 0.0195}
    out = etf_holdings._format_sector_weights(sw)
    assert out[0] == "| Technology | 33.60% |"
    assert out[1] == "| Energy | 4.00% |"
    assert out[2] == "| Real Estate | 1.95% |"


def test_format_sector_weights_drops_zero_and_negative():
    sw = {"technology": 0.5, "ghost": 0.0, "neg": -0.01}
    out = etf_holdings._format_sector_weights(sw)
    assert len(out) == 1
    assert "Technology" in out[0]


def test_format_asset_classes_drops_zero_positions():
    """SPY has 0% bonds and 0% preferred — those rows should not appear."""
    ac = {
        "stockPosition": 0.997, "cashPosition": 0.003,
        "bondPosition": 0.0, "preferredPosition": 0.0,
    }
    out = etf_holdings._format_asset_classes(ac)
    joined = "\n".join(out)
    assert "Stocks" in joined and "Cash" in joined
    assert "Bonds" not in joined and "Preferred" not in joined


# --- Top-holdings formatting -----------------------------------------------


def test_format_top_holdings_renders_table_and_total():
    rows, total = etf_holdings._format_top_holdings(_spy_holdings_df(), top_n=10)
    assert rows[0] == "| # | Symbol | Name | Weight |"
    # Order preserved (first row is NVDA per the live snapshot)
    assert "NVDA" in rows[2] and "NVIDIA Corp" in rows[2]
    # Total weight in (0, 1)
    assert total is not None and 0.3 < total < 0.5


def test_format_top_holdings_handles_none_and_empty():
    assert etf_holdings._format_top_holdings(None, top_n=10) == ([], None)
    assert etf_holdings._format_top_holdings(pd.DataFrame(), top_n=10) == ([], None)


def test_format_top_holdings_caps_at_top_n():
    rows, _ = etf_holdings._format_top_holdings(_spy_holdings_df(), top_n=3)
    # 1 header + 1 separator + 3 data rows = 5 rows
    assert len(rows) == 5


# --- End-to-end formatting -------------------------------------------------


def test_format_report_includes_all_sections_for_spy_shape():
    out = etf_holdings._format_report("SPY", _spy_funds_data_snapshot())
    assert "## ETF Holdings for SPY" in out
    assert "yfinance funds_data" in out
    # Overview compact line
    assert "Large Blend" in out and "State Street" in out
    # Asset class
    assert "### Asset-Class Breakdown" in out
    assert "Stocks" in out and "99.7%" in out
    # Sectors (technology biggest)
    assert "### Sector Weights" in out
    sectors_block = out.split("### Sector Weights")[1].split("###")[0]
    assert sectors_block.index("Technology") < sectors_block.index("Energy")
    # Top holdings + concentration
    assert "### Top 10 Holdings" in out
    assert "NVDA" in out and "AAPL" in out
    assert "Concentration" in out
    assert "of fund AUM" in out


def test_format_report_returns_unavailable_when_nothing_renders():
    """A snapshot with empty/missing data should not produce a half-empty
    section — fall back to the bracketed unavailable string instead."""
    out = etf_holdings._format_report(
        "X",
        {"asset_classes": {}, "sector_weightings": {}, "top_holdings": None,
         "fund_overview": {}},
    )
    assert out.startswith("[ETF holdings unavailable")


# --- Public-API behaviour --------------------------------------------------


def test_rejects_unsafe_ticker(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    out = etf_holdings.get_etf_holdings("../../etc/passwd")
    assert out.startswith("[ETF holdings unavailable")


def test_returns_unavailable_when_yfinance_raises(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})

    def boom(ticker):
        raise RuntimeError("yfinance 503")

    monkeypatch.setattr(etf_holdings, "_fetch", boom)
    out = etf_holdings.get_etf_holdings("SPY")
    assert out.startswith("[ETF holdings unavailable")
    assert "yfinance 503" in out


def test_returns_unavailable_for_non_etf_ticker(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    # Empty funds_data → not an ETF
    monkeypatch.setattr(
        etf_holdings, "_fetch",
        lambda t: {"sector_weightings": {}, "top_holdings": pd.DataFrame(),
                   "asset_classes": {}, "fund_overview": {}},
    )
    out = etf_holdings.get_etf_holdings("AAPL")
    assert out.startswith("[ETF holdings unavailable")
    assert "not an ETF" in out


def test_renders_report_for_etf_ticker(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(
        etf_holdings, "_fetch",
        lambda t: _spy_funds_data_snapshot(),
    )
    out = etf_holdings.get_etf_holdings("SPY")
    assert "## ETF Holdings for SPY" in out
    assert "Technology" in out
    assert "NVDA" in out


def test_caches_result_within_ttl(monkeypatch, tmp_path):
    """A second call should hit the file cache, not yfinance."""
    set_config({"data_cache_dir": str(tmp_path)})
    call_count = {"n": 0}

    def fetch(t):
        call_count["n"] += 1
        return _spy_funds_data_snapshot()

    monkeypatch.setattr(etf_holdings, "_fetch", fetch)
    a = etf_holdings.get_etf_holdings("SPY")
    b = etf_holdings.get_etf_holdings("SPY")
    assert a == b
    assert call_count["n"] == 1


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
def test_get_etf_holdings_live_spy(tmp_path):
    """yfinance is unauthenticated; this lives behind the integration
    marker only because it makes a network call. It should always
    return *something* — either a real report or the bracketed
    unavailable string — and never raise."""
    set_config({"data_cache_dir": str(tmp_path)})
    out = etf_holdings.get_etf_holdings("SPY")
    assert isinstance(out, str) and out
    assert out.startswith("##") or out.startswith("[ETF holdings unavailable")
