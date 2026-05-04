import os
from datetime import date, timedelta

import pytest

from tradingagents.dataflows import congress_trades
from tradingagents.dataflows.config import set_config


# --- Pure parser unit tests -------------------------------------------------


def test_parse_finnhub_row_maps_canonical_shape():
    row = {
        "name": "Pelosi, Nancy",
        "transactionDate": "2026-04-15",
        "transactionType": "Purchase",
        "position": "House",
        "amountFrom": 1_000_001,
        "amountTo": 5_000_000,
        "filingDate": "2026-04-30",
    }
    t = congress_trades._parse_finnhub_row(row)
    assert t is not None
    assert t["filer"] == "Pelosi, Nancy"
    assert t["chamber"] == "House"
    assert t["type"] == "Purchase"
    assert t["amount_min"] == 1_000_001
    assert t["amount_max"] == 5_000_000
    assert t["filing_lag_days"] == 15
    assert "$1.0M – $5.0M" == t["amount_label"]


def test_parse_finnhub_row_returns_none_on_garbage():
    # Missing/invalid date is tolerated by the parser, but the trade is
    # filtered out later by _within_cutoff. The parser itself returns the row.
    row = {"name": "Doe", "transactionType": "Sale"}
    t = congress_trades._parse_finnhub_row(row)
    assert t is not None
    assert t["type"] == "Sale"
    assert t["date"] == "—"


def test_parse_ssw_row_normalises_mmddyyyy_dates_and_party():
    row = {
        "transaction_date": "04/15/2026",
        "senator": "Tuberville, Tommy",
        "party": "Republican",
        "state": "AL",
        "type": "purchase",
        "amount": "$15,001 - $50,000",
        "ptr_link_date": "05/02/2026",
    }
    t = congress_trades._parse_ssw_row(row)
    assert t is not None
    assert t["date"] == "2026-04-15"
    assert t["filing_date"] == "2026-05-02"
    assert t["party"] == "R"
    assert t["state"] == "AL"
    assert t["chamber"] == "Senate"
    assert t["type"] == "Purchase"
    assert t["amount_min"] == 15_001
    assert t["amount_max"] == 50_000


def test_amount_range_to_floats_handles_various_inputs():
    assert congress_trades._amount_range_to_floats("$1,001 - $15,000") == (1001, 15000)
    assert congress_trades._amount_range_to_floats("$1,000,001 - $5,000,000") == (1_000_001, 5_000_000)
    assert congress_trades._amount_range_to_floats("$50,000") == (50_000, 50_000)
    assert congress_trades._amount_range_to_floats("") == (0.0, 0.0)


def test_parse_lambda_row_maps_canonical_shape():
    row = {
        "symbol": "AAPL",
        "name": "Pelosi, Nancy",
        "chamber": "House",
        "party": "Democrat",
        "state": "CA",
        "transaction_date": "2026-04-15",
        "transaction_type": "Purchase",
        "amount_min": 1_000_001,
        "amount_max": 5_000_000,
        "filing_date": "2026-04-30",
    }
    t = congress_trades._parse_lambda_row(row)
    assert t is not None
    assert t["filer"] == "Pelosi, Nancy"
    assert t["chamber"] == "House"
    assert t["party"] == "D"
    assert t["state"] == "CA"
    assert t["type"] == "Purchase"
    assert t["amount_min"] == 1_000_001
    assert t["amount_max"] == 5_000_000
    assert t["date"] == "2026-04-15"
    assert t["filing_date"] == "2026-04-30"
    assert t["filing_lag_days"] == 15


def test_parse_lambda_row_accepts_camelcase_and_range_string():
    """Tolerate camelCase fields and an ``amount`` range string in lieu of
    paired ``amount_min``/``amount_max`` — the live response shape isn't
    pinned in the public docs."""
    row = {
        "name": "Tuberville, Tommy",
        "chamber": "Senate",
        "party": "Republican",
        "state": "AL",
        "tradeDate": "04/15/2026",
        "type": "sell",
        "amount": "$15,001 - $50,000",
        "disclosure_date": "05/02/2026",
    }
    t = congress_trades._parse_lambda_row(row)
    assert t is not None
    assert t["chamber"] == "Senate"
    assert t["party"] == "R"
    assert t["type"] == "Sale"
    assert t["amount_min"] == 15_001
    assert t["amount_max"] == 50_000
    assert t["date"] == "2026-04-15"
    assert t["filing_date"] == "2026-05-02"


def test_parse_lambda_row_matches_live_api_shape():
    """Field names + shape captured from a real
    /api/congressional/trades response (AAPL, 2026-05). Older Senate rows
    have null party/state and lowercase chamber — verify we normalise."""
    row = {
        "symbol": "AAPL",
        "representative": "Thomas R Carper",
        "transactionDate": "2020-11-20",
        "disclosureDate": None,
        "type": "Sale (Partial)",
        "amount": "$50,001 - $100,000",
        "owner": "Spouse",
        "assetDescription": "Apple Inc.",
        "party": None,
        "state": None,
        "district": None,
        "chamber": "senate",
        "ptrLink": "https://efdsearch.senate.gov/...",
        "capGainsOver200": None,
        "comment": "--",
    }
    t = congress_trades._parse_lambda_row(row)
    assert t is not None
    assert t["filer"] == "Thomas R Carper"
    assert t["date"] == "2020-11-20"
    assert t["chamber"] == "Senate"  # title-cased from lowercase "senate"
    assert t["party"] == "—"          # null → placeholder
    assert t["state"] == "—"
    assert t["type"] == "Sale"        # "Sale (Partial)" normalises
    assert t["amount_min"] == 50_001
    assert t["amount_max"] == 100_000
    assert t["filing_date"] == "—"    # null disclosureDate → placeholder
    assert t["filing_lag_days"] is None


def test_extract_lambda_rows_matches_live_envelope():
    """Live envelope is ``{"trades": [...], "total": ..., "page": ..., "limit": ..., "hasMore": ...}``
    — no top-level ``data`` wrapper."""
    body = {
        "trades": [{"symbol": "AAPL", "representative": "X"}],
        "total": 436,
        "page": 0,
        "limit": 1,
        "hasMore": True,
    }
    rows = congress_trades._extract_lambda_rows(body)
    assert rows == [{"symbol": "AAPL", "representative": "X"}]


def test_extract_lambda_rows_handles_envelope_variants():
    rows = [{"symbol": "AAPL", "name": "X"}]
    assert congress_trades._extract_lambda_rows({"status": "ok", "data": rows}) == rows
    assert congress_trades._extract_lambda_rows({"data": {"trades": rows}}) == rows
    assert congress_trades._extract_lambda_rows({"data": {"results": rows}}) == rows
    assert congress_trades._extract_lambda_rows(rows) == rows
    assert congress_trades._extract_lambda_rows({"status": "ok"}) == []
    assert congress_trades._extract_lambda_rows("garbage") == []


def test_lambda_amounts_falls_back_through_field_variants():
    assert congress_trades._lambda_amounts({"amount_min": 1000, "amount_max": 5000}) == (1000, 5000)
    assert congress_trades._lambda_amounts({"amountFrom": 1000, "amountTo": 5000}) == (1000, 5000)
    assert congress_trades._lambda_amounts({"amount": "$1,001 - $15,000"}) == (1001, 15000)
    assert congress_trades._lambda_amounts({"amount": 50_000}) == (50_000, 50_000)
    assert congress_trades._lambda_amounts({}) == (0.0, 0.0)


def test_normalise_type_canonicalises_variants():
    assert congress_trades._normalise_type("Purchase") == "Purchase"
    assert congress_trades._normalise_type("buy") == "Purchase"
    assert congress_trades._normalise_type("Full Sale") == "Sale"
    assert congress_trades._normalise_type("sell") == "Sale"
    assert congress_trades._normalise_type("exchange") == "Exchange"
    assert congress_trades._normalise_type("") == "—"
    # Lambda Finance abbreviated codes with qualifiers
    assert congress_trades._normalise_type("S (Partial)") == "Sale"
    assert congress_trades._normalise_type("P (Full)") == "Purchase"
    assert congress_trades._normalise_type("E (Partial)") == "Exchange"
    assert congress_trades._normalise_type("S") == "Sale"
    assert congress_trades._normalise_type("P") == "Purchase"


def test_within_cutoff_excludes_old_trades():
    cutoff = date.today() - timedelta(days=180)
    old = congress_trades._Trade(date=(cutoff - timedelta(days=10)).isoformat(), type="Purchase",
                                  filer="X", amount_min=0, amount_max=0, amount_label="—",
                                  chamber="Senate", party="—", state="—",
                                  filing_date="—", filing_lag_days=None)
    new = congress_trades._Trade(date=(cutoff + timedelta(days=5)).isoformat(), type="Purchase",
                                  filer="Y", amount_min=0, amount_max=0, amount_label="—",
                                  chamber="Senate", party="—", state="—",
                                  filing_date="—", filing_lag_days=None)
    assert congress_trades._within_cutoff(new, cutoff) is True
    assert congress_trades._within_cutoff(old, cutoff) is False


# --- Reporting --------------------------------------------------------------


def test_format_report_emits_sentiment_and_table():
    trades = [
        congress_trades._Trade(
            date="2026-04-15", filer="Pelosi, Nancy", chamber="House", party="D", state="CA",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="2026-04-30", filing_lag_days=15,
        ),
        congress_trades._Trade(
            date="2026-04-10", filer="Tuberville, Tommy", chamber="Senate", party="R", state="AL",
            type="Sale", amount_min=15_001, amount_max=50_000,
            amount_label="$15k – $50k", filing_date="2026-04-25", filing_lag_days=15,
        ),
    ]
    report = congress_trades._format_report("AAPL", trades, lookback_days=180,
                                             source_label="Finnhub (House + Senate)")
    assert "AAPL" in report
    assert "1 unique buyer" in report and "1 unique seller" in report
    assert "net +0 buyers" in report
    assert "Pelosi, Nancy" in report and "Tuberville, Tommy" in report
    assert "House/D/CA" in report and "Senate/R/AL" in report
    assert "Finnhub (House + Senate)" in report


# --- Source-fallback chain --------------------------------------------------


def test_get_congress_trades_falls_through_to_senate_when_no_finnhub_key(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "",
        "lambda_finance_api_key": "",
    })
    captured = {}

    def fake_ssw(ticker, cutoff):
        captured["called_with"] = ticker
        return [
            congress_trades._Trade(
                date=(date.today() - timedelta(days=5)).isoformat(),
                filer="Tuberville, Tommy", chamber="Senate", party="R", state="AL",
                type="Purchase", amount_min=15_001, amount_max=50_000,
                amount_label="$15k – $50k", filing_date="—", filing_lag_days=None,
            )
        ]

    monkeypatch.setattr(congress_trades, "_fetch_senate_stock_watcher", fake_ssw)
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert captured.get("called_with") == "AAPL"
    assert "Senate Stock Watcher" in out
    assert "Tuberville" in out


def test_get_congress_trades_falls_through_when_finnhub_raises(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "",
    })
    monkeypatch.setattr(
        congress_trades, "_fetch_finnhub",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda ticker, cutoff: [],
    )
    out = congress_trades.get_congress_trades("XYZ", lookback_days=30)
    # Both sources empty → bracketed fallback that mentions the failure
    assert out.startswith("[Congressional disclosures unavailable")
    assert "boom" in out


def test_get_congress_trades_returns_finnhub_data_when_key_present(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "",
    })
    expected = [
        congress_trades._Trade(
            date=(date.today() - timedelta(days=2)).isoformat(),
            filer="Pelosi, Nancy", chamber="House", party="—", state="—",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="—", filing_lag_days=None,
        )
    ]
    monkeypatch.setattr(congress_trades, "_fetch_finnhub", lambda *a, **kw: expected)
    # Ensure SSW isn't called when Finnhub succeeds
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda *a, **kw: pytest.fail("SSW should not be called when Finnhub returns data"),
    )
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert "Finnhub (House + Senate)" in out
    assert "Pelosi, Nancy" in out


def test_lambda_finance_is_tried_before_finnhub_when_key_present(monkeypatch, tmp_path):
    """Lambda is the new primary — Finnhub must not be touched if Lambda
    returns rows."""
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "test-lambda-key",
    })
    expected = [
        congress_trades._Trade(
            date=(date.today() - timedelta(days=2)).isoformat(),
            filer="Pelosi, Nancy", chamber="House", party="D", state="CA",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="—", filing_lag_days=None,
        )
    ]
    monkeypatch.setattr(congress_trades, "_fetch_lambda_finance", lambda *a, **kw: expected)
    monkeypatch.setattr(
        congress_trades, "_fetch_finnhub",
        lambda *a, **kw: pytest.fail("Finnhub must not be called when Lambda returns data"),
    )
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda *a, **kw: pytest.fail("SSW must not be called when Lambda returns data"),
    )
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert "Lambda Finance (House + Senate)" in out
    assert "Pelosi, Nancy" in out


def test_falls_through_to_finnhub_when_lambda_returns_empty(monkeypatch, tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "test-key",
        "lambda_finance_api_key": "test-lambda-key",
    })
    finnhub_rows = [
        congress_trades._Trade(
            date=(date.today() - timedelta(days=2)).isoformat(),
            filer="Pelosi, Nancy", chamber="House", party="—", state="—",
            type="Purchase", amount_min=1_000_001, amount_max=5_000_000,
            amount_label="$1.0M – $5.0M", filing_date="—", filing_lag_days=None,
        )
    ]
    monkeypatch.setattr(congress_trades, "_fetch_lambda_finance", lambda *a, **kw: [])
    monkeypatch.setattr(congress_trades, "_fetch_finnhub", lambda *a, **kw: finnhub_rows)
    monkeypatch.setattr(
        congress_trades, "_fetch_senate_stock_watcher",
        lambda *a, **kw: pytest.fail("SSW must not be called when Finnhub returns data"),
    )
    out = congress_trades.get_congress_trades("AAPL", lookback_days=30)
    assert "Finnhub (House + Senate)" in out
    assert "Pelosi, Nancy" in out


def test_falls_through_when_lambda_raises(monkeypatch, tmp_path):
    """A Lambda HTTP failure must not abort the agent — chain continues."""
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "",
        "lambda_finance_api_key": "test-lambda-key",
    })
    monkeypatch.setattr(
        congress_trades, "_fetch_lambda_finance",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("lambda 502")),
    )
    monkeypatch.setattr(congress_trades, "_fetch_senate_stock_watcher", lambda *a, **kw: [])
    out = congress_trades.get_congress_trades("XYZ", lookback_days=30)
    assert out.startswith("[Congressional disclosures unavailable")
    assert "lambda 502" in out


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("FINNHUB_API_KEY"), reason="FINNHUB_API_KEY unset")
def test_get_congress_trades_live_finnhub_aapl(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": os.environ["FINNHUB_API_KEY"],
        "lambda_finance_api_key": "",
    })
    out = congress_trades.get_congress_trades("AAPL", lookback_days=180)
    assert isinstance(out, str) and out
    # Either real data or graceful no-data — never an unhandled crash
    assert out.startswith("##") or out.startswith("[Congressional disclosures unavailable")


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("LAMBDA_FINANCE_API_KEY"), reason="LAMBDA_FINANCE_API_KEY unset")
def test_get_congress_trades_live_lambda_finance_aapl(tmp_path):
    set_config({
        "data_cache_dir": str(tmp_path),
        "finnhub_api_key": "",
        "lambda_finance_api_key": os.environ["LAMBDA_FINANCE_API_KEY"],
    })
    out = congress_trades.get_congress_trades("AAPL", lookback_days=180)
    assert isinstance(out, str) and out
    assert out.startswith("##") or out.startswith("[Congressional disclosures unavailable")
