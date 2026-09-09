import json

from tradingagents.dataflows import alpha_vantage_fundamentals


def test_live_statement_filters_future_fiscal_periods(monkeypatch):
    from tradingagents.dataflows import date_window
    monkeypatch.setattr(date_window, "get_current_date", lambda: "2025-12-31")
    payload = {
        "symbol": "AAPL",
        "annualReports": [
            {"fiscalDateEnding": "2026-12-31", "reportedCurrency": "USD"},
            {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD"},
        ],
        "quarterlyReports": [
            {"fiscalDateEnding": "2026-03-31", "reportedCurrency": "USD"},
            {"fiscalDateEnding": "2025-09-30", "reportedCurrency": "USD"},
        ],
    }

    monkeypatch.setattr(
        alpha_vantage_fundamentals,
        "_make_api_request",
        lambda function_name, params: json.dumps(payload),
    )

    result = alpha_vantage_fundamentals.get_balance_sheet("AAPL", curr_date="2025-12-31")
    parsed = json.loads(result)

    assert parsed["annualReports"] == [
        {"fiscalDateEnding": "2025-12-31", "reportedCurrency": "USD"}
    ]
    assert parsed["quarterlyReports"] == [
        {"fiscalDateEnding": "2025-09-30", "reportedCurrency": "USD"}
    ]
