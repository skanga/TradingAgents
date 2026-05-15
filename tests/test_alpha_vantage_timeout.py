from unittest.mock import patch

import pytest
import requests

from tradingagents.dataflows.alpha_vantage_common import (
    DEFAULT_ALPHA_VANTAGE_TIMEOUT,
    AlphaVantageTemporaryError,
    _filter_csv_by_date_range,
    _make_api_request,
)


def test_alpha_vantage_request_uses_timeout(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "key")
    captured = {}

    class Response:
        text = "timestamp,open\n2026-01-01,1\n"

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        captured["timeout"] = timeout
        return Response()

    with patch("tradingagents.dataflows.alpha_vantage_common.requests.get", side_effect=fake_get):
        _make_api_request("TIME_SERIES_DAILY_ADJUSTED", {"symbol": "AAPL"})

    assert captured["timeout"] == DEFAULT_ALPHA_VANTAGE_TIMEOUT


def test_alpha_vantage_request_ignores_dead_global_entitlement(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "key")
    captured = {}

    class Response:
        text = "timestamp,open\n2026-01-01,1\n"

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        captured["params"] = params
        return Response()

    monkeypatch.setitem(
        __import__("tradingagents.dataflows.alpha_vantage_common", fromlist=[""]).__dict__,
        "_current_entitlement",
        "premium",
    )

    with patch("tradingagents.dataflows.alpha_vantage_common.requests.get", side_effect=fake_get):
        _make_api_request("TIME_SERIES_DAILY_ADJUSTED", {"symbol": "AAPL"})

    assert "entitlement" not in captured["params"]


def test_alpha_vantage_request_keeps_explicit_entitlement(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "key")
    captured = {}

    class Response:
        text = "timestamp,open\n2026-01-01,1\n"

        def raise_for_status(self):
            pass

    def fake_get(url, params, timeout):
        captured["params"] = params
        return Response()

    with patch("tradingagents.dataflows.alpha_vantage_common.requests.get", side_effect=fake_get):
        _make_api_request(
            "TIME_SERIES_DAILY_ADJUSTED",
            {"symbol": "AAPL", "entitlement": "realtime"},
        )

    assert captured["params"]["entitlement"] == "realtime"


def test_alpha_vantage_timeout_raises_temporary_error(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "key")
    with patch(
        "tradingagents.dataflows.alpha_vantage_common.requests.get",
        side_effect=requests.Timeout,
    ):
        with pytest.raises(AlphaVantageTemporaryError):
            _make_api_request("TIME_SERIES_DAILY_ADJUSTED", {"symbol": "AAPL"})


def test_alpha_vantage_csv_filter_failure_logs_warning(caplog):
    raw_csv = "timestamp,open\nnot-a-date,1\n"

    result = _filter_csv_by_date_range(raw_csv, "2026-01-01", "2026-01-02")

    assert result == raw_csv
    assert "Failed to filter CSV data by date range" in caplog.text
