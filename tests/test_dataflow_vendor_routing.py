import pytest
import requests
from yfinance.exceptions import YFRateLimitError

from tradingagents.dataflows.config import reset_config, use_config
from tradingagents.dataflows import interface
from tradingagents.dataflows.interface import get_vendor, route_to_vendor


def test_get_vendor_rejects_non_string_category_vendor():
    token = use_config({"data_vendors": {"core_stock_apis": ["yfinance"]}})
    try:
        with pytest.raises(ValueError, match="must be a string"):
            get_vendor("core_stock_apis", "get_stock_data")
    finally:
        reset_config(token)


def test_get_vendor_rejects_non_string_tool_vendor():
    token = use_config({"tool_vendors": {"get_stock_data": ["yfinance"]}})
    try:
        with pytest.raises(ValueError, match="must be a string"):
            get_vendor("core_stock_apis", "get_stock_data")
    finally:
        reset_config(token)


def test_route_to_vendor_rejects_unknown_method_before_routing():
    with pytest.raises(ValueError, match="not found in any category"):
        route_to_vendor("missing_method")


def test_route_to_vendor_falls_back_after_generic_transient_error(monkeypatch):
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        raise requests.Timeout("temporary timeout")

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return "fallback result"

    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_stock_data",
        {
            "yfinance": primary,
            "alpha_vantage": fallback,
        },
    )
    token = use_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    try:
        assert route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-01-10") == "fallback result"
        assert calls == ["primary", "fallback"]
    finally:
        reset_config(token)


def test_route_to_vendor_falls_back_after_connection_error(monkeypatch):
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        raise requests.ConnectionError("temporary connection reset")

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return "fallback result"

    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_stock_data",
        {
            "yfinance": primary,
            "alpha_vantage": fallback,
        },
    )
    token = use_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    try:
        assert route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-01-10") == "fallback result"
        assert calls == ["primary", "fallback"]
    finally:
        reset_config(token)


def _http_error(status_code: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status_code
    error = requests.HTTPError(f"{status_code} response")
    error.response = response
    return error


def test_route_to_vendor_does_not_fallback_after_permanent_http_error(monkeypatch):
    calls = []
    error = _http_error(403)

    def primary(*args, **kwargs):
        calls.append("primary")
        raise error

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return "fallback result"

    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_stock_data",
        {
            "yfinance": primary,
            "alpha_vantage": fallback,
        },
    )
    token = use_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    try:
        with pytest.raises(requests.HTTPError) as exc_info:
            route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-01-10")
        assert exc_info.value is error
        assert calls == ["primary"]
    finally:
        reset_config(token)


@pytest.mark.parametrize("status_code", [429, 500, 502, 503, 504])
def test_route_to_vendor_falls_back_after_transient_http_error(monkeypatch, status_code):
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        raise _http_error(status_code)

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return "fallback result"

    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_stock_data",
        {
            "yfinance": primary,
            "alpha_vantage": fallback,
        },
    )
    token = use_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    try:
        assert route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-01-10") == "fallback result"
        assert calls == ["primary", "fallback"]
    finally:
        reset_config(token)


def test_route_to_vendor_falls_back_after_yfinance_rate_limit(monkeypatch):
    calls = []

    def primary(*args, **kwargs):
        calls.append("primary")
        raise YFRateLimitError()

    def fallback(*args, **kwargs):
        calls.append("fallback")
        return "fallback result"

    monkeypatch.setitem(
        interface.VENDOR_METHODS,
        "get_stock_data",
        {
            "yfinance": primary,
            "alpha_vantage": fallback,
        },
    )
    token = use_config({"data_vendors": {"core_stock_apis": "yfinance"}})
    try:
        assert route_to_vendor("get_stock_data", "NVDA", "2026-01-01", "2026-01-10") == "fallback result"
        assert calls == ["primary", "fallback"]
    finally:
        reset_config(token)
