import sys
from types import SimpleNamespace

import pandas as pd
import pytest

from tradingagents.dataflows import y_finance


def test_get_stock_stats_bulk_builds_indicator_map_without_iterrows(monkeypatch):
    data = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2026-01-02", "2026-01-05", "2026-01-06"]),
            "rsi": [41.5, None, 52.0],
        }
    )

    def fail_iterrows(self):
        raise AssertionError("iterrows should not be used for indicator mapping")

    monkeypatch.setitem(sys.modules, "stockstats", SimpleNamespace(wrap=lambda df: df))
    monkeypatch.setattr(y_finance, "load_ohlcv", lambda symbol, curr_date: data.copy())
    monkeypatch.setattr(pd.DataFrame, "iterrows", fail_iterrows)

    result = y_finance._get_stock_stats_bulk("NVDA", "rsi", "2026-01-06")

    assert result == {
        "2026-01-02": "41.5",
        "2026-01-05": "N/A",
        "2026-01-06": "52.0",
    }


def test_get_stock_stats_indicators_window_logs_bulk_fallback(monkeypatch, caplog):
    monkeypatch.setattr(
        y_finance,
        "_get_stock_stats_bulk",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("bulk failed")),
    )
    monkeypatch.setattr(y_finance, "get_stockstats_indicator", lambda *args: "42")

    result = y_finance.get_stock_stats_indicators_window(
        "NVDA",
        "rsi",
        "2026-01-06",
        1,
    )

    assert "2026-01-06: 42" in result
    assert "Error getting bulk stockstats data" in caplog.text
    assert "bulk failed" in caplog.text


def test_get_stockstats_indicator_logs_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        y_finance.StockstatsUtils,
        "get_stock_stats",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("stats failed")),
    )

    result = y_finance.get_stockstats_indicator("NVDA", "rsi", "2026-01-06")

    assert result == ""
    assert "Error getting stockstats indicator data" in caplog.text
    assert "stats failed" in caplog.text


def test_get_yfin_data_online_raises_clear_date_error():
    with pytest.raises(ValueError, match="start_date must use YYYY-MM-DD format"):
        y_finance.get_YFin_data_online("NVDA", "01/06/2026", "2026-01-07")
