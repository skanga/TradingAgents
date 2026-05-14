import sys
from types import SimpleNamespace

import pandas as pd

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
