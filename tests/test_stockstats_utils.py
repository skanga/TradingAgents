import pandas as pd

from tradingagents.dataflows.config import reset_config, use_config
from tradingagents.dataflows.stockstats_utils import load_ohlcv


def test_load_ohlcv_downloads_fifteen_year_cache_window(monkeypatch, tmp_path):
    captured = {}

    class FixedTimestamp(pd.Timestamp):
        @classmethod
        def today(cls, tz=None):
            return cls("2026-05-14")

    def fake_download(symbol, start, end, **kwargs):
        captured["symbol"] = symbol
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame(
            {
                "Date": [pd.Timestamp("2026-05-13")],
                "Open": [100.0],
                "High": [101.0],
                "Low": [99.0],
                "Close": [100.5],
                "Volume": [1000],
            }
        )

    monkeypatch.setattr(pd, "Timestamp", FixedTimestamp)
    monkeypatch.setattr("tradingagents.dataflows.stockstats_utils.yf.download", fake_download)
    token = use_config({"data_cache_dir": str(tmp_path)})
    try:
        data = load_ohlcv("NVDA", "2026-05-14")
    finally:
        reset_config(token)

    assert captured == {
        "symbol": "NVDA",
        "start": "2011-05-14",
        "end": "2026-05-14",
    }
    assert len(data) == 1
