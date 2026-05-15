from tradingagents.dataflows import alpha_vantage_indicator


def test_alpha_vantage_indicator_failure_logs_warning(monkeypatch, caplog):
    monkeypatch.setattr(
        alpha_vantage_indicator,
        "_make_api_request",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("api failed")),
    )

    result = alpha_vantage_indicator.get_indicator("NVDA", "rsi", "2026-01-10", 3)

    assert result == "Error retrieving rsi data: api failed"
    assert "Error getting Alpha Vantage indicator data for rsi" in caplog.text
