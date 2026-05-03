import pandas as pd
import pytest

from tradingagents.dataflows import sector_analysis


def test_total_return_two_point_series():
    s = pd.Series([100.0, 110.0])
    assert sector_analysis._total_return(s) == pytest.approx(0.10)


def test_total_return_returns_none_for_short_series():
    assert sector_analysis._total_return(pd.Series([100.0])) is None
    assert sector_analysis._total_return(pd.Series([], dtype=float)) is None


def test_format_rs_report_tailwind_and_lagging_stock():
    report = sector_analysis._format_rs_report(
        ticker="AAPL",
        sector="Technology",
        sector_etf="XLK",
        lookback_days=63,
        ticker_ret=0.04,
        sector_ret=0.10,
        spy_ret=0.03,
    )
    assert "TAILWIND" in report
    assert "lagging its sector" in report
    assert "+4.00%" in report
    assert "+10.00%" in report
    assert report.startswith("##")


def test_format_rs_report_headwind_and_leading_stock():
    report = sector_analysis._format_rs_report(
        ticker="XOM",
        sector="Energy",
        sector_etf="XLE",
        lookback_days=63,
        ticker_ret=0.02,
        sector_ret=-0.04,
        spy_ret=0.03,
    )
    assert "HEADWIND" in report
    assert "leading its sector" in report


def test_format_corr_report_flags_notable_sensitivities():
    correlations = {
        "GLD": ("gold", -0.10),
        "USO": ("oil", 0.65),
        "BTC-USD": ("crypto", 0.20),
        "UUP": ("USD", -0.55),
        "^VIX": ("VIX", -0.30),
    }
    report = sector_analysis._format_corr_report("AAPL", 63, correlations)
    assert "+0.65" in report
    assert "-0.55" in report
    assert "Notable sensitivities" in report
    assert "USO (oil, +0.65): strong positive correlation." in report
    assert "UUP (USD, -0.55): strong negative correlation." in report


def test_format_corr_report_no_notable_sensitivities():
    correlations = {sym: (label, 0.10) for sym, label in sector_analysis._INTERMARKET.items()}
    report = sector_analysis._format_corr_report("AAPL", 63, correlations)
    assert "No basket assets exceed |0.5|" in report


def test_get_sector_relative_strength_unmapped_sector(monkeypatch):
    monkeypatch.setattr(sector_analysis, "_resolve_sector", lambda _t: "Imaginary Sector")
    out = sector_analysis.get_sector_relative_strength("XYZ", lookback_days=63)
    assert out.startswith("[Sector RS unavailable")
    assert "Imaginary Sector" in out


@pytest.mark.integration
def test_get_sector_relative_strength_live_aapl():
    out = sector_analysis.get_sector_relative_strength("AAPL", lookback_days=63)
    assert isinstance(out, str) and out
    assert not out.startswith("["), out
    assert "AAPL" in out
    assert "XLK" in out


@pytest.mark.integration
def test_get_intermarket_correlations_live_aapl():
    out = sector_analysis.get_intermarket_correlations("AAPL", lookback_days=63)
    assert isinstance(out, str) and out
    assert not out.startswith("["), out
    assert "AAPL" in out
    assert "GLD" in out
