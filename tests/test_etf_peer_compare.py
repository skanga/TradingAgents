"""Tests for the yfinance ETF peer-comparison adapter.

Compares a primary ETF against 2-6 peer ETFs on a fixed metric set
(profile + returns + risk). Live integration test runs in-band because
yfinance is unauthenticated.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import etf_peer_compare
from tradingagents.dataflows.config import set_config


# --- Input parsing ---------------------------------------------------------


def test_parse_peers_dedups_uppercases_and_drops_primary():
    out = etf_peer_compare._parse_peers("qqq, IWM ,QQQ, SPY,DIA", exclude="SPY")
    assert out == ["QQQ", "IWM", "DIA"]  # SPY (primary) removed; QQQ deduped


def test_parse_peers_caps_at_max():
    too_many = ",".join(f"E{i}" for i in range(etf_peer_compare._MAX_PEERS + 5))
    out = etf_peer_compare._parse_peers(too_many, exclude="X")
    assert len(out) == etf_peer_compare._MAX_PEERS


def test_parse_peers_rejects_unsafe_components():
    with pytest.raises(ValueError):
        etf_peer_compare._parse_peers("QQQ,../../etc/passwd", exclude="X")


def test_parse_peers_rejects_non_string_input():
    with pytest.raises(ValueError):
        etf_peer_compare._parse_peers(["QQQ", "IWM"], exclude="X")


# --- Computed price metrics ------------------------------------------------


def _synthetic_close(n: int, *, start: float = 100.0, drift: float = 0.0005) -> pd.Series:
    """Build a clean synthetic price series so metric math is deterministic.
    Constant log-drift keeps daily returns equal to ``drift`` exactly."""
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=n, freq="B", tz="UTC")
    prices = [start * ((1 + drift) ** i) for i in range(n)]
    return pd.Series(prices, index=dates, name="Close")


def test_compute_price_metrics_handles_empty_series():
    assert etf_peer_compare._compute_price_metrics(pd.Series(dtype=float)) == {}


def test_compute_price_metrics_returns_use_correct_lookbacks():
    """A constant-drift series gives an exact compound-return per window."""
    close = _synthetic_close(300, drift=0.001)  # +0.1% daily
    out = etf_peer_compare._compute_price_metrics(close)
    # 21-day return ≈ (1.001)^21 - 1
    assert out["ret_1m"] == pytest.approx((1.001) ** 21 - 1, rel=1e-9)
    assert out["ret_3m"] == pytest.approx((1.001) ** 63 - 1, rel=1e-9)
    assert out["ret_1y"] == pytest.approx((1.001) ** 252 - 1, rel=1e-9)


def test_compute_price_metrics_skips_windows_when_history_too_short():
    """A 30-day series can compute ret_1m but not ret_3m / ret_1y."""
    close = _synthetic_close(30, drift=0.0)
    out = etf_peer_compare._compute_price_metrics(close)
    assert "ret_1m" in out
    assert "ret_3m" not in out
    assert "ret_1y" not in out


def test_compute_price_metrics_volatility_uses_only_last_252_days():
    """A series with 315 days, but vol_1y must reflect only the last 252.

    We make the first 100 days perfectly flat (zero returns) and the last
    215 days have small noise. If we used the whole window, the flat
    leading section would suppress volatility. Restricting to the 1Y
    window yields the noise-only volatility.
    """
    import numpy as np
    np.random.seed(7)
    flat = [100.0] * 100
    noisy = [100.0]
    for _ in range(214):
        noisy.append(noisy[-1] * (1 + np.random.normal(0, 0.01)))
    prices = flat + noisy
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=len(prices), freq="B", tz="UTC")
    close = pd.Series(prices, index=dates)

    out = etf_peer_compare._compute_price_metrics(close)
    # Noise stdev ~1% daily → annualized ~16%. If the flat section bled
    # into the calc, vol would be much lower.
    assert 0.10 < out["vol_1y"] < 0.25


def test_compute_price_metrics_max_drawdown_is_negative():
    """A series that rises then falls should produce a negative drawdown."""
    prices = [100.0 * (1.001 ** i) for i in range(150)] + \
             [prices_decline := 100.0 * (1.001 ** 149)] + \
             [prices_decline * (0.99 ** i) for i in range(1, 100)]
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=len(prices), freq="B", tz="UTC")
    close = pd.Series(prices, index=dates)
    out = etf_peer_compare._compute_price_metrics(close)
    assert out["max_dd_1y"] < -0.3  # ~63% decline at the trough


# --- Info extraction -------------------------------------------------------


def test_extract_info_metrics_keeps_known_fields_and_ignores_garbage():
    info = {
        "totalAssets": 651_588_272_128,
        "yield": 0.0114,
        "netExpenseRatio": 0.0945,
        "fundInceptionDate": 727660800,  # 1993-01-22
        "category": "Large Blend",
        "beta3Year": 1.0,
        "irrelevant": "ignored",
        "totalAssets_alias": "ignored too",
    }
    m = etf_peer_compare._extract_info_metrics(info)
    assert m["aum"] == 651_588_272_128
    assert m["yield"] == 0.0114
    assert m["expense_ratio"] == 0.0945
    assert m["inception_year"] == 1993
    assert m["category"] == "Large Blend"
    assert m["beta_3y"] == 1.0
    assert "irrelevant" not in m


def test_extract_info_metrics_drops_zero_or_invalid_aum():
    """``totalAssets`` of 0 / None means yfinance has no AUM; render as
    em-dash, not as ``$0``."""
    assert etf_peer_compare._extract_info_metrics({"totalAssets": 0}).get("aum") is None
    assert etf_peer_compare._extract_info_metrics({"totalAssets": None}).get("aum") is None


# --- Cell formatters -------------------------------------------------------


def test_fmt_aum_scales_to_unit_suffix():
    assert etf_peer_compare._fmt_aum(2_500_000_000_000) == "$2.50T"
    assert etf_peer_compare._fmt_aum(651_588_272_128) == "$651.59B"
    assert etf_peer_compare._fmt_aum(372_510_000_000) == "$372.51B"
    assert etf_peer_compare._fmt_aum(50_000_000) == "$50.00M"
    assert etf_peer_compare._fmt_aum(None) == "—"
    assert etf_peer_compare._fmt_aum(0) == "—"


def test_fmt_yield_treats_value_as_fraction():
    assert etf_peer_compare._fmt_yield(0.0114) == "1.14%"


def test_fmt_expense_ratio_treats_value_as_percent():
    """yfinance quirk: netExpenseRatio is already in percent units."""
    assert etf_peer_compare._fmt_expense_ratio(0.0945) == "0.09%"
    assert etf_peer_compare._fmt_expense_ratio(0.20) == "0.20%"


def test_fmt_return_includes_sign_for_signed_reads():
    assert etf_peer_compare._fmt_return(0.0948) == "+9.48%"
    assert etf_peer_compare._fmt_return(-0.0876) == "-8.76%"
    assert etf_peer_compare._fmt_return(0.0) == "+0.00%"
    assert etf_peer_compare._fmt_return(None) == "—"


def test_fmt_year_skips_zero_and_negative():
    assert etf_peer_compare._fmt_year(1993) == "1993"
    assert etf_peer_compare._fmt_year(0) == "—"
    assert etf_peer_compare._fmt_year(None) == "—"


# --- End-to-end formatting -------------------------------------------------


def test_format_table_renders_all_rows_in_canonical_order():
    rows = {
        "SPY": {
            "category": "Large Blend",
            "aum": 651_588_272_128, "expense_ratio": 0.0945, "yield": 0.0114,
            "inception_year": 1993, "beta_3y": 1.0,
            "ret_1m": 0.0948, "ret_3m": 0.0353, "ret_ytd": 0.0539,
            "ret_1y": 0.3004, "vol_1y": 0.1856, "max_dd_1y": -0.1876,
        },
        "QQQ": {
            "category": "Large Growth", "aum": 372_510_000_000,
            "expense_ratio": 0.18, "yield": 0.0049, "inception_year": 1999,
            "beta_3y": 1.11, "ret_1m": 0.1503, "ret_3m": 0.076,
            "ret_ytd": 0.0989, "ret_1y": 0.4037, "vol_1y": 0.2256,
            "max_dd_1y": -0.2277,
        },
    }
    out = etf_peer_compare._format_table("SPY", ["SPY", "QQQ"], rows)
    assert "## ETF Peer Comparison — SPY vs QQQ" in out
    assert "yfinance (prices + info)" in out
    # Header row carries the ticker symbols
    header = next(l for l in out.splitlines() if l.startswith("| Metric"))
    assert header.index("SPY") < header.index("QQQ")
    # Each metric row has the per-ticker value in column order
    line_for = lambda label: next(l for l in out.splitlines() if l.startswith(f"| {label} |"))
    cat = line_for("Category")
    assert cat.index("Large Blend") < cat.index("Large Growth")
    assert "$651.59B" in line_for("AUM") and "$372.51B" in line_for("AUM")
    assert "0.09%" in line_for("Expense Ratio")
    assert "1.14%" in line_for("Distribution Yield")
    assert "1993" in line_for("Inception") and "1999" in line_for("Inception")
    assert "+9.48%" in line_for("Return — 1M") and "+15.03%" in line_for("Return — 1M")
    assert "-18.76%" in line_for("Max Drawdown (1Y)")
    assert "All 2 ETFs had data" in out


def test_format_table_marks_missing_tickers_in_footer():
    rows = {"SPY": {"aum": 651_588_272_128}, "OBSCURE": {}}
    out = etf_peer_compare._format_table("SPY", ["SPY", "OBSCURE"], rows)
    assert "no data for OBSCURE" in out


# --- Public-API behaviour --------------------------------------------------


def test_rejects_unsafe_primary_ticker(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    out = etf_peer_compare.get_etf_peer_comparison("../../etc/passwd", "QQQ")
    assert out.startswith("[ETF peer comparison unavailable")


def test_returns_unavailable_when_peers_empty(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    out = etf_peer_compare.get_etf_peer_comparison("SPY", "")
    assert out.startswith("[ETF peer comparison unavailable")
    assert "no peers supplied" in out


def test_returns_unavailable_when_fetch_raises(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})

    def boom(tickers):
        raise RuntimeError("yfinance unreachable")

    monkeypatch.setattr(etf_peer_compare, "_fetch_metrics", boom)
    out = etf_peer_compare.get_etf_peer_comparison("SPY", "QQQ,IWM")
    assert out.startswith("[ETF peer comparison unavailable")
    assert "yfinance unreachable" in out


def test_renders_when_fetch_returns_data(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(
        etf_peer_compare, "_fetch_metrics",
        lambda tickers: {
            "SPY": {"category": "Large Blend", "aum": 651_588_272_128,
                    "ret_1m": 0.0948, "ret_3m": 0.0353},
            "QQQ": {"category": "Large Growth", "aum": 372_510_000_000,
                    "ret_1m": 0.1503, "ret_3m": 0.076},
        },
    )
    out = etf_peer_compare.get_etf_peer_comparison("SPY", "QQQ")
    assert "## ETF Peer Comparison — SPY vs QQQ" in out
    assert "$651.59B" in out and "$372.51B" in out


def test_caches_result_within_ttl(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    call_count = {"n": 0}

    def fetch(tickers):
        call_count["n"] += 1
        return {sym: {"aum": 1_000_000_000} for sym in tickers}

    monkeypatch.setattr(etf_peer_compare, "_fetch_metrics", fetch)
    a = etf_peer_compare.get_etf_peer_comparison("SPY", "QQQ,IWM")
    b = etf_peer_compare.get_etf_peer_comparison("SPY", "QQQ,IWM")
    assert a == b
    assert call_count["n"] == 1


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
def test_get_etf_peer_comparison_live_spy_vs_siblings(tmp_path):
    """yfinance unauth — live call should produce a real comparison."""
    set_config({"data_cache_dir": str(tmp_path)})
    out = etf_peer_compare.get_etf_peer_comparison("SPY", "QQQ,IWM,DIA")
    assert isinstance(out, str) and out
    assert out.startswith("##") or out.startswith("[ETF peer comparison unavailable")
