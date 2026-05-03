import os
from datetime import date, timedelta

import pytest

from tradingagents.dataflows import macro_data
from tradingagents.dataflows.config import set_config


# --- FRED observation parsing -----------------------------------------------


def test_parse_observations_filters_dot_sentinel_and_invalid_rows():
    body = {
        "observations": [
            {"date": "2026-04-01", "value": "4.10"},
            {"date": "2026-04-02", "value": "."},          # FRED missing-value
            {"date": "2026-04-03", "value": ""},           # blank
            {"date": "garbage",     "value": "4.20"},      # bad date
            {"date": "2026-04-05", "value": "4.30"},
        ]
    }
    out = macro_data._parse_observations(body)
    assert [d.isoformat() for d, _ in out] == ["2026-04-01", "2026-04-05"]
    assert [v for _, v in out] == [4.10, 4.30]


def test_value_n_days_ago_picks_value_at_or_before_target():
    today = date(2026, 5, 1)
    obs = [
        (today - timedelta(days=60), 4.00),
        (today - timedelta(days=30), 4.20),
        (today - timedelta(days=15), 4.40),
        (today,                        4.50),
    ]
    # Target = 30 days ago → exact match
    assert macro_data._value_n_days_ago(obs, 30) == 4.20
    # Target = 25 days ago (between -30d and -15d) → 30d value is the largest <= target
    assert macro_data._value_n_days_ago(obs, 25) == 4.20
    # Target older than the oldest sample → fall back to oldest
    assert macro_data._value_n_days_ago(obs, 1000) == 4.00


# --- Per-series interpretation ---------------------------------------------


def test_interpret_inverted_curve_is_negative():
    score, label = macro_data._interpret("T10Y2Y", current=-0.45, delta=None, pct_change=None)
    assert score == -1
    assert "inverted" in label


def test_interpret_positive_curve_is_supportive():
    score, label = macro_data._interpret("T10Y2Y", current=0.75, delta=None, pct_change=None)
    assert score == 1
    assert "positive" in label


def test_interpret_hy_widening_is_risk_off():
    score, label = macro_data._interpret(
        "BAMLH0A0HYM2", current=4.20, delta=0.50, pct_change=None,
    )
    assert score == -1
    assert "widening" in label


def test_interpret_hy_tightening_is_risk_on():
    score, label = macro_data._interpret(
        "BAMLH0A0HYM2", current=3.20, delta=-0.20, pct_change=None,
    )
    assert score == 1
    assert "tightening" in label


def test_interpret_falling_long_yields_supportive():
    score, label = macro_data._interpret("DGS10", current=4.20, delta=-0.40, pct_change=None)
    assert score == 1
    assert "supportive" in label.lower()


def test_interpret_strengthening_dollar_is_headwind():
    score, label = macro_data._interpret("DTWEXBGS", current=125.0, delta=None, pct_change=0.025)
    assert score == -1
    assert "strengthening" in label


# --- Backdrop classification + report formatting ---------------------------


def _snap(series_id, label, units, current, delta, pct, score, interp):
    return macro_data._SeriesSnapshot(
        series_id=series_id, label=label, units=units,
        current=current, current_date="2026-05-01",
        prior=current - (delta or 0.0), delta=delta, pct_change=pct,
        score=score, interpretation=interp,
    )


def test_classify_backdrop_thresholds():
    assert macro_data._classify_backdrop(3) == "FAVORABLE"
    assert macro_data._classify_backdrop(2) == "FAVORABLE"
    assert macro_data._classify_backdrop(1) == "NEUTRAL"
    assert macro_data._classify_backdrop(-1) == "NEUTRAL"
    assert macro_data._classify_backdrop(-2) == "UNFAVORABLE"


def test_format_report_combines_signals_into_favorable_backdrop():
    snapshots = [
        _snap("DGS10",        "10Y Treasury",     "%",     4.20, -0.30, None,  1, "falling (eq-supportive)"),
        _snap("T10Y2Y",       "10Y-2Y Spread",    "pp",    0.65, None,  None,  1, "comfortably positive"),
        _snap("BAMLH0A0HYM2", "HY Credit Spread", "%",     3.30, -0.15, None,  1, "tightening (risk-on)"),
        _snap("DTWEXBGS",     "Broad USD Index",  "level", 120.0, None, -0.02, 1, "softening"),
    ]
    report = macro_data._format_report(snapshots, lookback_days=30, partial=False)
    assert "FAVORABLE" in report
    assert "10Y Treasury" in report
    assert "tightening (risk-on)" in report
    assert report.startswith("## Macroeconomic Environment")


def test_format_report_partial_flag_emitted():
    snapshots = [
        _snap("DGS2", "2Y Treasury", "%", 4.10, 0.0, None, 0, "stable"),
    ]
    out = macro_data._format_report(snapshots, lookback_days=30, partial=True)
    assert "(partial" in out


# --- Top-level entry point --------------------------------------------------


def test_get_macro_environment_returns_fallback_when_no_key(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "fred_api_key": ""})
    out = macro_data.get_macro_environment(lookback_days=30)
    assert out.startswith("[Macro environment unavailable: FRED_API_KEY")


def test_get_macro_environment_aggregates_partial_failures(tmp_path, monkeypatch):
    set_config({"data_cache_dir": str(tmp_path), "fred_api_key": "test-key"})

    def fake_build(series_id, label, units, _key, _lookback):
        if series_id in ("DGS10", "T10Y2Y"):
            return _snap(series_id, label, units, 4.0, -0.30, None, 1, "supportive")
        # All others "fail" (network error etc.)
        raise RuntimeError("simulated FRED outage")

    monkeypatch.setattr(macro_data, "_build_snapshot", fake_build)
    out = macro_data.get_macro_environment(lookback_days=30)
    assert out.startswith("## Macroeconomic Environment")
    assert "(partial" in out
    assert "DGS10" in out and "T10Y2Y" in out


def test_get_macro_environment_returns_fallback_when_all_fail(tmp_path, monkeypatch):
    set_config({"data_cache_dir": str(tmp_path), "fred_api_key": "test-key"})
    monkeypatch.setattr(
        macro_data, "_build_snapshot",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("network down")),
    )
    out = macro_data.get_macro_environment(lookback_days=30)
    assert out.startswith("[Macro environment unavailable")
    assert "network down" in out


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not os.environ.get("FRED_API_KEY"), reason="FRED_API_KEY unset")
def test_get_macro_environment_live(tmp_path):
    set_config({"data_cache_dir": str(tmp_path), "fred_api_key": os.environ["FRED_API_KEY"]})
    out = macro_data.get_macro_environment(lookback_days=30)
    assert isinstance(out, str) and out
    assert not out.startswith("["), out
    assert "Backdrop" in out
    assert any(label in out for label in ("FAVORABLE", "NEUTRAL", "UNFAVORABLE"))
