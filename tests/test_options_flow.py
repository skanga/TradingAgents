import pandas as pd
import pytest

from tradingagents.dataflows import options_flow


# --- Fixtures: synthetic option chains ---


def _calls():
    return pd.DataFrame(
        {
            "strike": [90, 95, 100, 105, 110],
            "lastPrice": [12.0, 7.5, 4.0, 1.5, 0.5],
            "volume": [50, 200, 1500, 800, 50],
            "openInterest": [100, 500, 1200, 200, 50],
            "impliedVolatility": [0.45, 0.40, 0.35, 0.32, 0.30],
        }
    )


def _puts():
    return pd.DataFrame(
        {
            "strike": [90, 95, 100, 105, 110],
            "lastPrice": [0.5, 1.5, 4.0, 7.5, 12.0],
            "volume": [600, 200, 100, 100, 50],
            "openInterest": [800, 300, 100, 100, 50],
            "impliedVolatility": [0.50, 0.45, 0.36, 0.34, 0.32],
        }
    )


# --- Pure-math helpers ------------------------------------------------------


def test_aggregate_pc_ratios_single_expiry():
    out = options_flow.aggregate_pc_ratios([{"calls": _calls(), "puts": _puts()}])
    assert out["call_vol"] == 50 + 200 + 1500 + 800 + 50
    assert out["put_vol"] == 600 + 200 + 100 + 100 + 50
    # P/C vol = 1050 / 2600 ≈ 0.404 (bullish)
    assert out["pc_vol"] == pytest.approx(1050 / 2600)
    # P/C OI = 1350 / 2050 ≈ 0.659
    assert out["pc_oi"] == pytest.approx(1350 / 2050)


def test_aggregate_pc_ratios_handles_empty_chains():
    empty = pd.DataFrame(columns=["volume", "openInterest"])
    out = options_flow.aggregate_pc_ratios([{"calls": empty, "puts": empty}])
    assert out["pc_vol"] is None and out["pc_oi"] is None


def test_max_pain_strike_at_balanced_strike():
    """With matching put/call OI distributions, max pain sits near the centre."""
    pain = options_flow.max_pain_strike(_calls(), _puts())
    # The synthetic chain has the heaviest combined OI around strike 95-100.
    # Strike where total intrinsic payout is minimised should sit in that range.
    assert 90 <= pain <= 105


def test_find_walls_returns_largest_oi_strike():
    call_wall = options_flow.find_walls(_calls())
    put_wall = options_flow.find_walls(_puts())
    assert call_wall == (100.0, 1200.0)  # row with the largest OI in calls
    assert put_wall == (90.0, 800.0)


def test_find_unusual_flow_flags_high_ratio_strikes():
    # Engineer one strike with vol >> 3x OI on the call side
    calls = _calls().copy()
    calls.loc[2, "volume"] = 5000   # OI was 1200 → ratio ≈ 4.17
    flow = options_flow.find_unusual_flow(
        [{"exp": "2026-06-19", "calls": calls, "puts": _puts()}], spot=100.0
    )
    assert flow, "expected at least one unusual-flow row"
    top = flow[0]
    assert top["side"] == "C" and top["strike"] == 100
    assert top["ratio"] == pytest.approx(5000 / 1200)
    assert "expiry" in top and top["expiry"] == "2026-06-19"


def test_atm_iv_picks_strike_closest_to_spot():
    # spot=102 → closest strike is 100; mean of 0.35 (call) and 0.36 (put) = 0.355
    iv = options_flow._atm_iv(_calls(), _puts(), spot=102.0)
    assert iv == pytest.approx((0.35 + 0.36) / 2)


# --- Reporting --------------------------------------------------------------


def test_format_summary_includes_pc_label_and_walls():
    per_expiry = [{"exp": "2026-06-19", "calls": _calls(), "puts": _puts()}]
    report = options_flow._format_summary("DEMO", spot=100.0, per_expiry=per_expiry)
    assert report.startswith("## Options Flow Summary for DEMO")
    assert "P/C volume ratio" in report
    assert "ATM IV" in report
    assert "Max pain strike" in report
    assert "Call wall: $100" in report
    assert "Put wall: $90" in report


def test_format_summary_pc_label_buckets():
    assert options_flow._pc_label(0.30) == "EXTREME bullish skew"
    assert options_flow._pc_label(0.80) == "modest bullish skew"
    assert options_flow._pc_label(1.0) == "balanced"
    assert options_flow._pc_label(1.20) == "modest bearish skew"
    assert options_flow._pc_label(1.60) == "EXTREME bearish skew"
    assert options_flow._pc_label(None) == "n/a"


def test_format_ivr_report_interpretations():
    elev = options_flow._format_ivr_report("X", atm_iv=0.40, hv_min=0.10, hv_max=0.50, ivr=85.0)
    assert "ELEVATED" in elev
    comp = options_flow._format_ivr_report("X", atm_iv=0.12, hv_min=0.10, hv_max=0.50, ivr=10.0)
    assert "COMPLACENCY" in comp
    neut = options_flow._format_ivr_report("X", atm_iv=0.25, hv_min=0.10, hv_max=0.50, ivr=40.0)
    assert "NEUTRAL" in neut


# --- Integration ------------------------------------------------------------


@pytest.mark.integration
def test_get_options_summary_live_aapl():
    out = options_flow.get_options_summary("AAPL")
    assert isinstance(out, str) and out
    assert not out.startswith("["), out
    assert "AAPL" in out and "P/C volume ratio" in out


@pytest.mark.integration
def test_get_iv_rank_live_aapl():
    out = options_flow.get_iv_rank("AAPL")
    assert isinstance(out, str) and out
    assert not out.startswith("["), out
    assert "IV Rank" in out
