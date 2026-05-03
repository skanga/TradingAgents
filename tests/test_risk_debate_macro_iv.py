"""Section 8 wiring tests: macro + IV snapshots flow into the risk debate."""

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.risk_mgmt.aggressive_debator import create_aggressive_debator
from tradingagents.agents.risk_mgmt.conservative_debator import create_conservative_debator
from tradingagents.agents.risk_mgmt.neutral_debator import create_neutral_debator
from tradingagents.graph.propagation import Propagator


# --- Propagator wiring -----------------------------------------------------


def test_initial_state_includes_macro_and_iv_snapshots():
    p = Propagator()
    state = p.create_initial_state(
        company_name="AAPL",
        trade_date="2026-05-02",
        past_context="prior decisions...",
        macro_snapshot="## Macroeconomic Environment ...",
        iv_snapshot="## Implied Volatility Rank ...",
    )
    assert state["macro_snapshot"].startswith("## Macroeconomic Environment")
    assert state["iv_snapshot"].startswith("## Implied Volatility Rank")
    assert state["past_context"] == "prior decisions..."


def test_initial_state_defaults_snapshots_to_empty_strings():
    p = Propagator()
    state = p.create_initial_state(company_name="AAPL", trade_date="2026-05-02")
    assert state["macro_snapshot"] == ""
    assert state["iv_snapshot"] == ""


# --- Debater prompts consume the snapshots ---------------------------------


def _state_with_snapshots(macro: str, iv: str) -> dict:
    """Minimal viable state for invoking a risk debater node."""
    return {
        "market_report": "M",
        "sentiment_report": "S",
        "news_report": "N",
        "fundamentals_report": "F",
        "trader_investment_plan": "Trader plan: BUY",
        "macro_snapshot": macro,
        "iv_snapshot": iv,
        "risk_debate_state": {
            "history": "",
            "aggressive_history": "",
            "conservative_history": "",
            "neutral_history": "",
            "current_aggressive_response": "",
            "current_conservative_response": "",
            "current_neutral_response": "",
            "latest_speaker": "",
            "judge_decision": "",
            "count": 0,
        },
    }


def _capture_prompt(node, state) -> str:
    """Invoke a debater node with a mock LLM and return the prompt string it was given."""
    captured = {}

    class _LLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            r = MagicMock()
            r.content = "rebuttal"
            return r

    # Re-create the debater closure with our capturing LLM
    factory_for = {
        "aggressive": create_aggressive_debator,
        "conservative": create_conservative_debator,
        "neutral": create_neutral_debator,
    }[node]
    inner = factory_for(_LLM())
    inner(state)
    return captured["prompt"]


@pytest.mark.parametrize("role", ["aggressive", "conservative", "neutral"])
def test_each_debater_prompt_embeds_both_snapshots(role):
    macro = "## Macroeconomic Environment\n**Backdrop**: UNFAVORABLE for equities"
    iv = "## Implied Volatility Rank\n- IV Rank: 78/100\n**Interpretation**: ELEVATED"
    prompt = _capture_prompt(role, _state_with_snapshots(macro, iv))
    assert "Macro Backdrop" in prompt
    assert "Implied Volatility Context" in prompt
    assert "UNFAVORABLE for equities" in prompt
    assert "IV Rank: 78/100" in prompt


@pytest.mark.parametrize("role", ["aggressive", "conservative", "neutral"])
def test_debater_prompt_uses_fallback_text_when_snapshots_empty(role):
    prompt = _capture_prompt(role, _state_with_snapshots("", ""))
    assert "Macro snapshot unavailable" in prompt
    assert "IV-rank snapshot unavailable" in prompt


# --- Role-specific framing -------------------------------------------------


def test_aggressive_prompt_frames_high_iv_as_partially_priced():
    prompt = _capture_prompt("aggressive", _state_with_snapshots("favorable", "high"))
    # The aggressive debater is told to reframe high IV as "already priced"
    assert "already partially priced" in prompt or "fading the consensus fear" in prompt


def test_conservative_prompt_warns_about_low_iv_complacency():
    prompt = _capture_prompt("conservative", _state_with_snapshots("ok", "low"))
    assert "complacency" in prompt or "low IV ahead of identifiable catalysts" in prompt


def test_neutral_prompt_uses_macro_and_iv_as_tiebreaker():
    prompt = _capture_prompt("neutral", _state_with_snapshots("mixed", "mid"))
    assert "tie-breaker" in prompt
