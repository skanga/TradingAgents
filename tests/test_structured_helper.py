"""Tests for ``invoke_structured_or_freetext``.

Regression target: a SPY Portfolio Manager run silently failed to produce
any verdict because (a) the LLM emitted malformed JSON that broke the
PortfolioDecision Pydantic parse, (b) the free-text fallback also
returned empty content, and (c) the empty string propagated to the
markdown writer which correctly skipped the section. The user saw a
report missing the entire Portfolio Manager Verdict with no indication
of why.

The helper now substitutes a bracketed placeholder when both paths
produce empty content, so the failure is visible to downstream agents
and the renderer.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tradingagents.agents.utils.structured import invoke_structured_or_freetext


def _ai(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


def test_returns_rendered_structured_output_when_present():
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = MagicMock()
    out = invoke_structured_or_freetext(
        structured_llm,
        plain_llm=MagicMock(),
        prompt="x",
        render=lambda r: "## Verdict\n\nBuy AAPL.",
        agent_name="Portfolio Manager",
    )
    assert out == "## Verdict\n\nBuy AAPL."


def test_falls_back_to_freetext_when_structured_raises():
    """The motivating SPY case: structured-output raises a Pydantic JSON
    validation error; free-text retry returns valid content."""
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = ValueError(
        "1 validation error for PortfolioDecision: Invalid JSON ..."
    )
    plain_llm = MagicMock()
    plain_llm.invoke.return_value = _ai("Free-text verdict: Hold AAPL.")
    out = invoke_structured_or_freetext(
        structured_llm,
        plain_llm=plain_llm,
        prompt="x",
        render=lambda r: r,
        agent_name="Portfolio Manager",
    )
    assert out == "Free-text verdict: Hold AAPL."
    structured_llm.invoke.assert_called_once()
    plain_llm.invoke.assert_called_once()


def test_falls_back_when_structured_renders_empty():
    """Edge case: structured succeeds but render() produces empty
    content (e.g. all schema fields came back as empty strings).
    Should still trigger free-text fallback."""
    structured_llm = MagicMock()
    structured_llm.invoke.return_value = MagicMock()
    plain_llm = MagicMock()
    plain_llm.invoke.return_value = _ai("Recovered verdict.")
    out = invoke_structured_or_freetext(
        structured_llm,
        plain_llm=plain_llm,
        prompt="x",
        render=lambda r: "   ",  # whitespace-only render
        agent_name="Portfolio Manager",
    )
    assert out == "Recovered verdict."


def test_substitutes_placeholder_when_both_paths_empty():
    """The exact silent-failure mode that motivated the helper change:
    structured raises, free-text returns empty content. Must produce
    a visible placeholder, never an empty string."""
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = ValueError("malformed JSON")
    plain_llm = MagicMock()
    plain_llm.invoke.return_value = _ai("")
    out = invoke_structured_or_freetext(
        structured_llm,
        plain_llm=plain_llm,
        prompt="x",
        render=lambda r: r,
        agent_name="Portfolio Manager",
    )
    assert out.startswith("[Portfolio Manager could not produce a verdict")
    assert "structured output failed" in out
    assert "Treat this section as missing" in out


def test_substitutes_placeholder_when_freetext_returns_whitespace():
    structured_llm = MagicMock()
    structured_llm.invoke.side_effect = RuntimeError("boom")
    plain_llm = MagicMock()
    plain_llm.invoke.return_value = _ai("   \n\n  ")
    out = invoke_structured_or_freetext(
        structured_llm,
        plain_llm=plain_llm,
        prompt="x",
        render=lambda r: r,
        agent_name="Research Manager",
    )
    assert out.startswith("[Research Manager could not produce")


def test_no_structured_llm_uses_plain_directly():
    """When the provider doesn't support structured output, structured_llm
    is None and we go straight to free text."""
    plain_llm = MagicMock()
    plain_llm.invoke.return_value = _ai("Free-text only.")
    out = invoke_structured_or_freetext(
        None,
        plain_llm=plain_llm,
        prompt="x",
        render=lambda r: "should not be called",
        agent_name="Trader",
    )
    assert out == "Free-text only."


def test_no_structured_llm_with_empty_freetext_substitutes_placeholder():
    plain_llm = MagicMock()
    plain_llm.invoke.return_value = _ai("")
    out = invoke_structured_or_freetext(
        None,
        plain_llm=plain_llm,
        prompt="x",
        render=lambda r: "ignored",
        agent_name="Trader",
    )
    assert out.startswith("[Trader could not produce")
