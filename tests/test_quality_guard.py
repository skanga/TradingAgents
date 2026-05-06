"""Tests for the analyst-output quality guard."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.agents.utils.quality_guard import (
    invoke_chain_with_quality_retry,
    is_degenerate_report,
    make_unavailable_report,
    strip_reasoning_leak,
)


def _ai(content: str, *, tool_calls=None) -> MagicMock:
    """Mock AIMessage-shaped object: `.content`, `.tool_calls` attributes."""
    m = MagicMock()
    m.content = content
    m.tool_calls = tool_calls or []
    return m


def _chain(*results):
    """Mock chain whose ``invoke`` returns the given AIMessage-shaped mocks
    in order, one per call."""
    chain = MagicMock()
    chain.invoke = MagicMock(side_effect=list(results))
    return chain


# --- is_degenerate_report --------------------------------------------------


def test_empty_or_whitespace_is_degenerate():
    assert is_degenerate_report("") is True
    assert is_degenerate_report("   \n\n  ") is True


def test_non_string_is_degenerate():
    assert is_degenerate_report(None) is True
    assert is_degenerate_report(42) is True
    assert is_degenerate_report({"foo": "bar"}) is True


def test_two_word_response_is_degenerate():
    """The motivating regression: an LLM that returned 'Call correct.'
    instead of a real Fundamentals report."""
    assert is_degenerate_report("Call correct.") is True


def test_short_unstructured_text_is_degenerate():
    assert is_degenerate_report("Looks fine.") is True
    assert is_degenerate_report("HOLD") is True


def test_short_but_structured_passes():
    """A short bullet list still has form — the LLM made a structural
    choice. Don't flag it."""
    bullet = "- Bullet one\n- Bullet two\n- Bullet three"
    assert is_degenerate_report(bullet) is False


def test_short_table_passes():
    table = "| A | B |\n|---|---|\n| 1 | 2 |"
    assert is_degenerate_report(table) is False


def test_short_heading_passes():
    """A heading-only response is still structured. Above-threshold body
    isn't required when structure is present."""
    assert is_degenerate_report("# Title\n\nA brief note.") is False


def test_long_unstructured_paragraph_passes():
    """If the LLM produced a substantive free-form paragraph, accept it
    even without explicit markdown structure."""
    long_text = "x " * 200  # 400 chars
    assert is_degenerate_report(long_text) is False


def test_threshold_boundary():
    """Under threshold + no structure → degenerate; at threshold → not."""
    # 199 chars, no structure
    assert is_degenerate_report("a" * 199) is True
    # 200 chars, no structure
    assert is_degenerate_report("a" * 200) is False


# --- make_unavailable_report ----------------------------------------------


def test_unavailable_report_includes_label_and_marker():
    out = make_unavailable_report(
        analyst_label="Fundamentals Analyst", original="Call correct."
    )
    assert "## Fundamentals Analyst — output unavailable" in out
    assert "Call correct." in out
    assert "missing" in out  # explicit signal for downstream debaters


def test_unavailable_report_handles_empty_original():
    out = make_unavailable_report(analyst_label="News Analyst", original="")
    assert "## News Analyst — output unavailable" in out
    assert "_(empty)_" in out


def test_unavailable_report_truncates_long_original():
    """A model that returned 5000 chars of garbage shouldn't cause us to
    embed all of it in the placeholder."""
    out = make_unavailable_report(
        analyst_label="X", original="garbage " * 1000
    )
    # Truncated form ends with the ellipsis the helper appends
    assert "…" in out
    assert len(out) < 800


@pytest.mark.parametrize("original", ["", "a", "x" * 50, "x" * 1000])
def test_unavailable_report_is_itself_not_degenerate(original):
    """The placeholder must pass the same quality guard so it doesn't
    look like another degenerate response to anyone reading it."""
    out = make_unavailable_report(analyst_label="X Analyst", original=original)
    assert not is_degenerate_report(out)


# --- strip_reasoning_leak --------------------------------------------------


def test_strip_reasoning_leak_removes_motivating_spy_run_pattern():
    """The exact leak observed in the May 4 SPY run, line 55."""
    leaked = (
        "Finally other: choose close_10_ema, macd, rsi, boll_ub, boll_lb, atr, vwma."
        "We can stop.**FINAL TRANSACTION PROPOSAL: HOLD** – The SPY charts show a "
        "neutral‑to‑bullish trend with moderate volatility..."
    )
    out = strip_reasoning_leak(leaked)
    assert out.startswith("**FINAL TRANSACTION PROPOSAL: HOLD**")
    assert "Finally other" not in out
    assert "We can stop" not in out


def test_strip_reasoning_leak_handles_clean_input_unchanged():
    """No leak marker → content returned verbatim."""
    clean = "## Market Analyst\n\nThe price action shows..."
    assert strip_reasoning_leak(clean) is clean or strip_reasoning_leak(clean) == clean


def test_strip_reasoning_leak_handles_empty_and_non_string():
    assert strip_reasoning_leak("") == ""
    assert strip_reasoning_leak(None) is None
    # Non-strings pass through unmodified (callers shouldn't pass them, but
    # the helper shouldn't crash if they do).
    assert strip_reasoning_leak(42) == 42


def test_strip_reasoning_leak_only_scans_first_chunk():
    """A legitimate body that mentions 'we can stop' deep in the report
    must not be truncated mid-content."""
    head = "## Real Heading\n\n" + ("Substantive paragraph. " * 100)
    body = head + "Finally we can stop here for clarity. More content."
    out = strip_reasoning_leak(body)
    # The 'we can stop' is past the 1000-char scan window → no strip
    assert out.startswith("## Real Heading")


def test_strip_reasoning_leak_strips_leading_whitespace_after_marker():
    leaked = "We can stop.\n\n## Real Heading\n\nReal content."
    out = strip_reasoning_leak(leaked)
    assert out.startswith("## Real Heading")
    assert "\n\n" not in out[: len("## Real Heading")]


def test_strip_reasoning_leak_handles_due_to_time_limit_paraphrase():
    """Observed verbatim on the 2026-05-05 SPY trial run, Market Analyst:
    'Need more indicators; due to time limit may stop.**FINAL TRANSACTION
    PROPOSAL:** _Awaiting full indicator set...'"""
    leaked = (
        "Need more indicators; due to time limit may stop."
        "**FINAL TRANSACTION PROPOSAL:** _Awaiting full indicator set – "
        "cannot finalize recommendation yet._"
    )
    out = strip_reasoning_leak(leaked)
    assert out.startswith("**FINAL TRANSACTION PROPOSAL:**")
    assert "Need more indicators" not in out
    assert "due to time limit" not in out


def test_strip_reasoning_leak_handles_need_toolname_paraphrase():
    """Observed verbatim on the 2026-05-05 SPY Market Analyst:
    'Need atr.## SPY Technical Analysis Report (2026-05-05) ...'"""
    leaked = "Need atr.## SPY Technical Analysis Report\n\nReal content here."
    out = strip_reasoning_leak(leaked)
    assert out.startswith("## SPY Technical Analysis Report")
    assert "Need atr" not in out


def test_strip_reasoning_leak_handles_need_multiple_toolnames():
    """'Need boll, atr.' was the AAPL Market Analyst variant."""
    leaked = "Need boll, atr.\n\n## Real Report\n\nBody."
    out = strip_reasoning_leak(leaked)
    assert out.startswith("## Real Report")
    assert "Need boll" not in out


def test_strip_reasoning_leak_does_not_match_legitimate_need_phrasing():
    """The 'Need <toolname>.' pattern is deliberately constrained to short
    alphanumeric+comma+space content followed by a period so it can't
    swallow legitimate report sentences like 'Need to monitor the Fed.'"""
    legit = "Need to monitor the Fed and the upcoming jobs report next month."
    out = strip_reasoning_leak(legit)
    # Legitimate sentence stays intact (the words after "Need" don't match
    # the constrained pattern — they include "to" which is fine, but the
    # period boundary lands at the END of the sentence, well past the
    # 40-char cap, so the regex doesn't match).
    assert out == legit


def test_strip_reasoning_leak_finally_other_works_when_paired_with_we_can_stop():
    """Real 2026-05-04 SPY case: 'Finally other: ... .We can stop.' followed
    by the report. The 'We can stop.' marker provides the clean boundary
    even when 'Finally other:' is in the same string."""
    leaked = (
        "Finally other: choose close_10_ema, macd, rsi, boll_ub.We can stop."
        "**FINAL TRANSACTION PROPOSAL: HOLD** – ..."
    )
    out = strip_reasoning_leak(leaked)
    assert out.startswith("**FINAL TRANSACTION PROPOSAL: HOLD**")
    assert "Finally other" not in out


# --- invoke_chain_with_quality_retry --------------------------------------


def test_invoke_returns_first_response_when_not_degenerate():
    good = _ai("# Heading\n\n" + "real content. " * 20)
    chain = _chain(good)
    msg, report = invoke_chain_with_quality_retry(
        chain, ["seed"], analyst_label="Fundamentals Analyst"
    )
    assert msg is good
    assert "real content" in report
    chain.invoke.assert_called_once()


def test_invoke_passes_through_tool_calls_without_retry():
    """A response that requests tool calls is mid-conversation, not a final
    answer. Don't retry, don't substitute — return verbatim with empty
    report so the LangGraph supervisor routes to ToolNode."""
    needs_tool = _ai("", tool_calls=[{"name": "get_fundamentals"}])
    chain = _chain(needs_tool)
    msg, report = invoke_chain_with_quality_retry(
        chain, ["seed"], analyst_label="Fundamentals Analyst"
    )
    assert msg is needs_tool
    assert report == ""
    chain.invoke.assert_called_once()


def test_invoke_retries_when_first_response_is_degenerate():
    """The motivating bug: first response is 'Call correct.', retry
    yields a real report — keep the retry."""
    bad = _ai("Call correct.")
    good = _ai("# Real Report\n\n" + "details. " * 30)
    chain = _chain(bad, good)
    msg, report = invoke_chain_with_quality_retry(
        chain, ["seed"], analyst_label="Fundamentals Analyst"
    )
    assert msg is good
    assert "Real Report" in report
    assert chain.invoke.call_count == 2

    # The retry call gets the original messages plus a stricter user prompt
    second_call_args = chain.invoke.call_args_list[1].args[0]
    assert second_call_args[:1] == ["seed"]
    assert second_call_args[-1][0] == "user"
    assert "previous response" in second_call_args[-1][1].lower()


def test_invoke_substitutes_placeholder_when_retry_also_degenerate():
    bad1 = _ai("Call correct.")
    bad2 = _ai("nope.")
    chain = _chain(bad1, bad2)
    msg, report = invoke_chain_with_quality_retry(
        chain, ["seed"], analyst_label="Fundamentals Analyst"
    )
    # Original message kept (so message history reflects what happened)
    assert msg is bad1
    # Report is the unavailable placeholder, citing the original output
    assert "Fundamentals Analyst — output unavailable" in report
    assert "Call correct." in report
    assert chain.invoke.call_count == 2


def test_invoke_substitutes_placeholder_when_retry_returns_tool_calls():
    """If the retry tries to call more tools instead of producing a
    report, that's still a failed retry — substitute the placeholder."""
    bad = _ai("HOLD")
    retry_with_tool = _ai("", tool_calls=[{"name": "get_news"}])
    chain = _chain(bad, retry_with_tool)
    msg, report = invoke_chain_with_quality_retry(
        chain, ["seed"], analyst_label="News Analyst"
    )
    assert msg is bad
    assert "News Analyst — output unavailable" in report
    assert chain.invoke.call_count == 2


def test_invoke_returns_empty_report_below_cap_with_tool_calls():
    """Below the cap, tool-calling rounds should still return an empty
    report so LangGraph routes through ToolNode for another iteration.
    The cap-aware substitution must not fire prematurely."""
    needs_tool = _ai("", tool_calls=[{"name": "get_x"}])
    chain = _chain(needs_tool)
    # 3 prior tool-bearing AIMessages in messages; cap=12 → plenty of room
    prior = [_ai("", tool_calls=[{"name": "x"}]) for _ in range(3)]
    msg, report = invoke_chain_with_quality_retry(
        chain, prior, analyst_label="Market Analyst", max_tool_rounds=12,
    )
    assert msg is needs_tool
    assert report == ""


def test_invoke_substitutes_placeholder_when_about_to_hit_round_cap():
    """The motivating SPY/AAPL Market Analyst regression: LLM keeps
    requesting tools, conditional_logic's tool-round cap is about to
    force-terminate, and without this branch the empty report propagates
    silently to the rendered output."""
    needs_tool = _ai("", tool_calls=[{"name": "get_x"}])
    chain = _chain(needs_tool)
    # 11 prior tool-bearing rounds + this 12th call hits cap of 12
    prior = [_ai("", tool_calls=[{"name": "x"}]) for _ in range(11)]
    msg, report = invoke_chain_with_quality_retry(
        chain, prior, analyst_label="Market Analyst", max_tool_rounds=12,
    )
    assert msg is needs_tool
    assert "Market Analyst — output unavailable" in report
    assert "force-terminated after 12 tool-calling rounds" in report


def test_invoke_cap_only_substitutes_when_max_tool_rounds_provided():
    """Backward compat: callers that don't pass max_tool_rounds get the
    pre-cap behaviour (empty report on tool calls, no substitution)."""
    needs_tool = _ai("", tool_calls=[{"name": "get_x"}])
    chain = _chain(needs_tool)
    prior = [_ai("", tool_calls=[{"name": "x"}]) for _ in range(11)]
    msg, report = invoke_chain_with_quality_retry(
        chain, prior, analyst_label="Market Analyst",
        # max_tool_rounds intentionally omitted
    )
    assert report == ""
