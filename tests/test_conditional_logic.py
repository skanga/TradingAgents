"""Conditional-logic loop-termination tests.

Regression target: a flaky free-tier LLM ran the AAPL Market Analyst 33
tool rounds before LangGraph's recursion limit killed the entire run
(2026-05-05 trial, run id 2026_05_05_01_01_00). The conditional logic
now caps tool rounds per analyst — verified here so a model that
keeps requesting tools forever can't crash the pipeline again.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.graph.conditional_logic import ConditionalLogic


def _ai_message(*, has_tool_calls: bool):
    """Mock AIMessage with the only attribute the conditional logic reads."""
    m = MagicMock()
    m.tool_calls = [{"name": "stub"}] if has_tool_calls else []
    return m


def _human_message():
    """Mock HumanMessage / placeholder. No tool_calls attribute."""
    m = MagicMock()
    m.tool_calls = None
    return m


def _state(messages):
    return {"messages": messages}


@pytest.mark.parametrize(
    "method_name, tools_node, clear_node",
    [
        ("should_continue_market",       "tools_market",       "Msg Clear Market"),
        ("should_continue_social",       "tools_social",       "Msg Clear Social"),
        ("should_continue_news",         "tools_news",         "Msg Clear News"),
        ("should_continue_fundamentals", "tools_fundamentals", "Msg Clear Fundamentals"),
        ("should_continue_options",      "tools_options",      "Msg Clear Options"),
    ],
)
class TestPerAnalystRouting:
    """Each analyst routes through the same shared logic; pin the contract once."""

    def test_routes_to_tools_when_last_message_requests_tools(
        self, method_name, tools_node, clear_node,
    ):
        cl = ConditionalLogic()
        state = _state([_human_message(), _ai_message(has_tool_calls=True)])
        assert getattr(cl, method_name)(state) == tools_node

    def test_routes_to_clear_when_last_message_has_no_tool_calls(
        self, method_name, tools_node, clear_node,
    ):
        cl = ConditionalLogic()
        state = _state([_human_message(), _ai_message(has_tool_calls=False)])
        assert getattr(cl, method_name)(state) == clear_node

    def test_forces_clear_after_tool_round_cap(
        self, method_name, tools_node, clear_node, caplog,
    ):
        """The motivating regression: LLM keeps requesting tools forever.
        After max_tool_rounds_per_analyst rounds, force termination."""
        cap = 5
        cl = ConditionalLogic(max_tool_rounds_per_analyst=cap)
        # Build a message log with `cap` AIMessages each carrying tool_calls.
        # The conditional fires AFTER each LLM response, so on the (cap)th
        # call we have exactly `cap` tool-bearing AIMessages and must stop.
        messages = [_human_message()]
        for _ in range(cap):
            messages.append(_ai_message(has_tool_calls=True))
        assert getattr(cl, method_name)(_state(messages)) == clear_node

    def test_routes_to_tools_one_below_cap(
        self, method_name, tools_node, clear_node,
    ):
        """Just below the cap, normal routing still applies."""
        cap = 5
        cl = ConditionalLogic(max_tool_rounds_per_analyst=cap)
        messages = [_human_message()]
        for _ in range(cap - 1):
            messages.append(_ai_message(has_tool_calls=True))
        assert getattr(cl, method_name)(_state(messages)) == tools_node


def test_default_cap_matches_documented_ceiling():
    """The default ceiling should comfortably fit Fundamentals' ~8 tools
    while still tripping a runaway 33-round loop. Documented as 12."""
    cl = ConditionalLogic()
    assert cl.max_tool_rounds_per_analyst == 12


def test_cap_is_inclusive_at_threshold(caplog):
    """At exactly cap rounds, force termination — don't allow one more
    iteration past the limit."""
    import logging
    caplog.set_level(logging.WARNING, logger="tradingagents.graph.conditional_logic")
    cl = ConditionalLogic(max_tool_rounds_per_analyst=3)
    messages = [_human_message()] + [_ai_message(has_tool_calls=True) for _ in range(3)]
    out = cl.should_continue_market(_state(messages))
    assert out == "Msg Clear Market"
    # And a warning is logged so operators can see why termination fired
    assert any(
        "Market Analyst hit tool-round cap" in r.message
        for r in caplog.records
    )
