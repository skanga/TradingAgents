"""Tests for the graph-node checkpoint wrapper.

The wrapper surfaces grep-able ENTER/EXIT lines around every named node
in the LangGraph workflow so the run log carries clear boundaries
between agents (without per-agent code edits).
"""

from __future__ import annotations

import logging
import re

import pytest

from tradingagents.graph.setup import _with_checkpoint


def test_wrapper_logs_enter_and_exit_with_ticker(caplog):
    caplog.set_level(logging.INFO, logger="tradingagents.graph.setup")
    fn = lambda state: {"market_report": "ok"}
    wrapped = _with_checkpoint("Market Analyst", fn)

    result = wrapped({"company_of_interest": "AAPL"})

    assert result == {"market_report": "ok"}
    messages = [r.message for r in caplog.records]
    assert any(re.fullmatch(r"ENTER Market Analyst \| AAPL", m) for m in messages)
    assert any(re.fullmatch(r"EXIT  Market Analyst \| AAPL \| \d+\.\ds", m) for m in messages)


def test_wrapper_emits_enter_before_exit(caplog):
    """ENTER must appear before EXIT in the log stream — this matters for
    log-tail parsers that pair them up by order."""
    caplog.set_level(logging.INFO, logger="tradingagents.graph.setup")
    wrapped = _with_checkpoint("Bull Researcher", lambda state: state)
    wrapped({"company_of_interest": "MSFT"})

    enter_idx = next(i for i, r in enumerate(caplog.records) if r.message.startswith("ENTER"))
    exit_idx = next(i for i, r in enumerate(caplog.records) if r.message.startswith("EXIT"))
    assert enter_idx < exit_idx


def test_wrapper_falls_back_to_question_mark_when_state_lacks_ticker(caplog):
    caplog.set_level(logging.INFO, logger="tradingagents.graph.setup")
    wrapped = _with_checkpoint("Bear Researcher", lambda state: state)
    wrapped({})  # no company_of_interest

    messages = [r.message for r in caplog.records]
    assert any(m.startswith("ENTER Bear Researcher | ?") for m in messages)


def test_wrapper_handles_non_dict_state(caplog):
    """A node may receive a non-dict input in tests or a future LangGraph
    update; ticker lookup must not crash the wrapper."""
    caplog.set_level(logging.INFO, logger="tradingagents.graph.setup")
    wrapped = _with_checkpoint("Trader", lambda state: state)
    wrapped(["not", "a", "dict"])

    messages = [r.message for r in caplog.records]
    assert any(m.startswith("ENTER Trader | ?") for m in messages)


def test_wrapper_logs_failure_and_propagates_exception(caplog):
    caplog.set_level(logging.INFO, logger="tradingagents.graph.setup")

    def boom(state):
        raise RuntimeError("kaboom")

    wrapped = _with_checkpoint("Research Manager", boom)
    with pytest.raises(RuntimeError, match="kaboom"):
        wrapped({"company_of_interest": "NVDA"})

    messages = [r.message for r in caplog.records]
    # ENTER fired, FAIL fired, EXIT did NOT fire (no graceful return)
    assert any(m.startswith("ENTER Research Manager | NVDA") for m in messages)
    assert any(re.match(r"FAIL  Research Manager \| NVDA \| \d+\.\ds \| RuntimeError: kaboom", m) for m in messages)
    assert not any(m.startswith("EXIT  Research Manager") for m in messages)


def test_wrapper_passes_through_extra_args_and_kwargs(caplog):
    """LangGraph may pass a config arg to nodes; the wrapper must not
    swallow it."""
    caplog.set_level(logging.INFO, logger="tradingagents.graph.setup")
    captured = {}

    def fn(state, config=None, **kwargs):
        captured["state"] = state
        captured["config"] = config
        captured["kwargs"] = kwargs
        return state

    wrapped = _with_checkpoint("Trader", fn)
    wrapped({"company_of_interest": "AAPL"}, config={"thread_id": "x"}, extra="y")

    assert captured["config"] == {"thread_id": "x"}
    assert captured["kwargs"] == {"extra": "y"}


def test_wrapper_preserves_return_value():
    sentinel = {"investment_plan": "buy aapl 1pct"}
    wrapped = _with_checkpoint("Research Manager", lambda state: sentinel)
    assert wrapped({"company_of_interest": "AAPL"}) is sentinel
