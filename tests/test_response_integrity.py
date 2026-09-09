"""Incomplete generations must not become final reports or learned reflections."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.structured import bind_structured, invoke_structured_or_freetext
from tradingagents.graph.reflection import Reflector
from tradingagents.graph.trading_graph import TradingAgentsGraph


class Report(BaseModel):
    text: str


@pytest.mark.parametrize("metadata", [
    {"finish_reason": "length"},
    {"stop_reason": "max_tokens"},
    {"finish_reason": "MAX_TOKENS"},
    {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
    {"finish_reason": "content_filter"},
])
def test_incomplete_freetext_retries_before_returning(metadata):
    llm = Mock()
    llm.invoke.side_effect = [
        AIMessage(content="Rating: Buy\nBecause", response_metadata=metadata),
        AIMessage(content="Rating: Hold\nEvidence is balanced."),
    ]
    result = invoke_structured_or_freetext(None, llm, "prompt", str, "Portfolio Manager")
    assert result == "Rating: Hold\nEvidence is balanced."
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("content", ["", " \n\t"])
def test_empty_freetext_fails_after_bounded_retry(content):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=content)
    with pytest.raises(ValueError, match="empty"):
        invoke_structured_or_freetext(None, llm, "prompt", str, "Trader")
    assert llm.invoke.call_count == 2


def test_structured_binding_requests_raw_metadata():
    llm = Mock()
    bind_structured(llm, Report, "Analyst")
    llm.with_structured_output.assert_called_once_with(Report, include_raw=True)


def test_truncated_structured_result_is_not_rendered():
    structured, plain, render = Mock(), Mock(), Mock()
    structured.invoke.return_value = {
        "raw": AIMessage(content="partial", response_metadata={"finish_reason": "length"}),
        "parsed": Report(text="partial"),
        "parsing_error": None,
    }
    plain.invoke.return_value = AIMessage(content="Complete fallback")
    assert invoke_structured_or_freetext(structured, plain, "prompt", render, "Analyst") == "Complete fallback"
    render.assert_not_called()


def test_valid_tool_structured_response_can_have_empty_text():
    structured, plain = Mock(), Mock()
    structured.invoke.return_value = {
        "raw": AIMessage(content="", response_metadata={"stop_reason": "tool_use"}),
        "parsed": Report(text="Complete report"),
        "parsing_error": None,
    }
    result = invoke_structured_or_freetext(structured, plain, "prompt", lambda r: r.text, "Analyst")
    assert result == "Complete report"
    plain.invoke.assert_not_called()


def test_content_blocks_are_returned_as_text_not_python_repr():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=[{"type": "text", "text": "Complete report"}])
    assert invoke_structured_or_freetext(None, llm, "prompt", str, "Trader") == "Complete report"


@pytest.mark.parametrize("message", [
    AIMessage(content=[{"type": "thinking", "thinking": "Private reasoning"}]),
    AIMessage(content="Planning to fetch", tool_calls=[
        {"name": "get_news", "args": {}, "id": "call_1"},
    ]),
])
def test_reasoning_and_tool_requests_are_not_deliverables(message):
    llm = Mock()
    llm.invoke.return_value = message
    with pytest.raises(ValueError):
        invoke_structured_or_freetext(None, llm, "prompt", str, "Trader")
    assert llm.invoke.call_count == 2


def test_transport_error_does_not_trigger_extra_generation():
    llm = Mock()
    llm.invoke.side_effect = TimeoutError("provider timed out")
    with pytest.raises(TimeoutError):
        invoke_structured_or_freetext(None, llm, "prompt", str, "Trader")
    assert llm.invoke.call_count == 1


def test_plain_response_without_metadata_is_preserved():
    llm = Mock()
    llm.invoke.return_value = SimpleNamespace(content="Complete answer")
    assert invoke_structured_or_freetext(None, llm, "prompt", str, "Trader") == "Complete answer"
    assert llm.invoke.call_count == 1


def test_reflection_can_recover_on_retry():
    llm = Mock()
    llm.invoke.side_effect = [AIMessage(content=""), AIMessage(content="Evidence was insufficient.")]
    assert Reflector(llm).reflect_on_final_decision("Rating: Hold", 0.01, 0.0) == "Evidence was insufficient."
    assert llm.invoke.call_count == 2


def test_structured_parsing_error_uses_validated_fallback():
    structured, plain, render = Mock(), Mock(), Mock()
    structured.invoke.return_value = {
        "raw": AIMessage(content="malformed JSON"),
        "parsed": None,
        "parsing_error": ValueError("invalid JSON"),
    }
    plain.invoke.return_value = AIMessage(content="Complete fallback")
    assert invoke_structured_or_freetext(structured, plain, "prompt", render, "Analyst") == "Complete fallback"
    render.assert_not_called()


def test_failed_reflection_remains_pending(tmp_path):
    memory = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.jsonl")})
    memory.store_decision("AAPL", "2026-01-05", "Rating: Buy\nPositive evidence.")
    llm = Mock()
    llm.invoke.return_value = AIMessage(
        content="The thesis was", response_metadata={"stop_reason": "max_tokens"}
    )
    graph = SimpleNamespace(
        memory_log=memory,
        reflector=Reflector(llm),
        _resolve_benchmark=lambda ticker: "SPY",
        _fetch_returns=lambda *a, **k: (0.05, 0.02, 5, "2026-01-12"),
    )
    TradingAgentsGraph._resolve_pending_entries(graph, "AAPL")
    assert len(memory.get_pending_entries()) == 1
    assert memory.get_past_context("AAPL") == ""
    assert llm.invoke.call_count == 2
