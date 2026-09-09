"""All direct model paths must validate output before updating agent state."""
from copy import deepcopy
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import tradingagents.agents as agents
from tradingagents.agents.utils.response_integrity import IncompleteResponseError
from tradingagents.graph.propagation import Propagator

DEBATERS = ["create_bull_researcher", "create_bear_researcher", "create_aggressive_debator",
            "create_conservative_debator", "create_neutral_debator"]
ANALYSTS = [("create_news_analyst", "news_report", "get_news"),
            ("create_market_analyst", "market_report", "get_stock_data"),
            ("create_fundamentals_analyst", "fundamentals_report", "get_fundamentals")]


def state():
    result = Propagator().create_initial_state("AAPL", "2024-01-15")
    result["trader_investment_plan"] = "plan"
    return result


def analyst_llm(responses):
    invoke = Mock(side_effect=responses)
    llm = Mock()
    llm.bind_tools.return_value = RunnableLambda(invoke)
    return llm, invoke


@pytest.mark.parametrize("factory", DEBATERS)
def test_debater_retries_before_persisting_argument(factory):
    llm = Mock()
    llm.invoke.side_effect = [AIMessage(content="PARTIAL", response_metadata={"finish_reason": "length"}),
                              AIMessage(content=[{"type": "text", "text": "Complete argument"}])]
    output = getattr(agents, factory)(llm)(state())
    debate = next(iter(output.values()))
    assert "PARTIAL" not in debate["history"]
    assert "Complete argument" in debate["history"]
    assert "'type': 'text'" not in debate["history"]
    assert debate["count"] == 1
    assert llm.invoke.call_count == 2
    assert llm.invoke.call_args_list[0] == llm.invoke.call_args_list[1]


@pytest.mark.parametrize("factory", DEBATERS)
def test_failed_debater_does_not_advance_state(factory):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="")
    original = state()
    before = deepcopy(original)
    with pytest.raises(IncompleteResponseError):
        getattr(agents, factory)(llm)(original)
    assert original == before
    assert llm.invoke.call_count == 2


@pytest.mark.parametrize("factory,report,tool_name", ANALYSTS)
@pytest.mark.parametrize("bad", [AIMessage(content=""),
    AIMessage(content="Partial", response_metadata={"stop_reason": "max_tokens"}),
    AIMessage(content="partial JSON", invalid_tool_calls=[{"name": "get_news", "args": "{", "id": "bad", "error": "invalid JSON"}]),
])
def test_analyst_retries_unusable_generation(factory, report, tool_name, bad):
    llm, invoke = analyst_llm([bad, AIMessage(content="Complete report")])
    output = getattr(agents, factory)(llm)(state())
    assert output[report] == "Complete report"
    assert len(output["messages"]) == 1
    assert invoke.call_count == 2
    assert invoke.call_args_list[0].args[0] == invoke.call_args_list[1].args[0]


@pytest.mark.parametrize("factory,report,tool_name", ANALYSTS)
def test_valid_tool_request_preserves_empty_content_and_protocol(factory, report, tool_name):
    response = AIMessage(content="", tool_calls=[{"name": tool_name, "args": {}, "id": "ok"}],
                         response_metadata={"finish_reason": "tool_calls"}, additional_kwargs={"reasoning_content": "reasoning"})
    llm, invoke = analyst_llm([response])
    output = getattr(agents, factory)(llm)(state())
    assert output["messages"][0] is response
    assert output[report] == ""
    assert invoke.call_count == 1


@pytest.mark.parametrize("factory,report,tool_name", ANALYSTS)
def test_analyst_normalizes_answer_blocks_without_discarding_envelope(factory, report, tool_name):
    response = AIMessage(content=[{"type": "thinking", "thinking": "Private"}, {"type": "text", "text": "Report"}])
    llm, invoke = analyst_llm([response])
    output = getattr(agents, factory)(llm)(state())
    assert output[report] == "Report"
    assert output["messages"][0] is response


@pytest.mark.parametrize("bad", [
    AIMessage(content="", tool_calls=[{"name": "get_news", "args": {}, "id": "a"}], response_metadata={"finish_reason": "length"}),
    AIMessage(content="", tool_calls=[{"name": "unknown_tool", "args": {}, "id": "a"}]),
    AIMessage(content="", tool_calls=[{"name": "get_news", "args": {}, "id": None}]),
    AIMessage(content="", tool_calls=[{"name": "get_news", "args": {}, "id": "a"}] * 2),
    AIMessage(content="I will fetch", response_metadata={"finish_reason": "tool_calls"}),
])
def test_bad_tool_requests_never_enter_tool_execution(bad):
    llm, invoke = analyst_llm([bad, bad])
    original = state()
    before = deepcopy(original)
    with pytest.raises(IncompleteResponseError):
        agents.create_news_analyst(llm)(original)
    assert original == before
    assert invoke.call_count == 2


def test_discarded_generation_does_not_duplicate_actual_tool_execution():
    from langchain_core.tools import tool
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from tradingagents.agents.utils.agent_states import AgentState
    executed = []

    @tool
    def get_news(ticker: str) -> str:
        """Fetch mocked ticker news."""
        executed.append(ticker)
        return "News evidence"

    call = {"name": "get_news", "args": {"ticker": "AAPL"}, "id": "call1"}
    llm, invoke = analyst_llm([
        AIMessage(content="", tool_calls=[call], response_metadata={"finish_reason": "length"}),
        AIMessage(content="", tool_calls=[call]), AIMessage(content="Complete report"),
    ])
    builder = StateGraph(AgentState)
    builder.add_node("analyst", agents.create_news_analyst(llm))
    builder.add_node("tools", ToolNode([get_news]))
    builder.add_edge(START, "analyst")
    builder.add_conditional_edges("analyst", lambda s: "tools" if s["messages"][-1].tool_calls else END)
    builder.add_edge("tools", "analyst")
    result = builder.compile().invoke(state())
    assert executed == ["AAPL"]
    assert invoke.call_count == 3
    assert result["news_report"] == "Complete report"
    assert sum(m.type == "tool" for m in result["messages"]) == 1


def test_failed_round_does_not_count_an_empty_contribution():
    from tradingagents.graph.debate_rounds import round_isolated_debater
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="")
    node = round_isolated_debater(agents.create_bull_researcher(llm), "investment_debate_state", "bull", ("bull", "bear"))
    original = state()
    before = deepcopy(original)
    with pytest.raises(IncompleteResponseError):
        node(original)
    assert original == before
    assert original["investment_debate_state"]["count"] == 0


def test_pending_tool_marker_is_not_prose_even_without_decoded_calls():
    from tradingagents.agents.utils.response_integrity import invoke_complete_text
    llm = Mock()
    llm.invoke.side_effect = [AIMessage(content="Planning", response_metadata={"stop_reason": "tool_use"}),
                              AIMessage(content="Complete")]
    assert invoke_complete_text(llm, "prompt") == "Complete"
    assert llm.invoke.call_count == 2


def test_response_policy_is_part_of_checkpoint_identity():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.selected_analysts = ("market",)
    assert "response_integrity_policy=1" in graph._run_signature("stock", "2024-01-15")


def test_tool_analyst_does_not_retry_transport_failure():
    llm, invoke = analyst_llm([TimeoutError("offline")])
    with pytest.raises(TimeoutError):
        agents.create_news_analyst(llm)(state())
    assert invoke.call_count == 1


def test_batch_narrative_does_not_publish_partial_text():
    from tradingagents.batch import BatchTickerResult, build_llm_narrative
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="PARTIAL", response_metadata={"finish_reason": "length"})
    assert build_llm_narrative([BatchTickerResult(ticker="AAPL", status="success")], None,
                               llm_factory=lambda _: llm) is None
    assert llm.invoke.call_count == 2
