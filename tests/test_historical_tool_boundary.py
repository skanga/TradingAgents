"""Historical data requires trusted execution dates and known publication vintages."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents.utils import (
    agent_utils,
    core_stock_tools,
    fundamental_data_tools,
    macro_data_tools,
    market_data_validation_tools,
    news_data_tools,
    prediction_markets_tools,
    technical_indicators_tools,
)
from tradingagents.agents.utils.agent_states import AgentState
from tradingagents.dataflows import alpha_vantage_fundamentals, y_finance


@pytest.mark.parametrize("supplied", [None, "2099-01-01"])
def test_executor_injects_trusted_cutoff_even_when_model_omits_or_spoofs_date(monkeypatch, supplied):
    route = Mock(return_value="data")
    monkeypatch.setattr(fundamental_data_tools, "route_to_vendor", route)
    builder = StateGraph(AgentState)
    builder.add_node("tools", ToolNode([fundamental_data_tools.get_balance_sheet]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    args = {"ticker": "AAPL", "state": {"trade_date": "2099-01-01"}}
    if supplied:
        args["curr_date"] = supplied
    builder.compile().invoke({
        "trade_date": "2024-01-15",
        "messages": [AIMessage(content="", tool_calls=[{"name": "get_balance_sheet", "args": args, "id": "call1"}])],
    })
    route.assert_called_once_with("get_balance_sheet", "AAPL", "quarterly", "2024-01-15")


@pytest.mark.parametrize("module", [y_finance, alpha_vantage_fundamentals])
@pytest.mark.parametrize("name", ["get_balance_sheet", "get_cashflow", "get_income_statement"])
def test_historical_statements_without_vintage_are_withheld_before_network(monkeypatch, module, name):
    if module is y_finance:
        request = Mock(side_effect=AssertionError("must not fetch current statements"))
        monkeypatch.setattr(module.yf, "Ticker", request)
    else:
        request = Mock(side_effect=AssertionError("must not fetch current statements"))
        monkeypatch.setattr(module, "_make_api_request", request)
    result = getattr(module, name)("AAPL", curr_date="2024-01-15")
    assert "withheld" in result.lower()
    assert "publication" in result.lower()
    assert "fiscal" in result.lower()
    request.assert_not_called()


@pytest.mark.parametrize("module, name, args, expected", [
    (core_stock_tools, "get_stock_data", ("AAPL", "2024-01-01", "2099-01-01"), ("get_stock_data", "AAPL", "2024-01-01", "2024-01-15")),
    (technical_indicators_tools, "get_indicators", ("AAPL", "rsi", "2099-01-01"), ("get_indicators", "AAPL", "rsi", "2024-01-15", 30)),
    (macro_data_tools, "get_macro_indicators", ("cpi", "2099-01-01"), ("get_macro_indicators", "cpi", "2024-01-15", None)),
    (news_data_tools, "get_global_news", ("2099-01-01",), ("get_global_news", "2024-01-15", None, None)),
])
def test_all_dated_tools_cap_future_arguments(monkeypatch, module, name, args, expected):
    route = Mock(return_value="data")
    monkeypatch.setattr(module, "route_to_vendor", route)
    getattr(module, name).func(*args, state={"trade_date": "2024-01-15"})
    route.assert_called_once_with(*expected)


def test_verified_snapshot_is_capped(monkeypatch):
    snapshot = Mock(return_value="snapshot")
    monkeypatch.setattr(market_data_validation_tools, "build_verified_market_snapshot", snapshot)
    market_data_validation_tools.get_verified_market_snapshot.func("AAPL", "2099-01-01", state={"trade_date": "2024-01-15"})
    snapshot.assert_called_once_with("AAPL", "2024-01-15", 30)


@pytest.mark.parametrize("module, name", [(prediction_markets_tools, "get_prediction_markets"), (news_data_tools, "get_insider_transactions")])
def test_live_only_tools_do_not_fetch_for_historical_runs(monkeypatch, module, name):
    route = Mock(return_value="current data")
    monkeypatch.setattr(module, "route_to_vendor", route)
    result = getattr(module, name).func("AAPL", state={"trade_date": "2024-01-15"})
    assert "withheld" in result
    route.assert_not_called()


def test_historical_identity_does_not_fetch_current_company_profile(monkeypatch):
    request = Mock(side_effect=AssertionError("current profile must not be fetched"))
    monkeypatch.setattr(agent_utils.yf, "Ticker", request)
    assert agent_utils.resolve_instrument_identity("TESTHISTORY", curr_date="2024-01-15") == {}
    request.assert_not_called()


def test_tool_state_is_hidden_from_model_schema():
    for module, name in [
        (fundamental_data_tools, "get_balance_sheet"), (news_data_tools, "get_news"),
        (core_stock_tools, "get_stock_data"), (technical_indicators_tools, "get_indicators"),
        (macro_data_tools, "get_macro_indicators"), (market_data_validation_tools, "get_verified_market_snapshot"),
        (prediction_markets_tools, "get_prediction_markets"),
    ]:
        schema = getattr(module, name).tool_call_schema.model_json_schema()
        assert "state" not in schema["properties"]


def test_invalid_range_is_rejected_before_vendor(monkeypatch):
    route = Mock()
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    with pytest.raises(ValueError, match="cutoff"):
        news_data_tools.get_news.func("AAPL", "2025-01-01", "2099-01-01", state={"trade_date": "2024-01-15"})
    route.assert_not_called()


def test_sentiment_prefetch_caps_future_run_date(monkeypatch):
    from tradingagents.agents.analysts import sentiment_analyst
    from tradingagents.agents.utils import tool_dates
    monkeypatch.setattr(tool_dates, "get_current_date", lambda: "2024-01-15")
    news = Mock(return_value="news")
    monkeypatch.setattr(sentiment_analyst.get_news, "func", news)
    monkeypatch.setattr(sentiment_analyst, "fetch_stocktwits_messages", Mock(return_value="none"))
    monkeypatch.setattr(sentiment_analyst, "fetch_reddit_posts", Mock(return_value="none"))
    llm = Mock()
    llm.with_structured_output.side_effect = NotImplementedError()
    llm.invoke.return_value = AIMessage(content="Neutral; insufficient evidence.")
    sentiment_analyst.create_sentiment_analyst(llm)({"company_of_interest": "AAPL", "trade_date": "2099-01-01", "messages": []})
    news.assert_called_once_with("AAPL", "2024-01-08", "2024-01-15")


def test_news_range_cannot_exceed_run_date(monkeypatch):
    route = Mock(return_value="news")
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    news_data_tools.get_news.func("AAPL", "2024-01-01", "2099-01-01", state={"trade_date": "2024-01-15"})
    route.assert_called_once_with("get_news", "AAPL", "2024-01-01", "2024-01-15")
