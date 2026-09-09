"""News provenance, deduplication and routing failures are data contracts."""

import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tradingagents.dataflows import alpha_vantage_news, interface, yfinance_news
from tradingagents.dataflows.config import use_config, reset_config
from tradingagents.dataflows.errors import VendorError
from tradingagents.agents.analysts.sentiment_analyst import _build_system_message


def _article(url="https://example.com/story?utm_source=yahoo"):
    return {"content": {"title": "Quarterly results", "summary": "Revenue increased.",
        "provider": {"displayName": "Actual Publisher"}, "canonicalUrl": {"url": url},
        "pubDate": "2024-01-10T12:00:00Z"}}


def _mock_yahoo(monkeypatch, articles):
    monkeypatch.setattr(yfinance_news, "yf_retry", lambda call: call())
    monkeypatch.setattr(yfinance_news.yf, "Ticker", lambda *a: SimpleNamespace(get_news=lambda **k: articles))
    monkeypatch.setattr(yfinance_news.yf, "Search", lambda **k: SimpleNamespace(news=articles))


def test_yahoo_deduplicates_tracking_urls_and_renders_provenance(monkeypatch):
    _mock_yahoo(monkeypatch, [_article(), _article("https://example.com/story?utm_source=other#top")])
    result = yfinance_news.get_news_yfinance("AAPL", "2024-01-01", "2024-01-15")
    assert result.count("### Quarterly results") == 1
    assert "Retrieval vendor: yfinance" in result
    assert "Actual Publisher" in result
    assert "2024-01-10T12:00:00+00:00" in result
    assert re.search(r"NEWS-[0-9a-f]{16}", result)


def test_same_article_has_same_id_across_vendor_and_global_feed(monkeypatch):
    _mock_yahoo(monkeypatch, [_article()])
    payload = {"feed": [{"title": "Quarterly results", "summary": "Revenue increased.",
        "url": "https://example.com/story", "source": "Actual Publisher",
        "time_published": "20240110T120000"}]}
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", lambda *a: json.dumps(payload))
    yahoo = yfinance_news.get_news_yfinance("AAPL", "2024-01-01", "2024-01-15")
    global_news = yfinance_news.get_global_news_yfinance("2024-01-15")
    alpha = alpha_vantage_news.get_news("AAPL", "2024-01-01", "2024-01-15")
    ids = [re.search(r"NEWS-[0-9a-f]{16}", result) for result in (yahoo, global_news, alpha)]
    assert all(ids)
    assert len({match.group() for match in ids}) == 1
    assert "alpha_vantage" in alpha


@pytest.mark.parametrize("method,args", [("get_news_yfinance", ("AAPL", "2024-01-01", "2024-01-15")), ("get_global_news_yfinance", ("2024-01-15",))])
def test_yahoo_failure_is_typed_not_returned_as_news(monkeypatch, method, args):
    monkeypatch.setattr(yfinance_news, "yf_retry", Mock(side_effect=TimeoutError("offline")))
    with pytest.raises(VendorError):
        getattr(yfinance_news, method)(*args)


def test_yahoo_failure_reaches_configured_fallback(monkeypatch):
    monkeypatch.setattr(yfinance_news, "yf_retry", Mock(side_effect=TimeoutError("offline")))
    fallback = Mock(return_value="Fallback news")
    monkeypatch.setitem(interface.VENDOR_METHODS, "get_news", {
        "yfinance": yfinance_news.get_news_yfinance, "alpha_vantage": fallback,
    })
    token = use_config({"data_vendors": {"news_data": "yfinance,alpha_vantage"}})
    try:
        assert interface.route_to_vendor("get_news", "AAPL", "2024-01-01", "2024-01-15") == "Fallback news"
    finally:
        reset_config(token)
    fallback.assert_called_once()


def test_empty_recent_feed_does_not_claim_historical_silence(monkeypatch):
    _mock_yahoo(monkeypatch, [])
    text = yfinance_news.get_news_yfinance("AAPL", "2024-01-01", "2024-01-15")
    assert "coverage" in text.lower()
    assert "not evidence" in text.lower()


def test_matching_news_requests_reuse_one_snapshot_per_run(monkeypatch):
    from tradingagents.agents.utils import news_data_tools
    from tradingagents.dataflows import news_evidence
    route = Mock(side_effect=["snapshot one", "snapshot two"])
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    with news_evidence.news_run_scope():
        first = news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15")
        second = news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15")
        assert first == second == "snapshot one"
        assert route.call_count == 1
    with news_evidence.news_run_scope():
        assert news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15") == "snapshot two"
    assert route.call_count == 2


def test_news_cache_does_not_cache_failures_or_mix_windows(monkeypatch):
    from tradingagents.agents.utils import news_data_tools
    from tradingagents.dataflows import news_evidence
    route = Mock(side_effect=[VendorError("offline"), "recovered", "different window"])
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    with news_evidence.news_run_scope():
        with pytest.raises(VendorError):
            news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15")
        assert news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15") == "recovered"
        assert news_data_tools.get_news.func("AAPL", "2024-01-02", "2024-01-15") == "different window"
    assert route.call_count == 3


@pytest.mark.parametrize("prefetch", [True, False])
def test_tool_executor_shares_news_across_contexts_and_parallel_calls(monkeypatch, prefetch):
    from langchain_core.messages import AIMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode
    from tradingagents.agents.utils.agent_states import AgentState
    from tradingagents.agents.utils import news_data_tools
    from tradingagents.dataflows.news_evidence import news_run_scope
    route = Mock(return_value="shared snapshot")
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    builder = StateGraph(AgentState)
    builder.add_node("tools", ToolNode([news_data_tools.get_news]))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    with news_run_scope():
        if prefetch:
            news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15")
        result = builder.compile().invoke({"trade_date": "2024-01-15", "messages": [AIMessage(content="", tool_calls=[{
            "name": "get_news", "id": call_id,
            "args": {"ticker": "AAPL", "start_date": "2024-01-01", "end_date": "2024-01-15"}
        } for call_id in ("news1", "news2")])]})
    assert result["messages"][-1].content == "shared snapshot"
    assert route.call_count == 1


def test_cli_stream_scopes_news_cache_to_consumed_run(monkeypatch):
    from tradingagents.agents.utils import news_data_tools
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"checkpoint_enabled": False}
    route = Mock(return_value="shared snapshot")
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)

    def stream(*args, **kwargs):
        for _ in range(2):
            yield news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15")

    graph.graph = SimpleNamespace(stream=stream)
    for _ in range(2):
        assert list(graph.stream_with_checkpoint({}, {}, "AAPL", "2024-01-15")) == ["shared snapshot"] * 2
    assert route.call_count == 2  # once each run, not once per analyst or forever


def test_cache_key_respects_vendor_configuration(monkeypatch):
    from tradingagents.agents.utils import news_data_tools
    from tradingagents.dataflows.news_evidence import news_run_scope
    route = Mock(side_effect=["yahoo", "alpha"])
    monkeypatch.setattr(news_data_tools, "route_to_vendor", route)
    with news_run_scope():
        news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15")
        token = use_config({"data_vendors": {"news_data": "alpha_vantage"}})
        try:
            assert news_data_tools.get_news.func("AAPL", "2024-01-01", "2024-01-15") == "alpha"
        finally:
            reset_config(token)
    assert route.call_count == 2


def test_alpha_news_deduplicates_and_filters_end_boundary(monkeypatch):
    item = {"title": "Earnings", "url": "https://example.com/a", "source": "Publisher", "time_published": "20240115T230000"}
    payload = {"feed": [item, {**item, "url": "https://example.com/a?utm_source=x"},
                        {**item, "url": "https://example.com/b", "time_published": "20240116T000000"}]}
    request = Mock(return_value=json.dumps(payload))
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", request)
    result = json.loads(alpha_vantage_news.get_news("AAPL", "2024-01-01", "2024-01-15"))
    assert len(result["feed"]) == 1
    assert result["feed"][0]["publisher"] == "Publisher"
    assert request.call_args.args[1]["time_to"] == "20240116T0000"


@pytest.mark.parametrize("payload", ["not json", "{}", '{"feed": [null]}'])
def test_alpha_malformed_news_is_a_typed_failure(monkeypatch, payload):
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", lambda *a: payload)
    with pytest.raises(VendorError):
        alpha_vantage_news.get_news("AAPL", "2024-01-01", "2024-01-15")


def test_alpha_global_news_uses_config_defaults_and_marks_empty_coverage(monkeypatch):
    request = Mock(return_value='{"feed": []}')
    monkeypatch.setattr(alpha_vantage_news, "_make_api_request", request)
    token = use_config({"global_news_lookback_days": 3, "global_news_article_limit": 11})
    try:
        result = json.loads(alpha_vantage_news.get_global_news("2024-01-15", None, None))
    finally:
        reset_config(token)
    assert result["feed"] == []
    assert "absence of news" in result["coverage"]
    assert request.call_args.args[1]["time_from"] == "20240112T0000"
    assert request.call_args.args[1]["limit"] == "11"


@pytest.mark.parametrize("factory_name", [
    "create_bull_researcher", "create_bear_researcher", "create_research_manager", "create_portfolio_manager",
    "create_trader", "create_aggressive_debator", "create_conservative_debator", "create_neutral_debator",
])
def test_downstream_synthesis_does_not_treat_shared_news_as_independent(factory_name):
    from langchain_core.messages import AIMessage
    import tradingagents.agents as agents
    from tradingagents.graph.propagation import Propagator
    state = Propagator().create_initial_state("AAPL", "2024-01-15")
    state.update(investment_plan="plan", trader_investment_plan="proposal")
    llm = Mock()
    llm.with_structured_output.side_effect = NotImplementedError()
    llm.invoke.return_value = AIMessage(content="Rating: Hold")
    getattr(agents, factory_name)(llm)(state)
    assert "independent corroboration" in str(llm.invoke.call_args.args[0])


def test_sentiment_does_not_hardcode_news_vendor():
    prompt = _build_system_message(ticker="AAPL", start_date="2024-01-01", end_date="2024-01-15",
                                   news_block="Retrieval vendor: alpha_vantage", stocktwits_block="none", reddit_block="none")
    assert "Yahoo Finance" not in prompt
    assert "independent corroboration" in prompt
