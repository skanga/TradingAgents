"""Per-role LLM plumbing: TradingAgentsGraph builds a role-keyed LLM map
that GraphSetup uses to route each agent factory to its assigned model.
"""

from unittest.mock import MagicMock, patch

import pytest


def _make_factory(model_to_llm: dict):
    """Patch helper: each create_llm_client call returns a client whose
    get_llm() yields the LLM keyed by the requested model string."""

    def factory(provider, model, base_url=None, **kwargs):
        llm = model_to_llm.setdefault(model, MagicMock(name=f"llm:{model}"))
        client = MagicMock(name=f"client:{model}")
        client.get_llm.return_value = llm
        return client

    return factory


def _config(**overrides) -> dict:
    base = {
        "llm_provider": "openrouter",
        "deep_think_llm": "deep-model",
        "quick_think_llm": "quick-model",
        "backend_url": "https://example.test/v1",
        "results_dir": "/tmp/ta-results",
        "data_cache_dir": "/tmp/ta-cache",
        "memory_log_path": "/tmp/ta-memory.md",
        "memory_log_max_entries": None,
        "checkpoint_enabled": False,
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1,
        "max_recur_limit": 100,
        "output_language": "English",
        "data_vendors": {
            "core_stock_apis": "yfinance",
            "technical_indicators": "yfinance",
            "fundamental_data": "yfinance",
            "news_data": "yfinance",
            "political_data": "finnhub",
            "options_data": "yfinance",
            "macro_data": "fred",
            "transcript_data": "motley_fool",
            "sector_data": "yfinance",
        },
        "tool_vendors": {},
        "fred_api_key": "",
        "sec_user_agent": "test ua",
        "finnhub_api_key": "",
        "enable_options_analyst": False,
        "google_thinking_level": None,
        "openai_reasoning_effort": None,
        "anthropic_effort": None,
    }
    base.update(overrides)
    return base


@pytest.fixture
def graph_with_factory():
    """Construct a TradingAgentsGraph with create_llm_client mocked.

    Returns a callable that takes a config dict + the model→llm map and
    returns the constructed graph instance.
    """
    def _build(model_to_llm: dict, *, selected_analysts=("market",), **config_overrides):
        config = _config(**config_overrides)
        with patch(
            "tradingagents.llm_clients.factory.create_llm_client",
            side_effect=_make_factory(model_to_llm),
        ), patch(
            # The graph imports create_llm_client at module top — patch the
            # binding inside graph.trading_graph too.
            "tradingagents.graph.trading_graph.create_llm_client",
            side_effect=_make_factory(model_to_llm),
        ):
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            return TradingAgentsGraph(
                config=config,
                selected_analysts=list(selected_analysts),
            )
    return _build


def test_role_map_with_all_overrides_uses_distinct_llms(graph_with_factory):
    seen: dict = {}
    ta = graph_with_factory(
        seen,
        structured_output_llm="struct-model",
        quant_llm="quant-model",
        light_llm="light-model",
    )
    rl = ta.role_llms
    # Every role key resolves
    for k in ("deep", "quick", "structured_output", "quant", "light"):
        assert k in rl and rl[k] is not None
    # Distinct LLMs for distinct models
    assert rl["deep"] is seen["deep-model"]
    assert rl["quick"] is seen["quick-model"]
    assert rl["structured_output"] is seen["struct-model"]
    assert rl["quant"] is seen["quant-model"]
    assert rl["light"] is seen["light-model"]


def test_role_map_falls_back_when_role_keys_unset(graph_with_factory):
    seen: dict = {}
    ta = graph_with_factory(
        seen,
        structured_output_llm="",
        quant_llm="",
        light_llm="",
    )
    rl = ta.role_llms
    # structured_output → deep ; quant/light → quick
    assert rl["structured_output"] is rl["deep"]
    assert rl["quant"] is rl["quick"]
    assert rl["light"] is rl["quick"]


def test_role_map_caches_clients_for_duplicate_models(graph_with_factory):
    seen: dict = {}
    ta = graph_with_factory(
        seen,
        structured_output_llm="deep-model",   # same as deep_think_llm
        quant_llm="shared-extra",
        light_llm="shared-extra",             # same as quant_llm → cache hit
    )
    rl = ta.role_llms
    # structured_output reuses the deep LLM (no new client built)
    assert rl["structured_output"] is rl["deep"]
    # quant + light share the same extra-model LLM
    assert rl["quant"] is rl["light"]
    assert rl["quant"] is seen["shared-extra"]


def test_graph_setup_routes_factories_through_role_map(graph_with_factory):
    """GraphSetup hands the right LLM to each create_* factory.

    We patch the per-agent factories to capture which LLM they receive.
    """
    captured: dict = {}

    def _capture(name):
        def factory(llm):
            captured[name] = llm
            return MagicMock(name=f"node:{name}")
        return factory

    # Patch every agent factory used by setup_graph
    patches = {
        "create_market_analyst": _capture("market"),
        "create_social_media_analyst": _capture("social"),
        "create_news_analyst": _capture("news"),
        "create_fundamentals_analyst": _capture("fundamentals"),
        "create_options_analyst": _capture("options"),
        "create_bull_researcher": _capture("bull"),
        "create_bear_researcher": _capture("bear"),
        "create_research_manager": _capture("research_mgr"),
        "create_trader": _capture("trader"),
        "create_aggressive_debator": _capture("aggressive"),
        "create_neutral_debator": _capture("neutral"),
        "create_conservative_debator": _capture("conservative"),
        "create_portfolio_manager": _capture("portfolio_mgr"),
    }

    with patch.multiple("tradingagents.graph.setup", **patches):
        seen: dict = {}
        ta = graph_with_factory(
            seen,
            structured_output_llm="struct-model",
            quant_llm="quant-model",
            light_llm="light-model",
            enable_options_analyst=True,
            selected_analysts=("market", "social", "news", "fundamentals"),
        )

    # Quant model: market, options, all 3 risk debaters
    quant_llm = ta.role_llms["quant"]
    assert captured["market"] is quant_llm
    assert captured["options"] is quant_llm
    assert captured["aggressive"] is quant_llm
    assert captured["neutral"] is quant_llm
    assert captured["conservative"] is quant_llm

    # Light model: social, news
    light_llm = ta.role_llms["light"]
    assert captured["social"] is light_llm
    assert captured["news"] is light_llm

    # Quick model: fundamentals, bull, bear, trader
    quick_llm = ta.role_llms["quick"]
    assert captured["fundamentals"] is quick_llm
    assert captured["bull"] is quick_llm
    assert captured["bear"] is quick_llm
    assert captured["trader"] is quick_llm

    # Structured-output model: research mgr, portfolio mgr
    struct_llm = ta.role_llms["structured_output"]
    assert captured["research_mgr"] is struct_llm
    assert captured["portfolio_mgr"] is struct_llm
