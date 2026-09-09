"""Neutral evidence and equal information access, verified without live models."""
from dataclasses import replace
from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

import tradingagents.agents as agents
from tradingagents.graph.propagation import Propagator
from tradingagents.graph.conditional_logic import ConditionalLogic


@pytest.mark.parametrize("factory", ["create_news_analyst", "create_market_analyst", "create_fundamentals_analyst", "create_sentiment_analyst"])
def test_analysts_request_evidence_not_transaction_proposals(monkeypatch, factory):
    import tradingagents.agents.analysts.sentiment_analyst as sentiment
    monkeypatch.setattr(sentiment.get_news, "func", lambda *a: "no news")
    monkeypatch.setattr(sentiment, "fetch_stocktwits_messages", lambda *a, **k: "no posts")
    monkeypatch.setattr(sentiment, "fetch_reddit_posts", lambda *a, **k: "no posts")
    captured = []
    llm = Mock()
    llm.bind_tools.return_value = RunnableLambda(lambda p: captured.append(p.to_messages()) or AIMessage(content="report"))
    llm.with_structured_output.side_effect = NotImplementedError()
    llm.invoke.side_effect = lambda p: captured.append(p) or AIMessage(content="report")
    getattr(agents, factory)(llm)(Propagator().create_initial_state("AAPL", "2024-01-15"))
    text = "\n".join(m.content for m in captured[0])
    assert "FINAL TRANSACTION PROPOSAL" not in text
    assert "Nvidia" not in text and "NVDA" not in text
    assert "evidence" in text.lower()


def test_neutral_is_substantive_and_missing_data_is_not_neutral():
    from tradingagents.agents.schemas import SentimentReport
    from tradingagents.agents.analysts.sentiment_analyst import _build_system_message
    schema = SentimentReport.model_fields["overall_band"].description
    prompt = _build_system_message(ticker="AAPL", start_date="2024-01-01", end_date="2024-01-15",
                                   news_block="", stocktwits_block="", reddit_block="")
    for text in (schema, prompt):
        assert "no net directional" in text
        assert "Missing data" in text
        assert "Neutral only" not in text


def _workflow(monkeypatch, rounds, **compile_kwargs):
    import tradingagents.graph.setup as setup
    captured = {speaker: [] for speaker in ("Bull", "Bear", "Aggressive", "Conservative", "Neutral")}
    llm = Mock()

    def invoke(prompt):
        speaker = next(s for s in captured if (f"a {s} Analyst" in prompt or f"the {s} Risk Analyst" in prompt))
        captured[speaker].append(prompt)
        return AIMessage(content=f"UNIQUE_{speaker}_{len(captured[speaker])}")

    llm.invoke.side_effect = invoke
    spec = setup.ANALYST_SPECS["market"]
    monkeypatch.setitem(setup.ANALYST_SPECS, "market", replace(spec, create_node=lambda llm: lambda state: {
        "market_report": "Balanced facts", "messages": [AIMessage(content="facts")],
    }))
    monkeypatch.setattr(setup, "create_research_manager", lambda llm: lambda state: {"investment_plan": "plan"})
    monkeypatch.setattr(setup, "create_trader", lambda llm: lambda state: {"trader_investment_plan": "plan"})
    monkeypatch.setattr(setup, "create_portfolio_manager", lambda llm: lambda state: {"final_trade_decision": "Rating: Hold"})
    workflow = setup.GraphSetup(llm, llm, {"market": lambda state: {}}, ConditionalLogic(rounds, rounds)).setup_graph(["market"])
    return workflow.compile(**compile_kwargs), captured


@pytest.mark.parametrize("rounds", [1, 2, 3])
def test_actual_graph_gives_independent_openings_and_equal_completed_rounds(monkeypatch, rounds):
    graph, captured = _workflow(monkeypatch, rounds)
    result = graph.invoke(Propagator().create_initial_state("AAPL", "2024-01-15"), {"recursion_limit": 100})
    for speaker, prompts in captured.items():
        assert len(prompts) == rounds
        assert "UNIQUE_" not in prompts[0]  # every opener uses only analyst evidence
        for index, prompt in enumerate(prompts[1:], start=2):
            for peer in captured:
                assert f"UNIQUE_{peer}_{index}" not in prompt
    assert result["investment_debate_state"]["count"] == 2 * rounds
    assert result["risk_debate_state"]["count"] == 3 * rounds
    for speaker in captured:
        key = "investment_debate_state" if speaker in ("Bull", "Bear") else "risk_debate_state"
        for turn in range(1, rounds + 1):
            assert result[key]["history"].count(f"UNIQUE_{speaker}_{turn}") == 1
    if rounds > 1:
        assert "UNIQUE_Bull_1" in captured["Bear"][1]
        assert "UNIQUE_Bear_1" in captured["Bull"][1]
        assert "UNIQUE_Neutral_1" in captured["Aggressive"][1]


def test_mid_round_checkpoint_resume_retains_blind_opening(monkeypatch):
    from langgraph.checkpoint.memory import InMemorySaver
    graph, captured = _workflow(monkeypatch, 1, checkpointer=InMemorySaver(), interrupt_after=["Bull Researcher"])
    config = {"configurable": {"thread_id": "resume-round"}}
    partial = graph.invoke(Propagator().create_initial_state("AAPL", "2024-01-15"), config)
    assert partial["investment_debate_state"]["count"] == 1
    final = graph.invoke(None, config)
    assert len(captured["Bull"]) == len(captured["Bear"]) == 1
    assert "UNIQUE_" not in captured["Bear"][0]
    assert final["investment_debate_state"]["count"] == 2


@pytest.mark.parametrize("factory,key,roles", [
    ("create_research_manager", "investment_debate_state", ("bull", "bear")),
    ("create_portfolio_manager", "risk_debate_state", ("aggressive", "conservative", "neutral")),
])
def test_judge_gives_each_role_equal_budget_and_ignores_chronological_order(factory, key, roles):
    llm = Mock()
    llm.with_structured_output.side_effect = NotImplementedError()
    llm.invoke.return_value = AIMessage(content="Rating: Hold")
    node = getattr(agents, factory)(llm)
    state = Propagator().create_initial_state("AAPL", "2024-01-15")
    state.update(investment_plan="plan", trader_investment_plan="plan")
    histories = [f"ROLE_{role}: " + "x" * 15000 for role in roles]
    state[key].update({f"{role}_history": text for role, text in zip(roles, histories)})
    state[key]["history"] = "\n".join(histories)
    node(state)
    first = llm.invoke.call_args.args[0]
    for role in roles:
        assert f"ROLE_{role}:" in first
    assert len(first) < 20000
    state[key]["history"] = "\n".join(reversed(histories))
    node(state)
    assert llm.invoke.call_args.args[0] == first


@pytest.mark.parametrize("factory", ["create_bull_researcher", "create_bear_researcher", "create_aggressive_debator", "create_conservative_debator", "create_neutral_debator"])
def test_debate_roles_can_concede_unsupported_claims(factory):
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="argument")
    state = Propagator().create_initial_state("AAPL", "2024-01-15")
    state["trader_investment_plan"] = "plan"
    getattr(agents, factory)(llm)(state)
    assert "concede unsupported claims" in llm.invoke.call_args.args[0]


@pytest.mark.parametrize("key,roles,factories", [
    ("investment_debate_state", ("bull", "bear"), ("create_bull_researcher", "create_bear_researcher")),
    ("risk_debate_state", ("aggressive", "conservative", "neutral"),
     ("create_aggressive_debator", "create_conservative_debator", "create_neutral_debator")),
])
def test_counterbalanced_openings_have_identical_role_prompts(key, roles, factories):
    from itertools import permutations
    from tradingagents.graph.debate_rounds import round_isolated_debater
    baseline = None
    for order in permutations(roles):
        prompts = {}
        nodes = {}
        for role, factory in zip(roles, factories):
            llm = Mock()
            llm.invoke.side_effect = lambda prompt, role=role: prompts.setdefault(role, prompt) and AIMessage(content=f"opening-{role}")
            nodes[role] = round_isolated_debater(getattr(agents, factory)(llm), key, role, roles)
        state = Propagator().create_initial_state("AAPL", "2024-01-15")
        state["trader_investment_plan"] = "plan"
        for role in order:
            state.update(nodes[role](state))
        if baseline is None:
            baseline = prompts
        assert prompts == baseline


def test_duplicate_speaker_and_missing_midround_snapshot_fail_closed():
    from tradingagents.graph.debate_rounds import round_isolated_debater
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="argument")
    node = round_isolated_debater(agents.create_bull_researcher(llm), "investment_debate_state", "bull", ("bull", "bear"))
    state = Propagator().create_initial_state("AAPL", "2024-01-15")
    state.update(node(state))
    with pytest.raises(ValueError, match="exactly once"):
        node(state)
    del state["investment_debate_state"]["round_snapshot"]
    with pytest.raises(ValueError, match="snapshot"):
        node(state)
    assert llm.invoke.call_count == 1
    assert state["investment_debate_state"]["count"] == 1


def test_debate_policy_is_in_checkpoint_identity():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.selected_analysts = ("market",)
    assert "debate_integrity_policy=1" in graph._run_signature("stock", "2024-01-15")


def test_shared_analyst_template_does_not_request_a_trade_proposal():
    from tradingagents.agents.utils.prompts import load_prompt_template
    assert "FINAL TRANSACTION PROPOSAL" not in load_prompt_template("tool_collaboration_system.md")


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_round_counts_rejected(value):
    with pytest.raises(ValueError):
        ConditionalLogic(max_debate_rounds=value)
    with pytest.raises(ValueError):
        ConditionalLogic(max_risk_discuss_rounds=value)
