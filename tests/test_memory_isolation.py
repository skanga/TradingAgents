"""Run-time memory must be namespace-isolated and bounded by known dates."""

import json
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.graph.trading_graph import TradingAgentsGraph


def _log(tmp_path, **config):
    return TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.jsonl"), **config})


def _resolve(log, namespace, lesson, date="2026-01-05", resolved="2026-01-12"):
    log.store_decision("AAPL", date, "Rating: Buy", namespace=namespace)
    log.batch_update_with_outcomes([{
        "ticker": "AAPL", "trade_date": date, "namespace": namespace,
        "raw_return": 0.02, "alpha_return": 0.01, "holding_days": 5,
        "reflection": lesson, "resolution_date": resolved,
    }])


def test_live_and_simulation_same_key_can_coexist_and_resolve_independently(tmp_path):
    log = _log(tmp_path)
    log.store_decision("AAPL", "2026-01-05", "Rating: Buy", namespace="live")
    _resolve(log, "simulation", "simulation lesson")
    assert len(log.load_entries()) == 2
    assert "Memory namespace: live" in log.format_entry(log.load_entries()[0])
    assert "Memory namespace: simulation" in log.format_entry(log.load_entries()[1])
    assert len(log.get_pending_entries(namespace="live")) == 1
    assert log.get_pending_entries(namespace="simulation") == []
    assert log.get_past_context("AAPL", as_of="2026-01-20", namespace="live") == ""
    _resolve(log, "live", "live lesson")
    reloaded = _log(tmp_path)
    for ticker in ("AAPL", "MSFT"):
        context = reloaded.get_past_context(ticker, as_of="2026-01-20", namespace="live")
        assert "live lesson" in context and "simulation lesson" not in context


def test_legacy_remains_readable_but_is_not_implicitly_live(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text(json.dumps({
        "version": 1, "date": "2026-01-05", "ticker": "AAPL", "rating": "Buy",
        "decision": "Rating: Buy", "pending": False, "reflection": "legacy lesson",
        "resolution_date": "2026-01-12",
    }), encoding="utf-8")
    log = _log(tmp_path)
    assert log.load_entries()[0]["namespace"] == "legacy"
    assert log.get_past_context("AAPL", namespace="live", as_of="2026-01-20") == ""
    log.store_decision("MSFT", "2026-01-20", "Rating: Hold", namespace="simulation")
    assert _log(tmp_path).load_entries()[0]["namespace"] == "legacy"
    assert "legacy lesson" in log.get_past_context("AAPL", namespace="legacy", as_of="2026-01-20")


def test_rotation_is_per_namespace(tmp_path):
    log = _log(tmp_path, memory_log_max_entries=1)
    _resolve(log, "live", "live lesson")
    _resolve(log, "simulation", "old simulation")
    _resolve(log, "simulation", "new simulation", date="2026-01-06")
    assert len(log.load_entries()) == 2
    assert "live lesson" in log.get_past_context("AAPL", namespace="live", as_of="2026-01-20")


def test_namespaced_context_requires_cutoff(tmp_path):
    with pytest.raises(ValueError, match="as_of"):
        _log(tmp_path).get_past_context("AAPL", namespace="live")


def test_cutoff_rejects_future_decision_even_if_resolution_is_backdated(tmp_path):
    log = _log(tmp_path)
    _resolve(log, "simulation", "future decision", date="2026-02-01", resolved="2026-01-12")
    assert log.get_past_context("AAPL", namespace="simulation", as_of="2026-01-20") == ""


def test_memory_cutoff_is_never_disabled_for_today_or_future():
    today = datetime.now().strftime("%Y-%m-%d")
    future = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
    assert TradingAgentsGraph._memory_as_of("2024-01-01") == "2024-01-01"
    assert TradingAgentsGraph._memory_as_of(today) == today
    assert TradingAgentsGraph._memory_as_of(future) == today


def test_pending_outcome_after_cutoff_never_invokes_reflector(tmp_path):
    log = _log(tmp_path)
    log.store_decision("AAPL", "2026-01-05", "Rating: Buy", namespace="simulation")
    graph = SimpleNamespace(
        memory_log=log, reflector=Mock(), _resolve_benchmark=lambda ticker: "SPY",
        _fetch_returns=Mock(return_value=(0.02, 0.01, 5, "2026-01-12", "2026-01-05")),
    )
    TradingAgentsGraph._resolve_pending_entries(graph, "AAPL", namespace="simulation", as_of="2026-01-10")
    graph.reflector.reflect_on_final_decision.assert_not_called()
    assert len(log.get_pending_entries(namespace="simulation")) == 1


def test_preparation_resolves_before_reading_with_one_scope(tmp_path):
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"memory_namespace": "simulation"}
    graph.memory_log = _log(tmp_path)
    _resolve(graph.memory_log, "live", "wrong scope")
    graph.memory_log.store_decision("AAPL", "2026-01-05", "Rating: Buy", namespace="simulation")
    graph._resolve_benchmark = lambda ticker: "SPY"
    graph._fetch_returns = Mock(return_value=(0.02, 0.01, 5, "2026-01-12", "2026-01-05"))
    graph.reflector = Mock()
    graph.reflector.reflect_on_final_decision.return_value = "newly resolved simulation lesson"
    context, namespace = graph.prepare_memory("AAPL", "2026-01-20")
    assert namespace == "simulation"
    assert "newly resolved simulation lesson" in context and "wrong scope" not in context


def test_namespace_selection_and_checkpoint_identity():
    today = datetime.now().strftime("%Y-%m-%d")
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.selected_analysts = ("market",)
    assert graph._memory_namespace(today) == "live"
    assert graph._memory_namespace("2024-01-01") == "simulation"
    assert graph._memory_namespace(today, "simulation") == "simulation"
    with pytest.raises(ValueError, match="memory_namespace"):
        graph._memory_namespace(today, "typo")
    live = graph._run_signature("stock", today)
    graph.config["memory_namespace"] = "simulation"
    assert graph._run_signature("stock", today) != live


def test_gui_stream_uses_shared_memory_boundary(tmp_path, monkeypatch):
    import tradingagents.graph.trading_graph as graph_module
    from gui import runner_worker
    from tradingagents.default_config import DEFAULT_CONFIG

    graph = Mock()
    graph.prepare_memory.return_value = ("safe context", "simulation")
    graph.propagator.get_graph_args.return_value = {}
    graph.graph.stream.return_value = [{"final_trade_decision": "Rating: Hold"}]
    graph.process_signal.return_value = "Hold"
    factory = Mock(return_value=graph)
    monkeypatch.setattr(graph_module, "TradingAgentsGraph", factory)
    monkeypatch.setitem(DEFAULT_CONFIG, "results_dir", str(tmp_path))
    monkeypatch.setattr(runner_worker, "emit", Mock())
    runner_worker.run({"ticker": "AAPL", "trade_date": "2026-01-20", "memory_namespace": "simulation"})
    graph.prepare_memory.assert_called_once_with("AAPL", "2026-01-20")
    graph.propagator.create_initial_state.assert_called_once_with("AAPL", "2026-01-20", past_context="safe context")
    graph.memory_log.get_past_context.assert_not_called()
    graph.memory_log.store_decision.assert_called_once_with(
        ticker="AAPL", trade_date="2026-01-20", final_trade_decision="Rating: Hold", namespace="simulation"
    )
    assert factory.call_args.kwargs["config"]["memory_namespace"] == "simulation"


@pytest.mark.parametrize("resolved", ["2026-01-0x", None])
def test_unverifiable_resolution_date_never_reaches_reflector(tmp_path, resolved):
    log = _log(tmp_path)
    log.store_decision("AAPL", "2026-01-05", "Rating: Buy", namespace="live")
    graph = SimpleNamespace(
        memory_log=log, reflector=Mock(), _resolve_benchmark=lambda ticker: "SPY",
        _fetch_returns=Mock(return_value=(0.02, 0.01, 5, resolved, "2026-01-05")),
    )
    graph.reflector.reflect_on_final_decision.return_value = "Should not be generated"
    TradingAgentsGraph._resolve_pending_entries(graph, "AAPL", namespace="live", as_of="2026-01-20")
    graph.reflector.reflect_on_final_decision.assert_not_called()


def test_missing_resolution_date_from_custom_fetcher_stays_pending(tmp_path):
    log = _log(tmp_path)
    log.store_decision("AAPL", "2026-01-05", "Rating: Buy", namespace="live")
    graph = SimpleNamespace(
        memory_log=log, reflector=Mock(), _resolve_benchmark=lambda ticker: "SPY",
        _fetch_returns=Mock(return_value=(0.02, 0.01, 5)),
    )
    TradingAgentsGraph._resolve_pending_entries(graph, "AAPL", namespace="live", as_of="2026-01-20")
    graph.reflector.reflect_on_final_decision.assert_not_called()
    assert len(log.get_pending_entries(namespace="live")) == 1
