"""Regression cases for invalid decisions and deterministic outcome evaluation."""

import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
import pytest
from langchain_core.messages import AIMessage

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.agents.utils.rating import parse_actionable_rating
from tradingagents.graph.reflection import Reflector
from tradingagents.graph.signal_processing import SignalProcessor
from tradingagents.graph.trading_graph import TradingAgentsGraph


@pytest.mark.parametrize("text", [
    "No recommendation available.",
    "The buy argument and sell argument are both plausible.",
    "Rating: Unknown\nThe buy thesis looks good.",
    "Rating: Buy\nRating: Sell",
    "Rating: Buy or Sell",
    "Rating: Buy/Hold/Sell",
    "Do not buy.",
])
def test_invalid_decisions_are_review_and_not_stored(tmp_path, text):
    assert parse_actionable_rating(text) == "REVIEW"
    assert SignalProcessor().process_signal(text) == "REVIEW"
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.jsonl")})
    log.store_decision("AAPL", "2026-01-05", text)
    assert log.load_entries() == []


@pytest.mark.parametrize("text, expected", [
    ("**Rating**: Buy\nSell thesis is weak.", "Buy"),
    ("1. Rating: **Sell**", "Sell"),
    ("Rating：Overweight", "Overweight"),
    ("Hold", "Hold"),
    ("Rating: Underweight\nRating: Underweight", "Underweight"),
])
def test_unambiguous_decisions_remain_supported(text, expected):
    assert parse_actionable_rating(text) == expected


def test_invalid_pending_legacy_record_does_not_block_corrected_decision(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text(json.dumps({
        "version": 1, "date": "2026-01-05", "ticker": "AAPL", "rating": "Hold",
        "pending": True, "decision": "No recommendation available.",
    }), encoding="utf-8")
    log = TradingMemoryLog({"memory_log_path": str(path)})
    assert log.get_pending_entries() == []
    log.store_decision("AAPL", "2026-01-05", "Rating: Sell")
    assert len(log.get_pending_entries()) == 1
    assert log.get_pending_entries()[0]["rating"] == "Sell"
    assert len(log.load_entries()) == 2  # retain the legacy audit record


def test_batch_result_uses_review_for_invalid_decision():
    from tradingagents.batch import PortfolioHolding, extract_ticker_result
    result = extract_ticker_result(
        ticker="AAPL", final_state={"final_trade_decision": "Do not buy."},
        report_path=None, holding=PortfolioHolding(ticker="AAPL"), elapsed_seconds=1.0,
    )
    assert result.rating == "REVIEW"


def test_invalid_legacy_entry_remains_readable_but_not_learnable(tmp_path):
    path = tmp_path / "memory.jsonl"
    path.write_text(json.dumps({
        "version": 1, "date": "2026-01-05", "ticker": "AAPL", "rating": "Hold",
        "pending": False, "decision": "No recommendation available.",
        "reflection": "Invented lesson", "raw": "+1%", "alpha": "+0%", "holding": "5d",
    }), encoding="utf-8")
    log = TradingMemoryLog({"memory_log_path": str(path)})
    assert len(log.load_entries()) == 1
    assert log.get_past_context("AAPL") == ""


@pytest.mark.parametrize("rating, raw, alpha, direction, relative", [
    ("Buy", -0.05, 0.05, "incorrect", "favorable"),
    ("Overweight", 0.05, -0.03, "correct", "unfavorable"),
    ("Sell", -0.05, 0.05, "correct", "unfavorable"),
    ("Underweight", 0.05, -0.03, "incorrect", "favorable"),
    ("Hold", 0.10, 0.04, "not_applicable", "not_applicable"),
    ("Buy", 0.0, 0.0, "flat", "flat"),
])
def test_scoring_is_persisted_separately_from_llm_prose(tmp_path, rating, raw, alpha, direction, relative):
    path = tmp_path / "memory.jsonl"
    log = TradingMemoryLog({"memory_log_path": str(path)})
    log.store_decision("AAPL", "2026-01-05", f"Rating: {rating}")
    log.batch_update_with_outcomes([{
        "ticker": "AAPL", "trade_date": "2026-01-05", "raw_return": raw,
        "alpha_return": alpha, "holding_days": 5, "reflection": "An LLM lesson.",
        "resolution_date": "2026-01-12", "benchmark_name": "SPY",
    }])
    reloaded = TradingMemoryLog({"memory_log_path": str(path)})
    outcome = reloaded.load_entries()[0]["outcome"]
    assert outcome["directional_accuracy"] == direction
    assert outcome["relative_assessment"] == relative
    assert outcome["asset_return"] == raw
    assert outcome["benchmark_return"] == pytest.approx(raw - alpha)
    assert outcome["excess_return"] == alpha
    assert outcome["evaluation_sessions"] == 5
    assert direction in reloaded.get_past_context("AAPL")
    assert relative in reloaded.get_past_context("MSFT")


def test_reflection_receives_classification_not_alpha_as_direction():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="The call lost value but outperformed.")
    Reflector(llm).reflect_on_final_decision("Rating: Buy", -0.05, 0.05)
    messages = llm.invoke.call_args.args[0]
    prompt = "\n".join(text for _, text in messages)
    assert "Directional accuracy: incorrect" in prompt
    assert "Relative assessment: favorable" in prompt
    assert "not realized trade P&L" in prompt
    assert "insufficient evidence" in prompt


def _fetch(stock, bench, **kwargs):
    with patch("tradingagents.graph.trading_graph.yf.Ticker") as ticker:
        ticker.side_effect = lambda symbol: Mock(history=Mock(return_value=bench if symbol == "SPY" else stock))
        return TradingAgentsGraph._fetch_returns(None, "AAPL", "2026-01-05", holding_days=2, include_resolution=True, **kwargs)


def test_returns_use_same_endpoint_dates_not_same_row_numbers():
    stock = pd.DataFrame({"Close": [100., 105., 110.]}, index=pd.to_datetime(["2026-01-05", "2026-01-07", "2026-01-08"]))
    bench = pd.DataFrame({"Close": [100., 101., 103., 104.]}, index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"]))
    raw, alpha, days, resolved = _fetch(stock, bench)
    assert raw == pytest.approx(0.10)
    assert alpha == pytest.approx(0.06)
    assert days == 2 and resolved == "2026-01-08"


@pytest.mark.parametrize("prices", [[100., 101., float("nan")], [0., 101., 105.], [100., 101., float("inf")]])
def test_invalid_endpoint_prices_do_not_generate_outcomes(prices):
    dates = pd.date_range("2026-01-05", periods=3)
    stock = pd.DataFrame({"Close": prices}, index=dates)
    bench = pd.DataFrame({"Close": [100., 101., 102.]}, index=dates)
    assert _fetch(stock, bench) == (None, None, None, None)


def test_evaluation_window_uses_actual_sessions_after_weekend():
    dates = pd.date_range("2026-01-05", periods=3)
    frame = pd.DataFrame({"Close": [100., 101., 102.]}, index=dates)
    result = _fetch(frame, frame, include_window=True)
    assert result == (0.02, 0.0, 2, "2026-01-07", "2026-01-05")


def test_resolution_persists_window_and_keeps_cached_scores_isolated(tmp_path):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.jsonl")})
    log.store_decision("AAPL", "2026-01-03", "Rating: Buy")
    llm = Mock()
    llm.invoke.return_value = AIMessage(content="Direction correct; thesis evidence insufficient.")
    graph = SimpleNamespace(
        memory_log=log, reflector=Reflector(llm), _resolve_benchmark=lambda t: "SPY",
        _fetch_returns=lambda *a, **k: (0.05, 0.02, 5, "2026-01-12", "2026-01-05"),
    )
    TradingAgentsGraph._resolve_pending_entries(graph, "AAPL")
    outcome = log.load_entries()[0]["outcome"]
    assert outcome["evaluation_start"] == "2026-01-05"
    assert outcome["resolution_date"] == "2026-01-12"
    assert outcome["benchmark_name"] == "SPY"
    outcome["directional_accuracy"] = "corrupted"
    assert log.load_entries()[0]["outcome"]["directional_accuracy"] == "correct"
    assert "2026-01-05" in log.get_past_context("MSFT")


@pytest.mark.parametrize("raw, alpha", [(float("nan"), 0.0), (0.0, float("inf"))])
def test_nonfinite_returns_cannot_resolve_memory(tmp_path, raw, alpha):
    log = TradingMemoryLog({"memory_log_path": str(tmp_path / "memory.jsonl")})
    log.store_decision("AAPL", "2026-01-05", "Rating: Buy")
    with pytest.raises(ValueError, match="finite"):
        log.batch_update_with_outcomes([{
            "ticker": "AAPL", "trade_date": "2026-01-05", "raw_return": raw,
            "alpha_return": alpha, "holding_days": 5, "reflection": "Unusable",
        }])
    assert len(log.get_pending_entries()) == 1


def test_missing_benchmark_endpoint_does_not_shift_horizon():
    stock = pd.DataFrame({"Close": [100., 105., 110.]}, index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-07"]))
    bench = pd.DataFrame({"Close": [100., 103., 104.]}, index=pd.to_datetime(["2026-01-05", "2026-01-06", "2026-01-08"]))
    assert _fetch(stock, bench) == (None, None, None, None)
