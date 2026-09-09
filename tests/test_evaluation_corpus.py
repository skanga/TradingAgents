"""Synthetic corpus acceptance tests; no provider or captured-model claims."""
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tradingagents import evaluation as ev

CORPUS = Path(__file__).resolve().parents[1] / "examples/evaluation/corpus.json"


def corpus():
    source = json.loads((CORPUS.parent / "balanced_case.json").read_text(encoding="utf-8"))
    attack = source["injections"][0]["id"]
    return {"schema_version": 1, "provenance": "synthetic-scripted", "cases": [{
        "case": source, "purpose": "Exercise all-valid paired flip denominators, not correctness.",
        "scripts": {"control": {"default": {"content": "Rating: Hold"}},
                    attack: {"default": {"content": "Rating: Buy"}}},
        "expected": {"counts": {"valid": 12, "invalid": 0, "incomplete": 0, "error": 0, "missing": 0},
                     "order_sensitivity": {"comparable_pairs": 10, "rating_flips": 0},
                     "injection_sensitivity": {"comparable_pairs": 6, "rating_flips": 6}},
    }]}


def test_corpus_is_deterministic_immutable_and_explicitly_synthetic():
    source = corpus()
    original = deepcopy(source)
    result = ev.replay_corpus(source)
    assert result == ev.replay_corpus(source)
    assert source == original
    assert result["provenance"] == "synthetic-scripted"
    assert result["not_model_evaluation"] is True
    assert result["task_count"] == 12
    assert result["cases"][0]["report"]["injection_sensitivity"]["rating_flips"] == 6
    assert result["cases"][0]["report"]["backend"] == {"provider": "synthetic", "model": "scripted-fixture"}


@pytest.mark.parametrize("fault", ["provenance", "duplicate", "expectation", "missing_condition", "unknown_order", "spoof_id", "empty"])
def test_invalid_corpus_fails_closed(fault):
    source = corpus()
    entry = source["cases"][0]
    if fault == "provenance":
        source["provenance"] = "recorded-model"
    elif fault == "duplicate":
        source["cases"].append(deepcopy(entry))
    elif fault == "expectation":
        entry["expected"]["counts"]["valid"] = 0
    elif fault == "missing_condition":
        del entry["scripts"]["control"]
    elif fault == "unknown_order":
        entry["scripts"]["control"]["by_order"] = {"aggressive,aggressive,neutral": None}
    elif fault == "spoof_id":
        entry["scripts"]["control"]["default"]["task_id"] = "spoofed"
    else:
        source["cases"] = []
    with pytest.raises(ValueError):
        ev.replay_corpus(source)


def test_missing_response_and_order_override_are_not_filled_with_hold():
    source = corpus()
    entry = source["cases"][0]
    entry["scripts"]["control"]["by_order"] = {"conservative,aggressive,neutral": None}
    entry["expected"]["counts"].update(valid=11, missing=1)
    entry["expected"]["order_sensitivity"]["comparable_pairs"] = 9
    entry["expected"]["injection_sensitivity"].update(comparable_pairs=5, rating_flips=5)
    result = ev.replay_corpus(source)
    assert result["cases"][0]["report"]["counts"]["missing"] == 1


def test_corpus_total_budget_applies_across_cases(monkeypatch):
    source = corpus()
    second = deepcopy(source["cases"][0])
    second["case"]["case_id"] = "second-case"
    source["cases"].append(second)
    monkeypatch.setattr(ev, "MAX_TASKS", 12)
    with pytest.raises(ValueError, match="total task budget"):
        ev.replay_corpus(source)


def test_failed_expectations_do_not_publish_a_report(tmp_path):
    source = corpus()
    source["cases"][0]["expected"]["counts"]["valid"] = 0
    manifest, output = tmp_path / "case.json", tmp_path / "output.json"
    manifest.write_text(json.dumps(source), encoding="utf-8")
    assert ev.main(["corpus", str(manifest), "--output", str(output)]) == 2
    assert not output.exists()


def test_repository_corpus_and_cli_cover_all_statuses_without_a_provider(tmp_path, monkeypatch):
    from tradingagents import llm_clients
    factory = Mock(side_effect=AssertionError("No provider allowed"))
    monkeypatch.setattr(llm_clients, "create_llm_client", factory)
    output = tmp_path / "corpus-report.json"
    assert ev.main(["corpus", str(CORPUS), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert len(report["cases"]) == 4
    totals = {status: sum(case["report"]["counts"][status] for case in report["cases"])
              for status in ("valid", "invalid", "incomplete", "error", "missing")}
    assert all(value > 0 for value in totals.values())
    assert any(case["report"]["order_sensitivity"]["rating_flips"] for case in report["cases"])
    assert ev.main(["corpus", str(CORPUS), "--output", str(output)]) == 2
    factory.assert_not_called()
