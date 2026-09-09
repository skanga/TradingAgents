"""Repeated identical prompts expose background variation, not causal bias."""
import json
from copy import deepcopy
from pathlib import Path

import pytest

from tradingagents import evaluation as ev


def plan(repeats=3):
    source = json.loads((Path(__file__).resolve().parents[1] / "examples/evaluation/balanced_case.json").read_text(encoding="utf-8"))
    return ev.build_plan(source, repeats=repeats, schedule_seed=17)


def responses(frozen):
    return [{"task_id": t["task_id"], "prompt_sha256": t["prompt_sha256"], "content": "Rating: Hold"}
            for t in frozen["tasks"]]


def test_repeat_controls_match_same_prompt_not_other_orders_or_conditions():
    frozen = plan()
    records = responses(frozen)
    changed = next(t for t in frozen["tasks"] if t["trial"] == 1)
    next(row for row in records if row["task_id"] == changed["task_id"])["content"] = "Rating: Underweight"
    report = ev.replay(frozen, records)
    metric = report["repeat_sensitivity"]
    assert metric["eligible_pairs"] == metric["comparable_pairs"] == 24
    assert metric["rating_flips"] == 1
    assert metric["flip_rate"] == 1 / 24
    assert "not independent" in report["repeat_limitations"]


def test_repeat_controls_exclude_missing_and_invalid_reference_outputs():
    frozen = plan(repeats=2)
    records = responses(frozen)
    missing = next(t for t in frozen["tasks"] if t["trial"] == 0)
    records = [row for row in records if row["task_id"] != missing["task_id"]]
    other = next(t for t in frozen["tasks"] if t["trial"] == 0 and t["task_id"] != missing["task_id"])
    next(row for row in records if row["task_id"] == other["task_id"])["content"] = "No actionable rating"
    metric = ev.replay(frozen, records)["repeat_sensitivity"]
    assert metric["eligible_pairs"] == 12
    assert metric["comparable_pairs"] == 10
    assert metric["excluded_pairs"] == 2
    assert metric["rating_flips"] == 0
    assert ev.replay(frozen, [])["repeat_sensitivity"]["flip_rate"] is None


def test_rehashed_plan_cannot_claim_changed_prompts_are_identical_repeats():
    frozen = plan(repeats=2)
    changed = frozen["tasks"][0]
    changed["prompt"] += " A different instruction."
    changed["prompt_sha256"] = ev._hash(changed["prompt"])
    frozen["plan_sha256"] = ev._hash({k: v for k, v in frozen.items() if k != "plan_sha256"})
    with pytest.raises(ValueError, match="repeat"):
        ev.replay(frozen, [])


def test_single_trial_artifacts_keep_their_original_report_contract():
    frozen = plan(repeats=1)
    original = deepcopy(frozen)
    report = ev.replay(frozen, responses(frozen))
    assert "repeat_sensitivity" not in report
    assert frozen == original
