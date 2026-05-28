import pytest

from tradingagents.graph.analyst_execution import (
    AnalystWallTimeTracker,
    build_analyst_execution_plan,
    get_initial_analyst_node,
    sync_analyst_tracker_from_chunk,
)


def test_build_plan_preserves_selected_order():
    plan = build_analyst_execution_plan(["news", "market"], concurrency_limit=2)

    assert [spec.key for spec in plan.specs] == ["news", "market"]
    assert plan.concurrency_limit == 2
    assert plan.specs[0].agent_node == "News Analyst"
    assert plan.specs[0].tool_node == "tools_news"
    assert plan.specs[0].clear_node == "Msg Clear News"


def test_rejects_unknown_analyst_keys():
    with pytest.raises(ValueError, match="unknown analyst key"):
        build_analyst_execution_plan(["market", "macro"])


def test_requires_positive_concurrency_limit():
    with pytest.raises(ValueError, match="concurrency limit"):
        build_analyst_execution_plan(["market"], concurrency_limit=0)


def test_get_initial_analyst_node_uses_plan_metadata():
    plan = build_analyst_execution_plan(["fundamentals", "news"])

    assert get_initial_analyst_node(plan) == "Fundamentals Analyst"


def test_social_key_displays_as_sentiment_analyst():
    plan = build_analyst_execution_plan(["social"])
    spec = plan.specs[0]

    assert spec.key == "social"
    assert spec.agent_node == "Sentiment Analyst"
    assert spec.clear_node == "Msg Clear Sentiment"
    assert spec.report_key == "sentiment_report"


def test_syncs_wall_time_from_sequential_chunks():
    plan = build_analyst_execution_plan(["market", "news"])
    tracker = AnalystWallTimeTracker(plan)

    sync_analyst_tracker_from_chunk(tracker, {}, now=10.0)
    assert tracker.get_wall_times() == {}

    sync_analyst_tracker_from_chunk(tracker, {"market_report": "done"}, now=13.0)
    assert tracker.get_wall_times() == {"market": 3.0}

    sync_analyst_tracker_from_chunk(
        tracker,
        {"market_report": "done", "news_report": "done"},
        now=18.0,
    )
    assert tracker.get_wall_times() == {"market": 3.0, "news": 5.0}
