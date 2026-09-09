"""Remaining original-plan controls: minor-edit floods and fictional examples."""
import json
from unittest.mock import Mock

import pytest

from tradingagents.agents.analysts.sentiment_analyst import _build_system_message
from tradingagents.dataflows import reddit, stocktwits
from tradingagents.dataflows.social_evidence import SocialSample

BASE = "The company reported strong demand across its core products while management expects steady expansion into additional markets"


def selected(bodies):
    sample = SocialSample()
    posts = [{"id": i, "author": f"author{i}", "text": text} for i, text in enumerate(bodies)]
    kept = sample.select(posts, 100, identity=lambda p: p["id"], author=lambda p: p["author"], text=lambda p: p["text"])
    return kept, sample


@pytest.mark.parametrize("variant", [BASE + " today", BASE.replace("strong", "robust"), BASE + "!!!"])
def test_minor_edit_campaigns_are_deduplicated(variant):
    kept, sample = selected([BASE, variant])
    assert len(kept) == 1
    assert sample.near_duplicates == 1
    assert "near-duplicates removed: 1" in sample.summary()


@pytest.mark.parametrize("left,right", [
    (BASE, BASE.replace("expects", "does not expect")),
    (BASE + " with 5% growth", BASE + " with 15% growth"),
    (BASE + " with +5% growth", BASE + " with -5% growth"),
    (BASE + " in Q1", BASE + " in Q2"),
    (BASE + " for $AAPL", BASE + " for $NVDA"),
    (BASE + " buy", BASE + " sell"),
])
def test_material_changes_are_not_collapsed(left, right):
    kept, _ = selected([left, right])
    assert len(kept) == 2


def test_short_and_oversized_text_is_not_fuzzily_prefix_matched():
    kept, _ = selected(["strong demand now", "strong demand today", BASE * 100 + " alpha", BASE * 100 + " beta"])
    assert len(kept) == 4


def test_near_duplicate_index_is_bounded(monkeypatch):
    import tradingagents.dataflows.social_evidence as social
    monkeypatch.setattr(social, "_MAX_NEAR_CANDIDATES", 2)
    kept, sample = selected([BASE + f" with {i}% growth" for i in range(4)])
    assert len(kept) == 4
    assert len(sample.near_candidates) == 2
    assert sample.near_omissions == 2


def test_near_duplicates_do_not_form_transitive_chains():
    second = BASE.replace("strong", "robust")
    third = second.replace("steady", "lasting")
    kept, _ = selected([BASE, second, third])
    assert [row["text"] for row in kept] == [BASE, third]


def test_stocktwits_minor_edit_flood_changes_retained_not_raw_denominator(monkeypatch):
    suffixes = ["today", "again", "indeed", "clearly", "reportedly", "certainly"]
    bodies = [BASE] + [BASE + " " + word for word in suffixes]
    messages = [{"id": i, "created_at": "2024-01-10T12:00:00Z", "body": text,
                 "user": {"username": f"user{i}"}, "entities": {"sentiment": {"basic": "Bullish"}}}
                for i, text in enumerate(bodies)]
    messages += [{**messages[0], "id": 100, "body": BASE + " sell", "user": {"username": "other"},
                  "entities": {"sentiment": {"basic": "Bearish"}}}]
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return json.dumps({"messages": messages}).encode()
    monkeypatch.setattr(stocktwits, "urlopen", lambda *args, **kwargs: Response())
    result = stocktwits.fetch_stocktwits_messages("AAPL", start_date="2024-01-01", end_date="2024-01-15")
    assert "Bullish: 1 (50%)" in result  # raw feed was 7/8 bullish
    assert "Label coverage: 2/2" in result
    assert "near-duplicates removed: 6" in result


def test_reddit_minor_edit_controls_span_subreddits_after_window_filter(monkeypatch):
    def post(i, title, epoch=1704888000):
        return {"id": str(i), "author": f"author{i}", "title": title, "selftext": "", "created_utc": epoch, "source": "rss"}
    feeds = [[post(0, BASE, 1800000000), post(1, BASE)],
             [post(2, BASE + " today"), post(3, BASE.replace("strong", "robust")), post(4, BASE + " sell")]]
    monkeypatch.setattr(reddit, "_fetch_subreddit", Mock(side_effect=feeds))
    result = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks", "investing"), inter_request_delay=0,
                                       start_date="2024-01-01", end_date="2024-01-15")
    assert "retained: 2" in result
    assert "near-duplicates removed: 2" in result
    assert "scores/comments unavailable" in result


def test_fictional_balanced_example_is_explicitly_not_evidence():
    prompt = _build_system_message(ticker="AAPL", start_date="2024-01-01", end_date="2024-01-15",
                                   news_block="", stocktwits_block="", reddit_block="")
    assert "Fictional example" in prompt
    assert "ExampleCo" in prompt
    assert "positive" in prompt and "negative" in prompt
    assert "not evidence" in prompt
    assert "Do not cite" in prompt
    assert "Nvidia" not in prompt and "NVDA" not in prompt


def test_updated_controls_invalidate_old_checkpoint_identity():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.selected_analysts = ("sentiment",)
    signature = graph._run_signature("stock", "2024-01-15")
    assert "social_evidence_policy=2" in signature
    assert "sentiment_example_policy=1" in signature
