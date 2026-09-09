"""Offline contracts for bounded, deconcentrated social evidence."""
import json
from unittest.mock import Mock

import pytest

from tradingagents.dataflows import reddit, stocktwits
from tradingagents.agents.analysts.sentiment_analyst import _build_system_message


class Response:
    def __init__(self, payload):
        self.body = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self, size=-1):
        return self.body if size < 0 else self.body[:size]


def message(i, label=None, author=None, body=None):
    return {"id": i, "created_at": "2024-01-10T12:00:00Z",
            "body": body or f"Opinion number {i}", "user": {"username": author or f"user{i}"},
            "entities": {"sentiment": {"basic": label}}}


def fetch_messages(monkeypatch, messages, **kwargs):
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: Response({"messages": messages}))
    return stocktwits.fetch_stocktwits_messages("AAPL", **kwargs)


def test_stocktwits_percentages_use_labeled_denominator(monkeypatch):
    text = fetch_messages(monkeypatch, [message(1, "Bullish"), message(2, "Bearish"), message(3)])
    assert "Bullish: 1 (50%)" in text
    assert "Bearish: 1 (50%)" in text
    assert "Label coverage: 2/3" in text
    assert "labeled posts only" in text


def test_no_labels_is_unavailable_not_zero_or_neutral(monkeypatch):
    text = fetch_messages(monkeypatch, [message(1), message(2)])
    assert "Bullish/Bearish ratio: unavailable" in text
    assert "Label coverage: 0/2" in text
    assert "(0%)" not in text


def test_stocktwits_deduplication_precedes_ratios_and_author_cap(monkeypatch):
    rows = [message(1, "Bullish", "spam", "BUY now"),
            message(2, "Bullish", "other", " buy   NOW ")]
    rows += [message(i, "Bullish", "spam") for i in range(3, 9)]
    rows += [message(10, "Bearish", "reader")]
    text = fetch_messages(monkeypatch, rows)
    assert "Total: 4" in text
    assert "Bullish: 3 (75%)" in text
    assert "duplicates removed: 1" in text
    assert "author-cap exclusions: 4" in text
    assert "largest known-author share" in text


def test_empty_copy_does_not_hide_valid_post_with_same_id(monkeypatch):
    text = fetch_messages(monkeypatch, [{**message(1), "body": ""}, message(1, "Bearish")])
    assert "Total: 1" in text
    assert "empty text: 1" in text
    assert "Bearish: 1 (100%)" in text


def test_duplicate_ids_and_window_filter_order(monkeypatch):
    future = {**message(1, "Bullish"), "created_at": "2025-01-10T12:00:00Z"}
    text = fetch_messages(monkeypatch, [future, message(1, "Bearish"), message(1, "Bearish", body="edited")],
                          start_date="2024-01-01", end_date="2024-01-15")
    assert "Total: 1" in text
    assert "Bearish: 1 (100%)" in text


def test_unknown_authors_are_not_one_author_or_claimed_independent(monkeypatch):
    rows = [{**message(i), "user": {}} for i in range(6)]
    text = fetch_messages(monkeypatch, rows)
    assert "Total: 6" in text
    assert "missing author: 6" in text
    assert "independence" in text


@pytest.mark.parametrize("payload", [{"messages": None}, {"messages": "bad"}, {}])
def test_invalid_stocktwits_feed_is_unavailable(monkeypatch, payload):
    monkeypatch.setattr(stocktwits, "urlopen", lambda *a, **k: Response(payload))
    assert "unavailable" in stocktwits.fetch_stocktwits_messages("AAPL")


@pytest.mark.parametrize("field,value", [("user", "bad"), ("entities", "bad"), ("body", ["bad"])])
def test_malformed_stocktwits_post_fields_are_unavailable(monkeypatch, field, value):
    text = fetch_messages(monkeypatch, [{**message(1), field: value}])
    assert "unavailable" in text


@pytest.mark.parametrize("limit", [0, -1, True])
def test_invalid_limits_rejected_before_network(monkeypatch, limit):
    fetch = Mock(side_effect=AssertionError("must not fetch"))
    monkeypatch.setattr(stocktwits, "urlopen", fetch)
    monkeypatch.setattr(reddit, "_fetch_subreddit", fetch)
    with pytest.raises(ValueError):
        stocktwits.fetch_stocktwits_messages("AAPL", limit=limit)
    with pytest.raises(ValueError):
        reddit.fetch_reddit_posts("AAPL", limit_per_sub=limit)
    fetch.assert_not_called()


def test_stocktwits_output_limit_and_stable_user_id(monkeypatch):
    rows = [{**message(i), "user": {"id": 123, "username": f"alias{i}"}} for i in range(10)]
    text = fetch_messages(monkeypatch, rows)
    assert "Total: 3" in text
    text = fetch_messages(monkeypatch, [message(i) for i in range(80)], limit=10000)
    assert "Total: 30" in text
    assert "limit exclusions: 50" in text


def post(i, author="reader", title=None):
    return {"id": str(i), "author": author, "title": title or f"Discussion {i}",
            "selftext": "", "source": "rss", "created_utc": 1704888000}


def test_reddit_deduplicates_and_caps_authors_across_subreddits(monkeypatch):
    feeds = [[post(1, "spam"), post(2, "spam"), post(3, "spam")],
             [post(1, "spam"), post(4, "spam"), post(5, "reader")]]
    monkeypatch.setattr(reddit, "_fetch_subreddit", Mock(side_effect=feeds))
    text = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks", "investing"), inter_request_delay=0)
    assert text.count("Discussion 1") == 1
    assert "Discussion 4" not in text
    assert "Discussion 5" in text
    assert "duplicates removed: 1" in text
    assert "author-cap exclusions: 1" in text
    assert "scores/comments unavailable" in text
    assert "not a representative" in text


def test_reddit_default_sample_larger_but_caller_limit_bounded(monkeypatch):
    fetch = Mock(return_value=[])
    monkeypatch.setattr(reddit, "_fetch_subreddit", fetch)
    reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",), inter_request_delay=0)
    assert fetch.call_args.args[2] == 15
    reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",), limit_per_sub=10000, inter_request_delay=0)
    assert fetch.call_args.args[2] == 100


def test_reddit_normalized_crosspost_and_repeated_subreddit(monkeypatch):
    fetch = Mock(side_effect=[[post(1, "one", "Company RESULTS")],
                              [post(2, "two", " company   results "), post(3, "three")]])
    monkeypatch.setattr(reddit, "_fetch_subreddit", fetch)
    text = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks", "stocks", "investing"), inter_request_delay=0)
    assert fetch.call_count == 2
    assert "retained: 2" in text
    assert "duplicates removed: 1" in text


def test_rss_preserves_author_and_post_identity(monkeypatch):
    atom = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>t3_abc</id><title>Results</title><author><name>/u/Alice</name></author><link href="https://reddit.com/comments/abc"/></entry></feed>'
    response = Response({})
    response.body = atom
    monkeypatch.setattr(reddit, "urlopen", lambda *a, **k: response)
    result = reddit._fetch_subreddit_rss("AAPL", "stocks", 15, 1)
    assert result[0].get("id") == "t3_abc"
    assert result[0].get("author") == "Alice"
    assert result[0].get("permalink") == "https://reddit.com/comments/abc"


@pytest.mark.parametrize("timestamp", ["bad", float("nan"), float("inf")])
def test_reddit_invalid_dates_are_unknown_live_and_excluded_historically(monkeypatch, timestamp):
    monkeypatch.setattr(reddit, "_fetch_subreddit", lambda *a, **k: [{**post(1), "created_utc": timestamp}])
    live = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",), inter_request_delay=0)
    assert "Discussion 1" in live
    historical = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",), inter_request_delay=0,
                                            start_date="2024-01-01", end_date="2024-01-15")
    assert "Discussion 1" not in historical


def test_deleted_reddit_authors_do_not_form_a_single_account(monkeypatch):
    rows = [post(i, "[deleted]") for i in range(6)]
    monkeypatch.setattr(reddit, "_fetch_subreddit", lambda *a, **k: rows)
    text = reddit.fetch_reddit_posts("AAPL", subreddits=("stocks",), inter_request_delay=0)
    assert "retained: 6" in text
    assert "missing author: 6" in text


def test_social_checkpoint_policy_is_part_of_run_identity():
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    graph = object.__new__(TradingAgentsGraph)
    graph.config = {"max_debate_rounds": 1, "max_risk_discuss_rounds": 1}
    graph.selected_analysts = ("sentiment",)
    assert "social_evidence_policy=2" in graph._run_signature("stock", "2024-01-15")


def test_social_prompt_uses_quality_not_fixed_ratio_trading_rules():
    text = _build_system_message(ticker="AAPL", start_date="2024-01-01", end_date="2024-01-15",
                                 news_block="none", stocktwits_block="none", reddit_block="none")
    assert "labeled posts only" in text
    assert "author concentration" in text
    assert "do not invent engagement" in text.lower()
    assert "70/30" not in text
    assert "3-upvote post is noise" not in text
