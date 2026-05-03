import os
from unittest.mock import MagicMock

import pytest

from tradingagents.dataflows import earnings_transcript
from tradingagents.dataflows.config import set_config


# --- Hedge-word + deflection counters --------------------------------------


def test_hedge_pattern_catches_phrases_and_words():
    text = (
        "We remain cautious on near-term demand. The macro environment is "
        "challenging and we expect headwinds in Q3. It remains to be seen "
        "whether pricing pressure continues."
    )
    rate = earnings_transcript._per_1k(text, earnings_transcript._HEDGE_RE)
    # 4 hits in ~30 words → very high per-1k figure
    assert rate > 100


def test_deflection_pattern_catches_non_answers():
    qa = (
        "Analyst: What's your gross margin guidance? "
        "CFO: We don't break that out at the segment level. "
        "Analyst: How about by region? "
        "CFO: We're not providing that level of detail today, but we'll share "
        "more color next quarter."
    )
    rate = earnings_transcript._per_1k(qa, earnings_transcript._DEFLECTION_RE)
    assert rate > 0


def test_per_1k_returns_zero_for_empty_text():
    assert earnings_transcript._per_1k("", earnings_transcript._HEDGE_RE) == 0.0


def test_word_count_excludes_punctuation():
    assert earnings_transcript._word_count("We're up 12% Y/Y, driven by demand.") >= 6


# --- Section splitter ------------------------------------------------------


def test_split_sections_on_questions_and_answers_header():
    text = "Prepared opening...\n\nblah blah\n\nQuestions and Answers\n\nAnalyst: ...\nCFO: ..."
    prepared, qa = earnings_transcript._split_sections(text)
    assert "Prepared opening" in prepared
    assert "Analyst:" in qa
    assert "Prepared opening" not in qa


def test_split_sections_falls_back_to_65_35_when_no_marker():
    text = "Some opening narrative. " * 50
    prepared, qa = earnings_transcript._split_sections(text)
    assert prepared and qa
    # Both halves should contain the repeated phrase
    assert "Some opening" in prepared and "Some opening" in qa


# --- LLM scoring -----------------------------------------------------------


def test_parse_json_object_handles_clean_json():
    raw = '{"prepared_sentiment": "positive", "qa_sentiment": "neutral", "prepared_reason": "x", "qa_reason": "y"}'
    out = earnings_transcript._parse_json_object(raw)
    assert out and out["prepared_sentiment"] == "positive"


def test_parse_json_object_extracts_json_from_prose():
    raw = "Sure, here is the JSON:\n```json\n{\"prepared_sentiment\":\"negative\"}\n```\nLet me know."
    out = earnings_transcript._parse_json_object(raw)
    assert out and out["prepared_sentiment"] == "negative"


def test_parse_json_object_returns_none_on_garbage():
    assert earnings_transcript._parse_json_object("not json at all") is None
    assert earnings_transcript._parse_json_object("") is None


def test_normalise_sentiment_buckets():
    assert earnings_transcript._normalise_sentiment("Positive") == "positive"
    assert earnings_transcript._normalise_sentiment("BULLISH") == "positive"
    assert earnings_transcript._normalise_sentiment("cautious") == "negative"
    assert earnings_transcript._normalise_sentiment("anything else") == "neutral"


def test_score_sentiment_with_llm_uses_response_when_valid(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = (
        '{"prepared_sentiment":"positive",'
        '"prepared_reason":"strong demand commentary",'
        '"qa_sentiment":"neutral",'
        '"qa_reason":"balanced answers"}'
    )
    monkeypatch.setattr(earnings_transcript, "_build_quick_llm", lambda: fake_llm)

    out = earnings_transcript._score_sentiment_with_llm("prepared text", "qa text")
    assert out["prepared_sentiment"] == "positive"
    assert out["qa_sentiment"] == "neutral"
    assert "strong demand" in out["prepared_reason"]
    fake_llm.invoke.assert_called_once()


def test_score_sentiment_with_llm_falls_back_when_llm_raises(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.invoke.side_effect = RuntimeError("provider down")
    monkeypatch.setattr(earnings_transcript, "_build_quick_llm", lambda: fake_llm)
    out = earnings_transcript._score_sentiment_with_llm("p", "qa")
    assert out["prepared_sentiment"] == "neutral"
    assert "unavailable" in out["prepared_reason"]


def test_score_sentiment_with_llm_falls_back_on_unparseable_response(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = "no json here at all"
    monkeypatch.setattr(earnings_transcript, "_build_quick_llm", lambda: fake_llm)
    out = earnings_transcript._score_sentiment_with_llm("p", "qa")
    assert out["prepared_sentiment"] == "neutral"


# --- Reporting --------------------------------------------------------------


def test_format_report_emits_table_and_risk_flags():
    sentiment = {
        "prepared_sentiment": "positive",
        "prepared_reason": "Strong demand language",
        "qa_sentiment": "negative",
        "qa_reason": "Repeated deflection on margin questions",
    }
    report = earnings_transcript._format_report(
        ticker="AAPL",
        title="Apple (AAPL) Q1 2026 Earnings Call Transcript",
        transcript_url="https://www.fool.com/earnings/call-transcripts/2026/02/01/apple-aapl-q1-2026.aspx",
        prepared_words=4200,
        qa_words=5500,
        prepared_hedge=2.0,
        qa_hedge=10.5,        # > 8 → triggers prepared/Q&A divergence flag
        qa_deflection=4.5,    # > 4 → triggers deflection flag
        sentiment=sentiment,
    )
    assert report.startswith("## Earnings Call Sentiment for AAPL")
    assert "Prepared Remarks" in report and "Q&A" in report
    assert "POSITIVE" in report and "NEGATIVE" in report
    assert "Risk flags" in report
    # Both risk flags should fire
    assert "guarded under questioning" in report
    assert "deflection rate is elevated" in report
    assert "diverging" in report  # Q&A vs prepared divergence flag


def test_format_report_no_risk_flags_when_signals_clean():
    sentiment = {
        "prepared_sentiment": "positive",
        "prepared_reason": "Clear acceleration commentary",
        "qa_sentiment": "positive",
        "qa_reason": "Direct answers throughout",
    }
    report = earnings_transcript._format_report(
        ticker="X", title="X Q1 Call", transcript_url="https://example.com",
        prepared_words=3000, qa_words=4000,
        prepared_hedge=1.0, qa_hedge=2.0, qa_deflection=1.0,
        sentiment=sentiment,
    )
    assert "Risk flags" not in report


# --- Top-level entry point -------------------------------------------------


def test_get_earnings_transcript_sentiment_handles_no_listing(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(earnings_transcript, "_find_latest_transcript_url", lambda _t: None)
    out = earnings_transcript.get_earnings_transcript_sentiment("AAPL")
    assert out.startswith("[Earnings transcript unavailable: no transcript found")


def test_get_earnings_transcript_sentiment_short_page_falls_back(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    monkeypatch.setattr(
        earnings_transcript, "_find_latest_transcript_url",
        lambda _t: "https://example.com/transcript",
    )
    monkeypatch.setattr(earnings_transcript, "_fetch_transcript_text", lambda _u: "tiny")
    out = earnings_transcript.get_earnings_transcript_sentiment("AAPL")
    assert out.startswith("[Earnings transcript unavailable")


def test_get_earnings_transcript_sentiment_full_pipeline_with_mocked_io(monkeypatch, tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    transcript = (
        "Apple (AAPL) Q1 2026 Earnings Call Transcript\n\n"
        "Operator: Welcome to the call.\n\n"
        "CEO: We had a strong quarter with broad-based growth across all segments. "
        "Revenue accelerated to record levels. " * 80
        + "\n\nQuestions and Answers\n\n"
        "Analyst: Can you walk through gross margin trends?\n"
        "CFO: We don't break that out at the segment level. "
        "We're not providing additional color today. "
        "We'll share more detail next quarter. " * 50
    )
    monkeypatch.setattr(
        earnings_transcript, "_find_latest_transcript_url",
        lambda _t: "https://www.fool.com/earnings/call-transcripts/2026/02/01/apple-aapl.aspx",
    )
    monkeypatch.setattr(earnings_transcript, "_fetch_transcript_text", lambda _u: transcript)

    fake_llm = MagicMock()
    fake_llm.invoke.return_value.content = (
        '{"prepared_sentiment":"positive","prepared_reason":"strong cited",'
        '"qa_sentiment":"negative","qa_reason":"deflective answers cited"}'
    )
    monkeypatch.setattr(earnings_transcript, "_build_quick_llm", lambda: fake_llm)

    out = earnings_transcript.get_earnings_transcript_sentiment("AAPL")
    assert out.startswith("## Earnings Call Sentiment for AAPL")
    assert "POSITIVE" in out and "NEGATIVE" in out
    assert "deflection" in out


# --- Integration -----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"),
    reason="needs an LLM key to score sentiment",
)
def test_get_earnings_transcript_sentiment_live_aapl(tmp_path):
    set_config({"data_cache_dir": str(tmp_path)})
    out = earnings_transcript.get_earnings_transcript_sentiment("AAPL")
    assert isinstance(out, str) and out
    # Either real report or graceful no-data — never an unhandled crash
    assert out.startswith("##") or out.startswith("[Earnings transcript unavailable")
