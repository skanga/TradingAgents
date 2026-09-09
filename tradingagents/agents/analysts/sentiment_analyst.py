"""Sentiment analyst — multi-source sentiment analysis for a target ticker.

Previously named ``social_media_analyst``. Renamed and redesigned because
the old version had a prompt that demanded social-media analysis but the
only tool available was Yahoo Finance news — which led LLMs to fabricate
Reddit/X/StockTwits content under prompt pressure (verified live).

The redesigned agent pre-fetches three complementary data sources before
the LLM is invoked and injects them into the prompt as structured blocks:

  1. News headlines     — configured vendor (publisher provenance attached)
  2. StockTwits messages — retail-trader posts indexed by cashtag, with
                           user-labeled Bullish/Bearish sentiment tags
  3. Reddit posts        — r/wallstreetbets, r/stocks, r/investing

The agent does not use tool-calling; the data is in the prompt from
turn 0. Output uses the structured-output pattern (json_schema for
OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic), falling
back to free-text generation for providers that lack native support, so
the sentiment header (band + score + confidence) is deterministic across
runs and providers instead of free-form per-model prose.

See: https://github.com/TauricResearch/TradingAgents/issues/557
See: https://github.com/TauricResearch/TradingAgents/issues/796
"""

from datetime import datetime, timedelta

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.schemas import SentimentReport, render_sentiment_report
from tradingagents.agents.utils.agent_utils import (
    get_instrument_context_from_state,
    get_language_instruction,
    get_news,
)
from tradingagents.agents.utils.prompt_boundaries import (
    UNTRUSTED_CONTENT_INSTRUCTION,
    evidence_block,
    evidence_history,
)
from tradingagents.agents.utils.structured import (
    NO_EXTERNAL_TOOLS,
    bind_structured,
    invoke_structured_or_freetext,
)
from tradingagents.agents.utils.tool_dates import bounded_date
from tradingagents.dataflows.news_evidence import SHARED_NEWS_INSTRUCTION
from tradingagents.dataflows.reddit import fetch_reddit_posts
from tradingagents.dataflows.stocktwits import fetch_stocktwits_messages


def _seven_days_back(trade_date: str) -> str:
    return (datetime.strptime(trade_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")


def create_sentiment_analyst(llm):
    """Create a sentiment analyst node for the trading graph.

    Pre-fetches news + StockTwits + Reddit data, injects them into the
    prompt as structured blocks, and produces a deterministic sentiment
    report via structured output (with a free-text fallback for providers
    that do not support it).
    """
    structured_llm = bind_structured(llm, SentimentReport, "Sentiment Analyst")

    def sentiment_analyst_node(state):
        ticker = state["company_of_interest"]
        end_date = bounded_date(state["trade_date"], state)
        start_date = _seven_days_back(end_date)
        instrument_context = get_instrument_context_from_state(state)

        # Pre-fetch all three sources. Each fetcher degrades gracefully and
        # returns a string (no exceptions surface from here), so the LLM
        # always sees something — either real data or a clear placeholder.
        news_block = get_news.func(ticker, start_date, end_date)
        # Pass the analysis window so a historical run trims social posts to it
        # instead of leaking today's chatter into a backtest (#1220).
        stocktwits_block = fetch_stocktwits_messages(
            ticker, limit=30, start_date=start_date, end_date=end_date
        )
        reddit_block = fetch_reddit_posts(ticker, start_date=start_date, end_date=end_date)

        system_message = _build_system_message(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            news_block=news_block,
            stocktwits_block=stocktwits_block,
            reddit_block=reddit_block,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Produce an evidence report, not a transaction proposal. Report supportive, adverse and neutral findings; do not force a directional conclusion."
                    # No tool-calling here: the data is pre-fetched into the
                    # prompt, so tool-range wording would only invite a
                    # hallucinated tool call (#1130).
                    " Today's date is {current_date}; treat it as 'now' for all analysis."
                    " " + NO_EXTERNAL_TOOLS +
                    "\n{system_message}",
                ),
                ("human", "{source_evidence}"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(current_date=end_date)
        prompt = prompt.partial(source_evidence="\n\n".join([
            "Instrument context:\n" + evidence_block(instrument_context),
            "News:\n" + evidence_block(news_block),
            "StockTwits:\n" + evidence_block(stocktwits_block),
            "Reddit:\n" + evidence_block(reddit_block),
        ]))

        # Format the template into a concrete message list so the structured
        # and free-text paths receive the same input. No bind_tools — the
        # data is already in the prompt.
        formatted_messages = prompt.format_messages(messages=evidence_history(state["messages"]))

        report_text = invoke_structured_or_freetext(
            structured_llm,
            llm,
            formatted_messages,
            render_sentiment_report,
            "Sentiment Analyst",
        )

        return {
            "messages": [AIMessage(content=report_text)],
            "sentiment_report": report_text,
        }

    return sentiment_analyst_node


def _build_system_message(
    *,
    ticker: str,
    start_date: str,
    end_date: str,
    news_block: str,
    stocktwits_block: str,
    reddit_block: str,
) -> str:
    """Assemble trusted instructions only; source blocks are separate human evidence.

    Source arguments remain accepted for compatibility with prompt callers.
    """
    return f"""You are a financial market sentiment analyst. Your task is to produce a comprehensive sentiment report for {ticker} covering the period from {start_date} to {end_date}, drawing on three complementary data sources that have already been collected for you.

{UNTRUSTED_CONTENT_INSTRUCTION}

## Data sources (pre-fetched, in separate evidence messages)

### News headlines — configured vendor, past 7 days
Read publisher and retrieval-vendor provenance from the supplied evidence.
{SHARED_NEWS_INSTRUCTION}

### StockTwits messages — retail-trader social platform indexed by cashtag
Fast-moving signal. Each message carries a user-labeled sentiment tag (Bullish / Bearish / no-label) plus the message body.

### Reddit posts — r/wallstreetbets, r/stocks, r/investing (past 7 days)
Community discussion from a bounded recent search sample. RSS does not supply upvote scores or comment counts; use engagement only when explicitly present.

## How to analyze this data (best practices)

1. **StockTwits percentages describe labeled posts only**, after duplicate and author controls. Report the labeled denominator and label coverage alongside the split. Unlabeled posts are not neutral or bearish votes; no labels means the ratio is unavailable. Self-selected labels from a bounded recent feed are not a population poll or a calibrated trading signal. Do not map fixed percentage thresholds to recommendations.

2. **Look for cross-source divergences.** If news framing is bearish but StockTwits is overwhelmingly bullish, that mismatch is itself a signal — it can mean retail is leaning into a thesis the news flow hasn't caught up to (or vice versa, that retail is chasing while institutions are cautious).

3. **Do not invent engagement.** Reddit RSS scores/comments are unavailable, not zero. When supplied, engagement measures attention, not truth, author independence, or investment quality. Do not dismiss low-engagement posts automatically. Read body excerpts for context.

4. **Distinguish reported events from opinion.** Verify event claims against their provenance. Social predictions and enthusiasm are opinions, not established facts or forecasts of returns.

5. **Check sample quality before interpreting repetition.** Cite retained counts, duplicate exclusions, author concentration, missing authors, and label coverage. Repeated or cross-posted text is not independent corroboration. The per-author cap reduces concentration but does not detect bots, coordinated accounts, or paraphrased campaigns; selection order can affect the result.

6. **Be honest about data limits.** Small or concentrated samples, low label coverage, missing engagement/author metadata, and unavailable sources weaken confidence: flag them in the `confidence` field and narrative. Account age, authenticity and bot-quality metrics have not been established. Missing matches in recent feeds do not establish historical silence. Sample counts describe the retrieved subset, not overall discussion volume.

7. **Identify catalysts and risks** that emerge across sources — news of upcoming earnings, product launches, competitive threats, macro headlines, etc.

8. **Past sentiment is not predictive.** Frame your conclusions as signal for the trader to weigh alongside fundamentals and technicals, not as a price call.

## Fictional example — illustration only, not evidence

ExampleCo is fictional: equally credible positive reports of customer enthusiasm and negative reports of service dissatisfaction, with no supported net direction, can justify Neutral with an explanation of the balance. If those sources instead support materially divergent conclusions, Mixed may be appropriate. Missing reports alone establish neither balance nor measured neutrality. Do not cite ExampleCo, copy this example into findings, or treat it as evidence about the analyzed instrument; assess only the supplied source data. This example implies no transaction or rating.

## Output fields

Fill the following fields:

- **overall_band**: Exactly one of Bullish / Mildly Bullish / Neutral / Mixed / Mildly Bearish / Bearish. Use Mixed when sources point in clearly different directions. Use Neutral when substantive evidence supports no net directional sentiment, including balanced or non-directional findings. Missing data is not evidence of neutrality: if no supported direction can be assessed, use Neutral as an unassessed placeholder with low confidence and explicitly state insufficient evidence.
- **overall_score**: A number from 0 (maximally bearish) to 10 (maximally bullish); 5 is neutral. Keep it consistent with overall_band.
- **confidence**: low / medium / high, based on data quality and sample size.
- **narrative**: Full source-by-source breakdown, divergences, dominant narrative themes, catalysts and risks, and a markdown summary table of key sentiment signals (direction, source, supporting evidence).

{get_language_instruction()}"""


# ---------------------------------------------------------------------------
# Backwards-compatibility shim
# ---------------------------------------------------------------------------
def create_social_media_analyst(llm):
    """Deprecated alias for :func:`create_sentiment_analyst`.

    Kept so existing code that imports ``create_social_media_analyst``
    continues to work.

    .. deprecated::
        Import :func:`create_sentiment_analyst` directly instead.
    """
    import warnings
    warnings.warn(
        "create_social_media_analyst is deprecated and will be removed in a "
        "future version. Use create_sentiment_analyst instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return create_sentiment_analyst(llm)
