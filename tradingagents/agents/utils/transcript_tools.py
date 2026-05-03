from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_earnings_transcript_sentiment(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """
    Retrieve the most recent earnings call transcript for a ticker and score
    its sentiment using the configured quick-thinking LLM. Reports separately
    on Management Prepared Remarks vs Q&A, hedge-word frequency, forward
    guidance tone shift vs prior quarter, and Q&A responsiveness. Uses the
    configured transcript_data vendor.
    """
    return route_to_vendor("get_earnings_transcript_sentiment", ticker)
