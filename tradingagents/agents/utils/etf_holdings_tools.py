from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_etf_holdings(
    ticker: Annotated[str, "ETF or fund ticker symbol (e.g. SPY, QQQ, IWM)"],
) -> str:
    """
    Retrieve sector-weight breakdown, top-10 holdings, asset-class mix, and
    concentration metric for an ETF. Use this in place of get_peer_comparison
    when the ticker is a fund — peer comparison via SEC filings doesn't apply
    to a basket. Returns a Markdown report or a bracketed unavailable string
    when the ticker isn't a fund / yfinance has no funds_data.
    """
    return route_to_vendor("get_etf_holdings", ticker)
