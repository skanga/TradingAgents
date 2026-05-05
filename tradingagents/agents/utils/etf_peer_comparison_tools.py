from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_etf_peer_comparison(
    ticker: Annotated[str, "Primary ETF ticker (will be the first column)"],
    peers: Annotated[
        str,
        "Comma-separated peer ETF tickers, e.g. 'QQQ,IWM,DIA' for SPY. "
        "Pick 2-4 peers that share the asset class but differ in tilt: "
        "broad-market (SPY → QQQ, IWM, DIA, RSP), tech (QQQ → XLK, VGT, "
        "SPYG), small-cap (IWM → IJR, VB), sector ETFs → other sector "
        "ETFs (XLK → VGT, IGV).",
    ],
) -> str:
    """
    Compare an ETF against 2-6 peer ETFs on a fixed metric set: category,
    AUM, expense ratio, distribution yield, inception, 3-year beta, total
    returns (1M / 3M / YTD / 1Y), 1-year annualized volatility, and 1-year
    max drawdown. Useful for ETF tickers where get_peer_comparison (SEC
    fundamentals) doesn't apply. Returns a Markdown table or a bracketed
    unavailable string when yfinance has no data.
    """
    return route_to_vendor("get_etf_peer_comparison", ticker, peers)
