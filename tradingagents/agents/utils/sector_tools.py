from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_sector_relative_strength(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """
    Compute the ticker's sector ETF return vs the S&P 500 (SPY) and the
    ticker's return vs its sector ETF. Surfaces sector-rotation tailwinds
    and headwinds the single-ticker technical view cannot see. Uses the
    configured sector_data vendor.
    """
    return route_to_vendor("get_sector_relative_strength", ticker)


@tool
def get_intermarket_correlations(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """
    63-day rolling correlations between ticker daily returns and gold (GLD),
    oil (USO), crypto (BTC-USD), USD (UUP proxy), and VIX. Highlights any
    correlation outside [-0.5, 0.5] as a meaningful sensitivity. Uses the
    configured sector_data vendor.
    """
    return route_to_vendor("get_intermarket_correlations", ticker)
