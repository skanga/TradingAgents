from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_options_summary(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """
    Snapshot of options-market positioning for a ticker: put/call volume and
    open-interest ratios, max-pain strike, dominant call/put walls, unusual
    volume spikes (volume > 3x OI), and ATM implied volatility for the
    nearest expiry. Uses the configured options_data vendor.
    """
    return route_to_vendor("get_options_summary", ticker)


@tool
def get_iv_rank(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """
    Implied Volatility Rank for a ticker (current ATM IV's percentile within
    its trailing 52-week range). IVR > 50 = elevated fear/uncertainty;
    IVR < 20 = complacency. Useful for sizing positions and contextualising
    risk. Uses the configured options_data vendor.
    """
    return route_to_vendor("get_iv_rank", ticker)
