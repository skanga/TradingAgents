from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor

@tool
def get_indicators(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[
        str,
        "Technical indicator(s) to retrieve. Pass a SINGLE comma-separated "
        "string with all indicators you want in this analysis "
        "(e.g. 'close_50_sma,close_200_sma,close_10_ema,macd,rsi,boll_ub,"
        "boll_lb,atr,vwma'). DO NOT call this tool once per indicator — "
        "batch them in one call.",
    ],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    look_back_days: Annotated[int, "how many days to look back"] = 30,
) -> str:
    """
    Retrieve one or more technical indicators for a given ticker symbol.
    Pass multiple indicators in a single comma-separated string to batch
    the call — the tool processes all of them in one invocation, which
    is much faster than calling once per indicator (and avoids burning
    through the analyst's tool-round budget).

    Uses the configured technical_indicators vendor.

    Args:
        symbol (str): Ticker symbol, e.g. AAPL, TSLA.
        indicator (str): One or more indicator names, comma-separated.
            Example: ``'rsi,macd,close_50_sma,boll_ub'``.
        curr_date (str): Current trading date, YYYY-mm-dd.
        look_back_days (int): Days to look back (default 30).

    Returns:
        str: Formatted dataframe(s), one block per requested indicator.
    """
    # LLMs sometimes pass multiple indicators as a comma-separated string;
    # split and process each individually.
    indicators = [i.strip().lower() for i in indicator.split(",") if i.strip()]
    results = []
    for ind in indicators:
        try:
            results.append(route_to_vendor("get_indicators", symbol, ind, curr_date, look_back_days))
        except ValueError as e:
            results.append(str(e))
    return "\n\n".join(results)