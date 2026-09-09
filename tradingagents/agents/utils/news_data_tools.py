from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.tool_dates import (
    DEFAULT_RUN_STATE,
    RunState,
    bounded_date,
    bounded_range,
    live_only_notice,
)
from tradingagents.dataflows.interface import route_to_vendor
from tradingagents.dataflows.news_evidence import shared_news_request


@tool
def get_news(
    ticker: Annotated[str, "Ticker symbol"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
    state: RunState = DEFAULT_RUN_STATE,
) -> str:
    """
    Retrieve news data for a given ticker symbol.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol
        start_date (str): Start date in yyyy-mm-dd format
        end_date (str): End date in yyyy-mm-dd format
    Returns:
        str: A formatted string containing news data
    """
    start_date, end_date = bounded_range(start_date, end_date, state)
    args = (ticker, start_date, end_date)
    return shared_news_request("get_news", args, lambda: route_to_vendor("get_news", *args))

@tool
def get_global_news(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int | None, "Days to look back; omit to use the configured default"] = None,
    limit: Annotated[int | None, "Max articles to return; omit to use the configured default"] = None,
    state: RunState = DEFAULT_RUN_STATE,
) -> str:
    """
    Retrieve global news data.
    Uses the configured news_data vendor. Defaults for look_back_days and
    limit come from DEFAULT_CONFIG (global_news_lookback_days,
    global_news_article_limit); pass explicit values to override.

    Args:
        curr_date (str): Current date in yyyy-mm-dd format
        look_back_days (int): Number of days to look back; omit to inherit config
        limit (int): Maximum number of articles to return; omit to inherit config

    Returns:
        str: A formatted string containing global news data
    """
    args = (bounded_date(curr_date, state), look_back_days, limit)
    return shared_news_request("get_global_news", args, lambda: route_to_vendor("get_global_news", *args))

@tool
def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol"],
    state: RunState = DEFAULT_RUN_STATE,
) -> str:
    """
    Retrieve insider transaction information about a company.
    Uses the configured news_data vendor.
    Args:
        ticker (str): Ticker symbol of the company
    Returns:
        str: A report of insider transaction data
    """
    notice = live_only_notice(state, "Insider transactions")
    return notice if notice else route_to_vendor("get_insider_transactions", ticker)
