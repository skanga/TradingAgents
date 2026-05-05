from langchain_core.tools import tool
from typing import Annotated

from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_peer_comparison(
    ticker: Annotated[str, "Primary ticker symbol (will be the first column)"],
    peers: Annotated[
        str,
        "Comma-separated peer ticker list, e.g. 'MSFT,GOOGL,AMZN'. Pick 2-4 "
        "sector peers — too many makes the table unreadable.",
    ],
    metrics: Annotated[
        str,
        "Comma-separated metric keys (snake_case). Empty string → defaults to "
        "revenue, net_income, gross_profit, operating_income. Other supported "
        "keys: total_assets, total_liabilities, stockholders_equity, "
        "eps_diluted, eps_basic, long_term_debt, cash_and_equivalents.",
    ] = "",
    year: Annotated[
        int,
        "Fiscal year to compare (e.g. 2024). 0 → defaults to last calendar year.",
    ] = 0,
) -> str:
    """
    Compare a primary ticker against sector peers on selected fundamentals
    metrics for one fiscal year. Returns a Markdown table with tickers as
    columns and metrics as rows. Tickers Lambda has no data for are rendered
    as em-dashes with a footer note (no silent omission).

    Routes through the configured ``peer_comparison_data`` vendor (only
    Lambda Finance today).
    """
    return route_to_vendor("get_peer_comparison", ticker, peers, metrics, year)
