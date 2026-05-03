from langchain_core.tools import tool
from typing import Annotated
from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_congress_trades(
    ticker: Annotated[str, "Ticker symbol"],
) -> str:
    """
    Retrieve recent Congressional STOCK Act stock-trade disclosures for a ticker.
    Includes legislator name, party, transaction type, dollar range, filing
    lag, and committee memberships relevant to the company's sector.
    Uses the configured political_data vendor.
    """
    return route_to_vendor("get_congress_trades", ticker)
