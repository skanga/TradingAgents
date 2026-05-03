from langchain_core.tools import tool


from tradingagents.dataflows.interface import route_to_vendor


@tool
def get_macro_environment() -> str:
    """
    Current macroeconomic conditions: 2Y/10Y Treasury yields, 10Y-2Y curve
    spread, high-yield credit spread, and broad USD trend. Returns a
    FAVORABLE / NEUTRAL / UNFAVORABLE macro-backdrop rating with point-by-point
    reasoning. Uses the configured macro_data vendor (FRED).
    """
    return route_to_vendor("get_macro_environment")
