from typing import Annotated

# Import from vendor-specific modules
from .y_finance import (
    get_YFin_data_online,
    get_stock_stats_indicators_window,
    get_fundamentals as get_yfinance_fundamentals,
    get_balance_sheet as get_yfinance_balance_sheet,
    get_cashflow as get_yfinance_cashflow,
    get_income_statement as get_yfinance_income_statement,
    get_insider_transactions as get_yfinance_insider_transactions,
)
from .yfinance_news import get_news_yfinance, get_global_news_yfinance
from .alpha_vantage import (
    get_stock as get_alpha_vantage_stock,
    get_indicator as get_alpha_vantage_indicator,
    get_fundamentals as get_alpha_vantage_fundamentals,
    get_balance_sheet as get_alpha_vantage_balance_sheet,
    get_cashflow as get_alpha_vantage_cashflow,
    get_income_statement as get_alpha_vantage_income_statement,
    get_insider_transactions as get_alpha_vantage_insider_transactions,
    get_news as get_alpha_vantage_news,
    get_global_news as get_alpha_vantage_global_news,
)
from .alpha_vantage_common import AlphaVantageRateLimitError

# Enhanced data adapters (scaffolded; see each module for implementation status)
from .sec_insider import get_insider_transactions as get_sec_insider_transactions
from .congress_trades import get_congress_trades as get_finnhub_congress_trades
from .lambda_finance_sec import (
    get_income_statement as get_lambda_finance_income_statement,
    get_balance_sheet as get_lambda_finance_balance_sheet,
)
from .options_flow import (
    get_options_summary as get_yfinance_options_summary,
    get_iv_rank as get_yfinance_iv_rank,
)
from .macro_data import get_macro_environment as get_fred_macro_environment
from .earnings_transcript import (
    get_earnings_transcript_sentiment as get_motley_fool_earnings_transcript_sentiment,
)
from .sector_analysis import (
    get_sector_relative_strength as get_yfinance_sector_relative_strength,
    get_intermarket_correlations as get_yfinance_intermarket_correlations,
)

# Configuration and routing logic
from .config import get_config

# Tools organized by category
TOOLS_CATEGORIES = {
    "core_stock_apis": {
        "description": "OHLCV stock price data",
        "tools": [
            "get_stock_data"
        ]
    },
    "technical_indicators": {
        "description": "Technical analysis indicators",
        "tools": [
            "get_indicators"
        ]
    },
    "fundamental_data": {
        "description": "Company fundamentals",
        "tools": [
            "get_fundamentals",
            "get_balance_sheet",
            "get_cashflow",
            "get_income_statement"
        ]
    },
    "news_data": {
        "description": "News and insider data",
        "tools": [
            "get_news",
            "get_global_news",
            "get_insider_transactions",
        ]
    },
    "political_data": {
        "description": "Congressional STOCK Act disclosures",
        "tools": [
            "get_congress_trades",
        ]
    },
    "options_data": {
        "description": "Options-flow positioning signals",
        "tools": [
            "get_options_summary",
            "get_iv_rank",
        ]
    },
    "macro_data": {
        "description": "Macro yields, spreads, and dollar trend",
        "tools": [
            "get_macro_environment",
        ]
    },
    "transcript_data": {
        "description": "Earnings call transcript sentiment",
        "tools": [
            "get_earnings_transcript_sentiment",
        ]
    },
    "sector_data": {
        "description": "Sector relative strength and inter-market correlations",
        "tools": [
            "get_sector_relative_strength",
            "get_intermarket_correlations",
        ]
    },
}

VENDOR_LIST = [
    "yfinance",
    "alpha_vantage",
    "sec",
    "finnhub",
    "fred",
    "motley_fool",
    "lambda_finance",
]

# Mapping of methods to their vendor-specific implementations
VENDOR_METHODS = {
    # core_stock_apis
    "get_stock_data": {
        "alpha_vantage": get_alpha_vantage_stock,
        "yfinance": get_YFin_data_online,
    },
    # technical_indicators
    "get_indicators": {
        "alpha_vantage": get_alpha_vantage_indicator,
        "yfinance": get_stock_stats_indicators_window,
    },
    # fundamental_data
    "get_fundamentals": {
        "alpha_vantage": get_alpha_vantage_fundamentals,
        "yfinance": get_yfinance_fundamentals,
    },
    "get_balance_sheet": {
        "alpha_vantage": get_alpha_vantage_balance_sheet,
        "yfinance": get_yfinance_balance_sheet,
        "lambda_finance": get_lambda_finance_balance_sheet,
    },
    "get_cashflow": {
        "alpha_vantage": get_alpha_vantage_cashflow,
        "yfinance": get_yfinance_cashflow,
    },
    "get_income_statement": {
        "alpha_vantage": get_alpha_vantage_income_statement,
        "yfinance": get_yfinance_income_statement,
        "lambda_finance": get_lambda_finance_income_statement,
    },
    # news_data
    "get_news": {
        "alpha_vantage": get_alpha_vantage_news,
        "yfinance": get_news_yfinance,
    },
    "get_global_news": {
        "yfinance": get_global_news_yfinance,
        "alpha_vantage": get_alpha_vantage_global_news,
    },
    "get_insider_transactions": {
        "alpha_vantage": get_alpha_vantage_insider_transactions,
        "yfinance": get_yfinance_insider_transactions,
        "sec": get_sec_insider_transactions,
    },
    # political_data
    "get_congress_trades": {
        # Internally tries Finnhub first (uses FINNHUB_API_KEY) then falls
        # back to Senate Stock Watcher (no key, Senate-only).
        "finnhub": get_finnhub_congress_trades,
    },
    # options_data
    "get_options_summary": {
        "yfinance": get_yfinance_options_summary,
    },
    "get_iv_rank": {
        "yfinance": get_yfinance_iv_rank,
    },
    # macro_data
    "get_macro_environment": {
        "fred": get_fred_macro_environment,
    },
    # transcript_data
    "get_earnings_transcript_sentiment": {
        "motley_fool": get_motley_fool_earnings_transcript_sentiment,
    },
    # sector_data
    "get_sector_relative_strength": {
        "yfinance": get_yfinance_sector_relative_strength,
    },
    "get_intermarket_correlations": {
        "yfinance": get_yfinance_intermarket_correlations,
    },
}

def get_category_for_method(method: str) -> str:
    """Get the category that contains the specified method."""
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any category")

def get_vendor(category: str, method: str = None) -> str:
    """Get the configured vendor for a data category or specific tool method.
    Tool-level configuration takes precedence over category-level.
    """
    config = get_config()

    # Check tool-level configuration first (if method provided)
    if method:
        tool_vendors = config.get("tool_vendors", {})
        if method in tool_vendors:
            return tool_vendors[method]

    # Fall back to category-level configuration
    return config.get("data_vendors", {}).get(category, "default")

def route_to_vendor(method: str, *args, **kwargs):
    """Route method calls to appropriate vendor implementation with fallback support."""
    category = get_category_for_method(method)
    vendor_config = get_vendor(category, method)
    primary_vendors = [v.strip() for v in vendor_config.split(',')]

    if method not in VENDOR_METHODS:
        raise ValueError(f"Method '{method}' not supported")

    # Build fallback chain: primary vendors first, then remaining available vendors
    all_available_vendors = list(VENDOR_METHODS[method].keys())
    fallback_vendors = primary_vendors.copy()
    for vendor in all_available_vendors:
        if vendor not in fallback_vendors:
            fallback_vendors.append(vendor)

    for vendor in fallback_vendors:
        if vendor not in VENDOR_METHODS[method]:
            continue

        vendor_impl = VENDOR_METHODS[method][vendor]
        impl_func = vendor_impl[0] if isinstance(vendor_impl, list) else vendor_impl

        try:
            return impl_func(*args, **kwargs)
        except AlphaVantageRateLimitError:
            continue  # Only rate limits trigger fallback

    raise RuntimeError(f"No available vendor for '{method}'")