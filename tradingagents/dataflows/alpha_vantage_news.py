import contextlib
import json
from datetime import datetime, timedelta, timezone

from .alpha_vantage_common import _make_api_request, format_datetime_for_api
from .config import get_config
from .date_window import in_window
from .errors import VendorError
from .news_evidence import evidence_id


def _enrich_news(response: str, start_date: str, end_date: str) -> str:
    try:
        payload = json.loads(response)
    except (TypeError, json.JSONDecodeError) as exc:
        raise VendorError("Alpha Vantage news response was not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("feed"), list):
        raise VendorError("Alpha Vantage news response has no valid feed")
    start, end = datetime.fromisoformat(start_date), datetime.fromisoformat(end_date)
    seen = set()
    articles = []
    for item in payload["feed"]:
        if not isinstance(item, dict):
            raise VendorError("Alpha Vantage news feed contains a malformed article")
        published = None
        with contextlib.suppress(ValueError, TypeError):
            published = datetime.strptime(item.get("time_published", ""), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        if not in_window(published, start, end):
            continue
        identity = evidence_id({"title": item.get("title", ""), "publisher": item.get("source", ""),
                                "link": item.get("url", ""), "pub_date": published})
        if identity in seen:
            continue
        seen.add(identity)
        articles.append({**item, "evidence_id": identity, "retrieval_vendor": "alpha_vantage",
                         "publisher": item.get("source", "Unknown"),
                         "published_at": published.isoformat() if published else None})
    payload.update({"feed": articles, "items": str(len(articles)), "retrieval_vendor": "alpha_vantage",
                    "coverage": "Bounded vendor results; missing matches do not establish an absence of news."})
    return json.dumps(payload, ensure_ascii=False)

def get_news(ticker, start_date, end_date) -> str:
    """Returns live and historical market news & sentiment data from premier news outlets worldwide.

    Covers stocks, cryptocurrencies, forex, and topics like fiscal policy, mergers & acquisitions, IPOs.

    Args:
        ticker: Stock symbol for news articles.
        start_date: Start date for news search.
        end_date: End date for news search.

    Returns:
        Dictionary containing news sentiment data or JSON string.
    """

    params = {
        "tickers": ticker,
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api((datetime.fromisoformat(end_date) + timedelta(days=1)).strftime("%Y-%m-%d")),
    }

    return _enrich_news(_make_api_request("NEWS_SENTIMENT", params), start_date, end_date)

def get_global_news(curr_date, look_back_days: int | None = None, limit: int | None = None) -> str:
    """Returns global market news & sentiment data without ticker-specific filtering.

    Covers broad market topics like financial markets, economy, and more.

    Args:
        curr_date: Current date in yyyy-mm-dd format.
        look_back_days: Number of days to look back (default 7).
        limit: Maximum number of articles (default 50).

    Returns:
        Dictionary containing global news sentiment data or JSON string.
    """
    config = get_config()
    look_back_days = config["global_news_lookback_days"] if look_back_days is None else look_back_days
    limit = config["global_news_article_limit"] if limit is None else limit

    # Calculate start date
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - timedelta(days=look_back_days)
    start_date = start_dt.strftime("%Y-%m-%d")

    params = {
        "topics": "financial_markets,economy_macro,economy_monetary",
        "time_from": format_datetime_for_api(start_date),
        "time_to": format_datetime_for_api((curr_dt + timedelta(days=1)).strftime("%Y-%m-%d")),
        "limit": str(limit),
    }

    return _enrich_news(_make_api_request("NEWS_SENTIMENT", params), start_date, curr_date)


def get_insider_transactions(symbol: str) -> str:
    """Returns latest and historical insider transactions by key stakeholders.

    Covers transactions by founders, executives, board members, etc.

    Args:
        symbol: Ticker symbol. Example: "IBM".

    Returns:
        Dictionary containing insider transaction data or JSON string.
    """

    params = {
        "symbol": symbol,
    }

    return _make_api_request("INSIDER_TRANSACTIONS", params)
