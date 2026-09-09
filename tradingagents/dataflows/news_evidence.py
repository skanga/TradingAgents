"""Vendor-independent news identity and provenance, without sentiment weighting."""

import hashlib
import json
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock

from .config import get_config
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .date_window import to_utc

_run_cache: ContextVar[tuple[dict, RLock] | None] = ContextVar("news_run_cache", default=None)


@contextmanager
def news_run_scope():
    """Fresh cache for each graph/GUI run, inherited by tool execution contexts."""
    token = _run_cache.set(({}, RLock()))
    try:
        yield
    finally:
        _run_cache.reset(token)


def shared_news_request(method: str, args: tuple, fetch) -> str:
    """Cache successful requests only within one run and effective configuration."""
    scope = _run_cache.get()
    if scope is None:
        return fetch()
    cache, lock = scope
    key = (method, args, json.dumps(get_config(), sort_keys=True, default=str))
    # ToolNode can issue identical requests concurrently. The lock makes those
    # requests share one snapshot rather than race two vendor fetches.
    with lock:
        if key not in cache:
            cache[key] = fetch()
        return cache[key]


SHARED_NEWS_INSTRUCTION = (
    "Preserve NEWS- evidence IDs when citing articles. Repeated IDs in sentiment, "
    "news, or other reports refer to the same evidence, not independent corroboration. "
    "Count that evidence once when synthesizing conclusions. Distinguish the "
    "retrieval vendor from the original publisher; missing coverage is not silence."
)


def evidence_id(article: dict) -> str:
    """Normalize known tracking parameters; retain meaningful URL query parameters."""
    link = article.get("link") or ""
    if link:
        parts = urlsplit(link)
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"guccounter", "gclid", "fbclid"}]
        identity = urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path,
                               urlencode(sorted(query)), ""))
    else:
        date = article.get("pub_date")
        identity = json.dumps([
            " ".join((article.get("title") or "").casefold().split()),
            (article.get("publisher") or "").casefold(),
            to_utc(date).isoformat() if date is not None else None,
        ])
    return "NEWS-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def render_article(article: dict, vendor: str) -> str:
    date = article.get("pub_date")
    published = to_utc(date).isoformat() if date is not None else "unknown"
    return (
        f"### {article['title']} (source: {article['publisher']})\n"
        f"Evidence ID: {evidence_id(article)}\n"
        f"Retrieval vendor: {vendor}\nPublished: {published}\n"
        + (f"{article['summary']}\n" if article.get("summary") else "")
        + (f"Link: {article['link']}\n" if article.get("link") else "") + "\n"
    )


YAHOO_COVERAGE_NOTICE = (
    "Coverage limitation: Yahoo supplies a bounded recent feed, not a complete "
    "historical archive. Missing matches are not evidence of an absence of news."
)
