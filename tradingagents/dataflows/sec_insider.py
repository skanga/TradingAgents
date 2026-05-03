"""SEC EDGAR Form 4 insider transaction adapter.

Free, no API key required. SEC asks consumers to identify themselves with a
descriptive ``User-Agent`` (read from ``sec_user_agent`` in DEFAULT_CONFIG)
and to keep request rate at or below 10/s.

Pipeline:
1. Resolve ``ticker`` → CIK via the public ``company_tickers.json`` map.
2. Fetch the issuer's recent submissions via
   ``https://data.sec.gov/submissions/CIK{CIK}.json``.
3. For each Form 4 inside the lookback window, fetch the primary XML
   document and parse out the non-derivative transactions.
4. Aggregate into a Markdown report with per-transaction rows plus a
   cluster summary (unique buyers vs sellers, large-purchase flag).

Caching uses :mod:`tradingagents.dataflows._cache`:
- CIK map: 7 days (changes are rare)
- Submissions: 1 hour (filings appear daily)
- Per-Form-4 XML: 30 days (immutable once filed)
- Final report per (ticker, lookback): 1 hour
"""

from __future__ import annotations

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Optional, TypedDict

import requests

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_SOURCE = "sec_form4"
_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_DOC_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_dashes}/{primary}"
_RATE_LIMIT_SLEEP = 0.15  # 10 req/s ceiling per SEC fair-access guidelines
_MAX_FILINGS_PER_RUN = 50  # cap so a heavy filer doesn't exhaust the rate budget


class _Transaction(TypedDict, total=False):
    date: str
    filer: str
    title: str
    code: str          # P (purchase), S (sale), A, M, ...
    direction: str     # "A" acquired (buy) or "D" disposed (sell)
    shares: float
    price: float
    value: float
    post_shares: float


def get_insider_transactions(ticker: str, lookback_days: int = 90) -> str:
    """Fetch SEC Form 4 insider filings for ``ticker`` over the last ``lookback_days``."""
    cache_key = {"ticker": ticker.upper(), "lookback_days": lookback_days}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=3600)
    if cached is not None:
        return cached

    try:
        cik = _ticker_to_cik(ticker)
        if cik is None:
            return f"[SEC Form 4: no CIK match for {ticker}. Proceed with available data.]"

        submissions = _get_submissions(cik)
        cutoff = datetime.utcnow().date() - timedelta(days=lookback_days)
        form4s = _select_recent_form4s(submissions, cutoff)

        transactions: List[_Transaction] = []
        for filing in form4s[:_MAX_FILINGS_PER_RUN]:
            xml = _fetch_form4_xml(cik, filing["accession"], filing["primary_doc"])
            if xml is None:
                continue
            transactions.extend(_parse_form4_xml(xml))
            time.sleep(_RATE_LIMIT_SLEEP)

        report = _format_report(ticker, transactions, lookback_days, len(form4s))
        cache_put(_SOURCE, cache_key, report)
        return report
    except requests.RequestException as e:
        logger.warning("SEC EDGAR network error for %s: %s", ticker, e)
        return f"[SEC Form 4 unavailable: {e}. Proceed with available data.]"
    except Exception as e:
        logger.exception("SEC EDGAR unexpected error for %s", ticker)
        return f"[SEC Form 4 unavailable: {e}. Proceed with available data.]"


# --- HTTP helpers -----------------------------------------------------------


def _session() -> requests.Session:
    s = requests.Session()
    ua = get_config().get("sec_user_agent") or "TradingAgents contact@example.com"
    s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
    return s


def _ticker_to_cik(ticker: str) -> Optional[str]:
    """Map ``ticker`` to a 10-digit zero-padded CIK string. None if unknown."""
    cache_key = {"src": _SOURCE, "kind": "cik_map"}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=7 * 24 * 3600)
    if cached is not None:
        mapping = json.loads(cached)
    else:
        resp = _session().get(_TICKER_MAP_URL, timeout=15)
        resp.raise_for_status()
        raw = resp.json()
        # company_tickers.json is keyed by index ("0", "1", ...) with
        # {cik_str, ticker, title} values. Flatten to {TICKER: cik}.
        mapping = {row["ticker"].upper(): str(row["cik_str"]) for row in raw.values()}
        cache_put(_SOURCE, cache_key, json.dumps(mapping))
    cik = mapping.get(ticker.upper())
    return cik.zfill(10) if cik else None


def _get_submissions(cik_padded: str) -> dict:
    cache_key = {"src": _SOURCE, "kind": "submissions", "cik": cik_padded}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=3600)
    if cached is not None:
        return json.loads(cached)
    resp = _session().get(_SUBMISSIONS_URL.format(cik=cik_padded), timeout=15)
    resp.raise_for_status()
    body = resp.text
    cache_put(_SOURCE, cache_key, body)
    return json.loads(body)


def _select_recent_form4s(submissions: dict, cutoff_date) -> List[dict]:
    """Pull recent Form 4 entries from the submissions JSON within ``cutoff_date``."""
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    primary_docs = recent.get("primaryDocument", [])

    out = []
    for form, acc, fdate, primary in zip(forms, accessions, filing_dates, primary_docs):
        if form != "4":
            continue
        try:
            d = datetime.strptime(fdate, "%Y-%m-%d").date()
        except ValueError:
            continue
        if d < cutoff_date:
            continue
        out.append({"accession": acc, "filing_date": fdate, "primary_doc": primary})
    # newest first
    out.sort(key=lambda f: f["filing_date"], reverse=True)
    return out


def _fetch_form4_xml(cik_padded: str, accession: str, primary: str) -> Optional[str]:
    """Fetch the primary XML document for a Form 4 filing.

    Form 4 filings have ``primary_doc`` set to the XML filename (e.g.
    ``wf-form4_171234567890.xml``); rare amendments may use a different
    extension, in which case we skip the filing rather than guess.
    """
    if not primary or not primary.lower().endswith(".xml"):
        return None
    cache_key = {"src": _SOURCE, "kind": "form4_xml", "acc": accession}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=30 * 24 * 3600)
    if cached is not None:
        return cached
    cik_int = str(int(cik_padded))  # the Archives URL wants the un-padded CIK
    acc_no_dashes = accession.replace("-", "")
    url = _DOC_URL.format(cik_int=cik_int, acc_no_dashes=acc_no_dashes, primary=primary)
    try:
        resp = _session().get(url, timeout=15)
        resp.raise_for_status()
        cache_put(_SOURCE, cache_key, resp.text)
        return resp.text
    except requests.RequestException as e:
        logger.warning("SEC EDGAR fetch failed for %s: %s", accession, e)
        return None


# --- XML parsing ------------------------------------------------------------


def _xml_text(node: Optional[ET.Element], path: str) -> str:
    """Return the trimmed text at ``path`` under ``node``, or empty string."""
    if node is None:
        return ""
    found = node.find(path)
    if found is None or found.text is None:
        return ""
    return found.text.strip()


def _xml_float(node: Optional[ET.Element], path: str) -> Optional[float]:
    raw = _xml_text(node, path)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _parse_form4_xml(xml_text: str) -> List[_Transaction]:
    """Extract non-derivative transactions from a Form 4 XML document."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("Form 4 XML parse error: %s", e)
        return []

    # Reporting owner (first one — Form 4 can have multiple but most have one)
    owner_el = root.find("reportingOwner")
    name = _xml_text(owner_el, "reportingOwnerId/rptOwnerName")
    rel = owner_el.find("reportingOwnerRelationship") if owner_el is not None else None
    title_parts: List[str] = []
    if rel is not None:
        if _xml_text(rel, "isOfficer") in ("1", "true"):
            t = _xml_text(rel, "officerTitle")
            title_parts.append(t or "Officer")
        if _xml_text(rel, "isDirector") in ("1", "true"):
            title_parts.append("Director")
        if _xml_text(rel, "isTenPercentOwner") in ("1", "true"):
            title_parts.append("10% Owner")
    title = ", ".join(title_parts) or "—"

    out: List[_Transaction] = []
    for txn in root.findall("nonDerivativeTable/nonDerivativeTransaction"):
        date = _xml_text(txn, "transactionDate/value")
        code = _xml_text(txn, "transactionCoding/transactionCode")
        shares = _xml_float(txn, "transactionAmounts/transactionShares/value")
        price = _xml_float(txn, "transactionAmounts/transactionPricePerShare/value")
        direction = _xml_text(txn, "transactionAmounts/transactionAcquiredDisposedCode/value")
        post_shares = _xml_float(
            txn, "postTransactionAmounts/sharesOwnedFollowingTransaction/value"
        )
        if shares is None:
            continue
        value = shares * price if price is not None else 0.0
        out.append(
            _Transaction(
                date=date or "—",
                filer=name or "—",
                title=title,
                code=code or "—",
                direction=direction or "—",
                shares=shares,
                price=price if price is not None else 0.0,
                value=value,
                post_shares=post_shares if post_shares is not None else 0.0,
            )
        )
    return out


# --- Reporting --------------------------------------------------------------


_LARGE_PURCHASE_USD = 500_000


def _format_report(
    ticker: str,
    txns: List[_Transaction],
    lookback_days: int,
    filing_count: int,
) -> str:
    if not txns:
        if filing_count == 0:
            return (
                f"## SEC Form 4 Insider Transactions for {ticker} — "
                f"last {lookback_days} days\n\n"
                f"No Form 4 filings recorded in this window.\n"
            )
        return (
            f"## SEC Form 4 Insider Transactions for {ticker} — "
            f"last {lookback_days} days\n\n"
            f"{filing_count} filings retrieved but no parseable non-derivative "
            "transactions (likely all derivative grants/exercises).\n"
        )

    purchases = [t for t in txns if t["direction"] == "A" and t["code"] == "P"]
    sales = [t for t in txns if t["direction"] == "D" and t["code"] == "S"]
    unique_buyers = {t["filer"] for t in purchases}
    unique_sellers = {t["filer"] for t in sales}
    bought_total = sum(t["value"] for t in purchases)
    sold_total = sum(t["value"] for t in sales)
    big = [t for t in purchases if t["value"] >= _LARGE_PURCHASE_USD]

    lines = [
        f"## SEC Form 4 Insider Transactions for {ticker} — last {lookback_days} days",
        "",
        f"**Cluster summary**: {len(unique_buyers)} unique buyer(s) vs "
        f"{len(unique_sellers)} unique seller(s) "
        f"(net {len(unique_buyers) - len(unique_sellers):+d} buyers). "
        f"Volume: ${bought_total:,.0f} bought, ${sold_total:,.0f} sold.",
    ]
    if big:
        lead = big[0]
        lines.append(
            f"**Notable**: {len(big)} large purchase(s) ≥ "
            f"${_LARGE_PURCHASE_USD:,} (largest: {lead['filer']}, "
            f"${lead['value']:,.0f} on {lead['date']})."
        )
    lines.append("")
    lines.append("### Transactions")
    lines.append("| Date | Filer | Title | Code | Shares | Price | Value | % of Holdings |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in txns:
        # Pre-transaction holdings: post + shares (sale) or post - shares (purchase)
        if t["direction"] == "A":
            pre = t["post_shares"] - t["shares"]
        else:
            pre = t["post_shares"] + t["shares"]
        if pre and pre > 0:
            pct = f"{(t['shares'] / (pre + t['shares'])) * 100:.1f}%"
        else:
            pct = "—"
        price_str = f"${t['price']:.2f}" if t["price"] else "—"
        value_str = f"${t['value']:,.0f}" if t["value"] else "—"
        code_str = f"{t['code']} ({_code_label(t['code'], t['direction'])})"
        lines.append(
            f"| {t['date']} | {t['filer']} | {t['title']} | {code_str} "
            f"| {t['shares']:,.0f} | {price_str} | {value_str} | {pct} |"
        )
    lines.append("")
    lines.append("Source: SEC EDGAR (data.sec.gov)")
    return "\n".join(lines)


def _code_label(code: str, direction: str) -> str:
    base = {
        "P": "purchase",
        "S": "sale",
        "A": "grant",
        "M": "option exercise",
        "F": "tax withholding",
        "G": "gift",
        "D": "disposition",
        "X": "option exercise",
        "C": "conversion",
    }.get(code.upper(), "other")
    return base
