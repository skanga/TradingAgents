"""Lambda Finance SEC fundamentals adapter.

Two endpoints, both free-tier:

- ``GET /api/sec/income-statement/{ticker}`` — params: ``years``, ``period``
- ``GET /api/sec/balance-sheet/{ticker}``    — params: ``years``

Both return JSON with a list of fiscal-period rows. The response shape isn't
pinned in the public docs; the parser tolerates the documented envelope
``{"status": ..., "data": ...}`` plus several common alternates and
field-name spellings (snake_case, camelCase, Alpha-Vantage-style names).

The adapter returns a Markdown report (header + period-by-period table). On
any failure (auth, HTTP error, parse error, no rows), it returns a bracketed
string starting with ``[`` so the calling agent can keep going — same
contract as every other vendor in this package.

Caching uses :mod:`tradingagents.dataflows._cache` with a 24-hour TTL keyed
on (ticker, endpoint, freq, curr_date). SEC filings update infrequently, so
re-fetching more often than that is wasted bandwidth.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

_SOURCE = "lambda_finance_sec"
_BASE_URL = "https://www.lambdafin.com/api/sec"
_TIMEOUT = 30
_DEFAULT_YEARS = 4
_CACHE_TTL_SECONDS = 24 * 3600


# --- Public API -------------------------------------------------------------


def get_income_statement(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Fetch income-statement rows for ``ticker`` and render as Markdown."""
    return _fetch_and_format(
        ticker=ticker,
        freq=freq,
        curr_date=curr_date,
        endpoint="income-statement",
        statement_label="Income Statement",
    )


def get_balance_sheet(
    ticker: str,
    freq: str = "quarterly",
    curr_date: Optional[str] = None,
) -> str:
    """Fetch balance-sheet rows for ``ticker`` and render as Markdown.

    The Lambda Finance balance-sheet endpoint doesn't accept a ``period``
    parameter — ``freq`` is honoured only as a post-fetch filter.
    """
    return _fetch_and_format(
        ticker=ticker,
        freq=freq,
        curr_date=curr_date,
        endpoint="balance-sheet",
        statement_label="Balance Sheet",
    )


# --- Core fetch + format ----------------------------------------------------


def _fetch_and_format(
    *,
    ticker: str,
    freq: str,
    curr_date: Optional[str],
    endpoint: str,
    statement_label: str,
) -> str:
    try:
        ticker_safe = safe_ticker_component(ticker)
    except ValueError as e:
        return f"[{statement_label} unavailable for {ticker!r}: {e}]"

    config = get_config()
    api_key = (config.get("lambda_finance_api_key") or "").strip()
    if not api_key:
        return (
            f"[{statement_label} unavailable for {ticker_safe}: "
            f"LAMBDA_FINANCE_API_KEY not set. Proceed with available data.]"
        )

    cache_key = {
        "endpoint": endpoint,
        "ticker": ticker_safe.upper(),
        "freq": freq,
        "curr_date": curr_date or "",
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        rows = _fetch(endpoint, ticker_safe, freq, api_key)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Lambda Finance %s failed for %s: %s", endpoint, ticker_safe, e
        )
        return (
            f"[{statement_label} unavailable for {ticker_safe}: {e}. "
            f"Proceed with available data.]"
        )

    rows = _filter_by_curr_date(rows, curr_date)
    if not rows:
        return (
            f"[{statement_label} unavailable for {ticker_safe}: "
            f"no rows returned by Lambda Finance for freq={freq}, "
            f"curr_date={curr_date or 'n/a'}. Proceed with available data.]"
        )

    report = _format_statement(ticker_safe.upper(), statement_label, freq, rows)
    cache_put(_SOURCE, cache_key, report)
    return report


def _fetch(endpoint: str, ticker: str, freq: str, api_key: str) -> List[Dict[str, Any]]:
    """Hit the Lambda Finance endpoint and return a normalised list of rows."""
    params: Dict[str, Any] = {"years": _DEFAULT_YEARS}
    # Income-statement supports ``period``; balance-sheet ignores it.
    if endpoint == "income-statement" and freq.lower() == "annual":
        params["period"] = "FY"
    resp = requests.get(
        f"{_BASE_URL}/{endpoint}/{ticker.upper()}",
        params=params,
        headers={"X-API-Key": api_key},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    return _extract_rows(body)


def _extract_rows(body: Any) -> List[Dict[str, Any]]:
    """Pull the list of fiscal-period rows out of a Lambda response.

    Accepts a top-level list, ``{"data": [...]}``, ``{"data": {"reports":
    [...]}}``, or any of the alternative inner keys we've seen in similar APIs
    (``annualReports``, ``quarterlyReports``, ``periods``, ``results``,
    ``items``).
    """
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    data = body.get("data", body)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        merged: List[Dict[str, Any]] = []
        for key in (
            "reports",
            "periods",
            "results",
            "items",
            "annualReports",
            "quarterlyReports",
        ):
            inner = data.get(key)
            if isinstance(inner, list):
                merged.extend(r for r in inner if isinstance(r, dict))
        if merged:
            return merged
    return []


# --- Filtering --------------------------------------------------------------


def _filter_by_curr_date(rows: List[Dict[str, Any]], curr_date: Optional[str]) -> List[Dict[str, Any]]:
    """Drop rows whose period end date is after ``curr_date`` (look-ahead bias)."""
    if not curr_date:
        return rows
    cutoff = curr_date
    out: List[Dict[str, Any]] = []
    for r in rows:
        end = _row_period_end(r)
        if not end or end <= cutoff:
            out.append(r)
    return out


def _row_period_end(row: Dict[str, Any]) -> str:
    """Best-effort cutoff date for look-ahead bias filtering, YYYY-MM-DD.

    For Lambda Finance, ``filing_date`` is the only reliable date — its
    ``end_date``/``report_date`` fields are populated with the prior-year
    comparative period from the filing, not the current period. Filing-date
    semantics are also more correct for look-ahead filtering: it's when the
    data was publicly available, which is what we actually need to gate.
    The other aliases keep this helper portable in case a sibling vendor
    is ever routed through the same module.
    """
    for key in (
        "filing_date",
        "fiscalDateEnding",
        "periodEndDate",
        "period_end",
        "periodEnd",
        "endDate",
        "end_date",
        "report_date",
        "date",
    ):
        v = row.get(key)
        if isinstance(v, str) and len(v) >= 10:
            return v[:10]
    return ""


def _row_period_label(row: Dict[str, Any]) -> str:
    """Human-readable period label for column headers.

    Prefers ``(fiscal_year, fiscal_period)`` over any date field — Lambda's
    raw date fields are unreliable (see ``_row_period_end`` docstring), and
    ``"FY2025"`` / ``"Q1 FY2026"`` reads better than a YYYY-MM-DD anyway.
    """
    fy = row.get("fiscal_year") or row.get("fiscalYear")
    fp = row.get("fiscal_period") or row.get("fiscalPeriod")
    if fy and fp:
        if str(fp).upper() == "FY":
            return f"FY{fy}"
        return f"{fp} FY{fy}"
    end = _row_period_end(row)
    return end or "—"


# --- Formatting -------------------------------------------------------------

# Display label → list of source-field aliases (first match wins). Keeping the
# subset focused on what an analyst actually reads avoids drowning the LLM in
# 30+ raw line items.
_INCOME_STATEMENT_FIELDS: List[Tuple[str, List[str]]] = [
    ("Revenue",          ["revenue", "totalRevenue", "total_revenue"]),
    ("Gross Profit",     ["grossProfit", "gross_profit"]),
    ("Operating Income", ["operatingIncome", "operating_income", "operatingIncomeLoss"]),
    ("Net Income",       ["netIncome", "net_income", "netIncomeApplicableToCommonShares"]),
    ("EPS (Diluted)",    ["dilutedEPS", "epsDiluted", "diluted_eps", "eps_diluted"]),
    ("EPS (Basic)",      ["basicEPS", "epsBasic", "basic_eps", "eps_basic"]),
]

_BALANCE_SHEET_FIELDS: List[Tuple[str, List[str]]] = [
    ("Total Assets",            ["totalAssets", "total_assets"]),
    ("Total Liabilities",       ["totalLiabilities", "total_liabilities"]),
    ("Total Equity",            ["totalEquity", "totalShareholderEquity", "total_equity", "stockholders_equity", "stockholdersEquity"]),
    ("Cash & Equivalents",      ["cashAndCashEquivalents", "cash_and_cash_equivalents", "cash_and_equivalents", "cash"]),
    ("Current Assets",          ["currentAssets", "current_assets", "totalCurrentAssets"]),
    ("Current Liabilities",     ["currentLiabilities", "current_liabilities", "totalCurrentLiabilities"]),
    ("Long-Term Debt",          ["longTermDebt", "long_term_debt"]),
    ("Short-Term Debt",         ["shortTermDebt", "short_term_debt"]),
]


def _format_statement(
    ticker: str,
    label: str,
    freq: str,
    rows: List[Dict[str, Any]],
) -> str:
    """Render one statement (income or balance) as a Markdown table."""
    fields = _INCOME_STATEMENT_FIELDS if label == "Income Statement" else _BALANCE_SHEET_FIELDS

    rows_sorted = sorted(rows, key=_row_period_end, reverse=True)
    period_labels = [_row_period_label(r) for r in rows_sorted]

    lines = [
        f"## {label} for {ticker} ({freq})",
        f"_Source: Lambda Finance (SEC). Retrieved {datetime.utcnow().strftime('%Y-%m-%d')}._",
        "",
    ]
    header = "| Line Item | " + " | ".join(period_labels) + " |"
    sep = "|---|" + "|".join(["---"] * len(period_labels)) + "|"
    lines.append(header)
    lines.append(sep)

    for label_text, aliases in fields:
        cells = [_format_cell(_pick_value(r, aliases)) for r in rows_sorted]
        lines.append(f"| {label_text} | " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(f"_{len(rows_sorted)} period(s) shown. Values in reporting currency as filed._")
    return "\n".join(lines)


def _pick_value(row: Dict[str, Any], aliases: List[str]) -> Any:
    for key in aliases:
        if key in row and row[key] is not None and row[key] != "":
            return row[key]
    return None


def _format_cell(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        return _money(float(v))
    s = str(v).strip()
    if not s or s.lower() == "none":
        return "—"
    try:
        return _money(float(s))
    except (TypeError, ValueError):
        return s


def _money(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    a = abs(amount)
    if a >= 1_000_000_000:
        return f"{sign}${a / 1_000_000_000:.2f}B"
    if a >= 1_000_000:
        return f"{sign}${a / 1_000_000:.2f}M"
    if a >= 1_000:
        return f"{sign}${a / 1_000:.2f}k"
    if a >= 10:
        return f"{sign}${a:.0f}"
    # Likely a per-share figure (EPS) — keep two decimals.
    return f"{sign}${a:.2f}"
