"""Lambda Finance peer-comparison adapter.

Single endpoint, free-tier:

    GET /api/sec/compare?tickers={t1,t2,...}&metrics={m1,m2,...}&year={Y}

Returns a flat top-level array of dicts, one per ticker that Lambda has
data for, of shape::

    [
      {"ticker": "AAPL", "company_name": "Apple Inc.",
       "revenue": 391035000000.0, "net_income": 93736000000.0, ...},
      ...
    ]

Lambda silently drops tickers it has no data for (rather than returning
a row of nulls), so the formatter explicitly surfaces missing tickers
in a footer note — the absence is signal an analyst should see, not
something the report should hide.

Returns a Markdown comparison table on success, a bracketed "[…]" string
on failure. Same contract as every other vendor adapter in this package.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

_SOURCE = "lambda_finance_compare"
_URL = "https://www.lambdafin.com/api/sec/compare"
_TIMEOUT = 30
_CACHE_TTL_SECONDS = 24 * 3600

# Tickers + metrics + year is small. Hard-cap so a runaway LLM tool call
# doesn't fan out arbitrarily wide.
_MAX_PEERS = 10
_MAX_METRICS = 12

# Default metric set: a focused income-statement snapshot. Analysts can
# pass a different list via the tool param if they need balance-sheet
# fields too.
_DEFAULT_METRICS = ("revenue", "net_income", "gross_profit", "operating_income")

# Display labels for known metric keys. Unknown keys fall through to a
# "Title Case" of the snake_case key so a forgotten label doesn't break
# the table — it just renders less prettily.
_METRIC_LABELS: Dict[str, str] = {
    "revenue":             "Revenue",
    "cost_of_revenue":     "Cost of Revenue",
    "gross_profit":        "Gross Profit",
    "operating_expenses":  "Operating Expenses",
    "operating_income":    "Operating Income",
    "interest_expense":    "Interest Expense",
    "income_before_tax":   "Income Before Tax",
    "income_tax_expense":  "Income Tax Expense",
    "net_income":          "Net Income",
    "eps_basic":           "EPS (Basic)",
    "eps_diluted":         "EPS (Diluted)",
    "total_assets":        "Total Assets",
    "total_liabilities":   "Total Liabilities",
    "stockholders_equity": "Stockholders' Equity",
    "cash_and_equivalents":"Cash & Equivalents",
    "long_term_debt":      "Long-Term Debt",
    "short_term_debt":     "Short-Term Debt",
    "current_assets":      "Current Assets",
    "current_liabilities": "Current Liabilities",
    "retained_earnings":   "Retained Earnings",
}


# --- Public API -------------------------------------------------------------


def get_peer_comparison(
    ticker: str,
    peers: str,
    metrics: str = "",
    year: int = 0,
) -> str:
    """Compare ``ticker`` against ``peers`` on selected ``metrics`` for ``year``.

    Args:
        ticker: Primary ticker (rendered first in the table).
        peers: Comma-separated peer tickers (e.g. ``"MSFT,GOOGL,AMZN"``).
        metrics: Comma-separated metric keys. Empty / missing falls back
            to a focused income-statement snapshot.
        year: Fiscal year (e.g. ``2024``). ``0`` falls back to last
            calendar year — the most recent full FY likely to have a
            10-K filed by mid-year.

    Returns:
        Markdown comparison table on success, a bracketed unavailable
        string on any failure (auth missing, HTTP error, parse error,
        empty response).
    """
    try:
        primary = safe_ticker_component(ticker)
    except ValueError as e:
        return f"[Peer comparison unavailable for {ticker!r}: {e}]"

    try:
        peer_list = _parse_ticker_csv(peers)
    except ValueError as e:
        return f"[Peer comparison unavailable: bad peer list for {ticker}: {e}]"
    if not peer_list:
        return (
            f"[Peer comparison unavailable for {primary}: no peers supplied. "
            f"Pass a comma-separated list (e.g. 'MSFT,GOOGL,AMZN').]"
        )

    metric_list = _parse_metrics(metrics)
    fiscal_year = year if year > 0 else datetime.utcnow().year - 1

    config = get_config()
    api_key = (config.get("lambda_finance_api_key") or "").strip()
    if not api_key:
        return (
            f"[Peer comparison unavailable for {primary}: "
            f"LAMBDA_FINANCE_API_KEY not set. Proceed with available data.]"
        )

    requested = [primary, *peer_list]
    cache_key = {
        "ticker": primary.upper(),
        "peers": ",".join(p.upper() for p in peer_list),
        "metrics": ",".join(metric_list),
        "year": fiscal_year,
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        rows = _fetch(requested, metric_list, fiscal_year, api_key)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Lambda Finance compare failed for %s vs %s (FY%d): %s",
            primary, ",".join(peer_list), fiscal_year, e,
        )
        return (
            f"[Peer comparison unavailable for {primary}: {e}. "
            f"Proceed with available data.]"
        )

    if not rows:
        return (
            f"[Peer comparison unavailable for {primary}: Lambda returned no "
            f"rows for any of {', '.join(requested)} in FY{fiscal_year}. "
            f"Proceed with available data.]"
        )

    report = _format_table(
        primary=primary,
        requested=requested,
        rows=rows,
        metric_keys=metric_list,
        fiscal_year=fiscal_year,
    )
    cache_put(_SOURCE, cache_key, report)
    return report


# --- Fetch + parse ----------------------------------------------------------


def _fetch(
    tickers: Sequence[str],
    metrics: Sequence[str],
    year: int,
    api_key: str,
) -> List[Dict[str, Any]]:
    resp = requests.get(
        _URL,
        params={
            "tickers": ",".join(t.upper() for t in tickers),
            "metrics": ",".join(metrics),
            "year": year,
        },
        headers={"X-API-Key": api_key},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    return _extract_rows(body)


def _extract_rows(body: Any) -> List[Dict[str, Any]]:
    """Lambda returns a top-level list, but accept the documented
    ``{data: [...]}`` envelope and ``{data: {results: [...]}}`` shape too —
    keeps the parser tolerant if Lambda harmonises endpoint envelopes
    later without breaking us."""
    if isinstance(body, list):
        return [r for r in body if isinstance(r, dict)]
    if not isinstance(body, dict):
        return []
    data = body.get("data", body)
    if isinstance(data, list):
        return [r for r in data if isinstance(r, dict)]
    if isinstance(data, dict):
        for key in ("results", "comparison", "rows", "items"):
            inner = data.get(key)
            if isinstance(inner, list):
                return [r for r in inner if isinstance(r, dict)]
    return []


# --- Input validation -------------------------------------------------------


def _parse_ticker_csv(raw: str) -> List[str]:
    """Split a comma-separated ticker list, validate each, dedupe."""
    if not isinstance(raw, str):
        return []
    seen: set[str] = set()
    out: List[str] = []
    for chunk in raw.split(","):
        sym = chunk.strip().upper()
        if not sym or sym in seen:
            continue
        # Raises ValueError for unsafe components — propagated to caller.
        safe_ticker_component(sym)
        out.append(sym)
        seen.add(sym)
    if len(out) > _MAX_PEERS:
        raise ValueError(
            f"too many peers ({len(out)} supplied, max {_MAX_PEERS})"
        )
    return out


def _parse_metrics(raw: str) -> List[str]:
    """Split + lower-case + dedupe metric keys, capping at ``_MAX_METRICS``.

    Empty / non-string falls back to ``_DEFAULT_METRICS``.
    """
    if not isinstance(raw, str) or not raw.strip():
        return list(_DEFAULT_METRICS)
    seen: set[str] = set()
    out: List[str] = []
    for chunk in raw.split(","):
        key = chunk.strip().lower()
        if not key or key in seen:
            continue
        out.append(key)
        seen.add(key)
        if len(out) >= _MAX_METRICS:
            break
    return out or list(_DEFAULT_METRICS)


# --- Formatting -------------------------------------------------------------


def _format_table(
    *,
    primary: str,
    requested: Sequence[str],
    rows: List[Dict[str, Any]],
    metric_keys: Sequence[str],
    fiscal_year: int,
) -> str:
    """Render the comparison as a Markdown table with tickers as columns.

    Tickers in ``requested`` are emitted in the order the caller asked
    for, regardless of the order Lambda returned. Tickers Lambda omitted
    appear with em-dash cells, and a footer note names them so the
    omission is visible signal — silently dropping a peer would let the
    report mislead.
    """
    by_ticker = {
        r.get("ticker", "").upper(): r for r in rows if isinstance(r, dict)
    }
    requested_upper = [t.upper() for t in requested]
    missing = [t for t in requested_upper if t not in by_ticker]

    peer_label = ", ".join(t for t in requested_upper if t != primary.upper())
    lines: List[str] = []
    lines.append(
        f"## Peer Comparison — {primary.upper()} vs {peer_label} (FY{fiscal_year})"
    )
    lines.append(
        f"_Source: Lambda Finance (SEC). "
        f"Retrieved {datetime.utcnow().strftime('%Y-%m-%d')}._"
    )
    lines.append("")

    header = "| Metric | " + " | ".join(requested_upper) + " |"
    sep = "|---|" + "|".join(["---"] * len(requested_upper)) + "|"
    lines.append(header)
    lines.append(sep)

    for key in metric_keys:
        cells: List[str] = []
        for t in requested_upper:
            row = by_ticker.get(t)
            if row is None:
                cells.append("—")
                continue
            cells.append(_format_cell(row.get(key)))
        lines.append(f"| {_metric_label(key)} | " + " | ".join(cells) + " |")

    lines.append("")
    if missing:
        lines.append(
            f"_Note: Lambda Finance returned no FY{fiscal_year} data for "
            f"{', '.join(missing)} (rendered as em-dashes above)._"
        )
    else:
        lines.append(f"_All {len(requested_upper)} requested tickers had data._")
    return "\n".join(lines)


def _metric_label(key: str) -> str:
    return _METRIC_LABELS.get(key, key.replace("_", " ").title())


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
    """Same scale logic as lambda_finance_sec._money — duplicated rather
    than imported so neither module pulls a private helper across files."""
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
    return f"{sign}${a:.2f}"
