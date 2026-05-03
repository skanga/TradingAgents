"""Congressional STOCK Act disclosure adapter.

Source chain (tried in order, falling through on any error):

1. **Finnhub** — ``GET /api/v1/stock/congressional-trading?symbol={ticker}``
   with the ``FINNHUB_API_KEY`` from config. Free tier covers House + Senate.
2. **Senate Stock Watcher** — community-maintained S3 bucket of all
   parsed Senate PTRs. Free, no key, but Senate-only.

Both return Markdown via the same formatter. On any failure (missing key,
HTTP error, parse error, no rows), the function returns a bracketed
fallback string starting with ``[`` so the calling agent can keep going.

Caching uses :mod:`tradingagents.dataflows._cache`:
- Final per-ticker report: 6 hours
- Senate Stock Watcher full dataset: 24 hours (the file is ~50 MB; we
  only re-download once a day and filter in memory)
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, List, Optional, TypedDict

import requests

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_SOURCE = "congress_trades"
_FINNHUB_URL = "https://finnhub.io/api/v1/stock/congressional-trading"
_SENATE_SW_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/"
    "aggregate/all_transactions.json"
)
_TIMEOUT = 30


class _Trade(TypedDict, total=False):
    date: str           # transaction date (YYYY-MM-DD)
    filer: str          # legislator name
    chamber: str        # "House" / "Senate" / "—"
    party: str          # "D" / "R" / "I" / "—"
    state: str          # 2-letter state code or "—"
    type: str           # "Purchase" / "Sale" / "Exchange"
    amount_min: float   # USD lower bound
    amount_max: float   # USD upper bound
    amount_label: str   # human-readable range
    filing_date: str
    filing_lag_days: Optional[int]


# --- Public API -------------------------------------------------------------


def get_congress_trades(ticker: str, lookback_days: int = 180) -> str:
    """Fetch STOCK Act disclosures touching ``ticker`` over ``lookback_days``."""
    cache_key = {
        "kind": "report",
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=6 * 3600)
    if cached is not None:
        return cached

    config = get_config()
    cutoff = datetime.utcnow().date() - timedelta(days=lookback_days)
    finnhub_key = (config.get("finnhub_api_key") or "").strip()

    sources_tried: List[str] = []
    trades: List[_Trade] = []
    used_source: Optional[str] = None
    last_error: Optional[str] = None

    if finnhub_key:
        try:
            trades = _fetch_finnhub(ticker, finnhub_key, cutoff)
            used_source = "Finnhub (House + Senate)"
        except Exception as e:
            logger.warning("Finnhub congressional-trading failed for %s: %s", ticker, e)
            last_error = f"finnhub: {e}"
            sources_tried.append("finnhub")
    else:
        sources_tried.append("finnhub (no FINNHUB_API_KEY set)")

    if not trades:
        try:
            trades = _fetch_senate_stock_watcher(ticker, cutoff)
            used_source = "Senate Stock Watcher (Senate only)"
        except Exception as e:
            logger.warning("Senate Stock Watcher failed for %s: %s", ticker, e)
            last_error = f"senate_stock_watcher: {e}"
            sources_tried.append("senate_stock_watcher")

    if not trades:
        msg = (
            f"[Congressional disclosures unavailable for {ticker}: "
            f"no data from {' → '.join(sources_tried) or 'configured sources'}"
        )
        if last_error:
            msg += f"; last error {last_error}"
        msg += ". Proceed with available data.]"
        return msg

    report = _format_report(ticker, trades, lookback_days, used_source or "unknown")
    cache_put(_SOURCE, cache_key, report)
    return report


# --- Source: Finnhub --------------------------------------------------------


def _fetch_finnhub(ticker: str, api_key: str, cutoff_date) -> List[_Trade]:
    resp = requests.get(
        _FINNHUB_URL,
        params={"symbol": ticker.upper(), "token": api_key},
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    if not isinstance(body, dict):
        raise ValueError(f"unexpected response shape: {type(body).__name__}")
    rows = body.get("data") or []
    return [t for t in (_parse_finnhub_row(r) for r in rows) if t is not None and _within_cutoff(t, cutoff_date)]


def _parse_finnhub_row(row: dict) -> Optional[_Trade]:
    """Map a Finnhub row to the shared :class:`_Trade` shape."""
    try:
        date = (row.get("transactionDate") or "").strip()
        filer = (row.get("name") or "—").strip()
        chamber = (row.get("position") or "—").strip() or "—"
        ttype = _normalise_type(row.get("transactionType") or "")
        amount_from = _safe_float(row.get("amountFrom"))
        amount_to = _safe_float(row.get("amountTo")) or amount_from
        filing_date = (row.get("filingDate") or "").strip()
        lag = _filing_lag(date, filing_date)
        return _Trade(
            date=date or "—",
            filer=filer,
            chamber=chamber,
            party="—",  # Finnhub endpoint doesn't include party
            state="—",
            type=ttype,
            amount_min=amount_from or 0.0,
            amount_max=amount_to or amount_from or 0.0,
            amount_label=_amount_label(amount_from, amount_to),
            filing_date=filing_date or "—",
            filing_lag_days=lag,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("skipping malformed finnhub row %r: %s", row, e)
        return None


# --- Source: Senate Stock Watcher ------------------------------------------


def _fetch_senate_stock_watcher(ticker: str, cutoff_date) -> List[_Trade]:
    """Fetch the aggregate Senate PTR dataset and filter to ``ticker``."""
    cache_key = {"kind": "ssw_full"}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=24 * 3600)
    if cached is not None:
        rows = _parse_ssw_payload(cached)
    else:
        resp = requests.get(_SENATE_SW_URL, timeout=120)
        resp.raise_for_status()
        body = resp.text
        cache_put(_SOURCE, cache_key, body)
        rows = _parse_ssw_payload(body)

    target = ticker.upper().lstrip("$").strip()
    out: List[_Trade] = []
    for row in rows:
        # Senate Stock Watcher prepends a $ to ticker symbols sometimes
        row_ticker = (row.get("ticker") or "").upper().lstrip("$").strip()
        if row_ticker != target:
            continue
        trade = _parse_ssw_row(row)
        if trade and _within_cutoff(trade, cutoff_date):
            out.append(trade)
    return out


def _parse_ssw_payload(body: str) -> List[dict]:
    import json
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        raise ValueError(f"Senate Stock Watcher returned non-JSON body: {e}") from e
    if isinstance(data, dict) and "transactions" in data:
        return list(data["transactions"])
    if isinstance(data, list):
        return data
    raise ValueError(f"unexpected Senate Stock Watcher payload shape: {type(data).__name__}")


def _parse_ssw_row(row: dict) -> Optional[_Trade]:
    try:
        date = _normalise_ssw_date(row.get("transaction_date") or "")
        filer = (row.get("senator") or row.get("owner") or "—").strip()
        party = _shorten_party(row.get("party") or "—")
        state = (row.get("state") or "—").strip() or "—"
        ttype = _normalise_type(row.get("type") or "")
        amount_min, amount_max = _amount_range_to_floats(row.get("amount") or "")
        filing_date = _normalise_ssw_date(row.get("ptr_link_date") or row.get("filing_date") or "")
        lag = _filing_lag(date, filing_date)
        return _Trade(
            date=date or "—",
            filer=filer,
            chamber="Senate",
            party=party,
            state=state,
            type=ttype,
            amount_min=amount_min,
            amount_max=amount_max,
            amount_label=_amount_label(amount_min, amount_max),
            filing_date=filing_date or "—",
            filing_lag_days=lag,
        )
    except Exception as e:  # noqa: BLE001
        logger.debug("skipping malformed SSW row %r: %s", row, e)
        return None


# --- Shared helpers ---------------------------------------------------------


def _within_cutoff(trade: _Trade, cutoff_date) -> bool:
    raw = trade.get("date") or ""
    if not raw or raw == "—":
        return False
    try:
        d = datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return False
    return d >= cutoff_date


def _normalise_type(raw: str) -> str:
    s = raw.strip().lower()
    if "purchase" in s or s in ("buy", "p"):
        return "Purchase"
    if "sale" in s or "sell" in s or s == "s":
        return "Sale"
    if "exchange" in s:
        return "Exchange"
    return raw.strip().title() or "—"


def _safe_float(v: Any) -> Optional[float]:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _filing_lag(transaction_date: str, filing_date: str) -> Optional[int]:
    try:
        td = datetime.strptime(transaction_date, "%Y-%m-%d").date()
        fd = datetime.strptime(filing_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    return (fd - td).days


def _normalise_ssw_date(raw: str) -> str:
    """Senate Stock Watcher uses MM/DD/YYYY, sometimes ISO. Return YYYY-MM-DD."""
    raw = (raw or "").strip()
    if not raw:
        return ""
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d/%m/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


def _shorten_party(party: str) -> str:
    p = (party or "").strip().lower()
    if p.startswith("rep"):
        return "R"
    if p.startswith("dem"):
        return "D"
    if p.startswith("ind"):
        return "I"
    return party.strip()[:1].upper() if party else "—"


def _amount_label(lo: Optional[float], hi: Optional[float]) -> str:
    if not lo and not hi:
        return "—"
    if lo and hi and lo != hi:
        return f"${_money(lo)} – ${_money(hi)}"
    return f"${_money(lo or hi)}"


def _money(amount: Optional[float]) -> str:
    if amount is None:
        return "0"
    if amount >= 1_000_000:
        return f"{amount / 1_000_000:.1f}M"
    if amount >= 1_000:
        return f"{amount / 1_000:.0f}k"
    return f"{amount:.0f}"


def _amount_range_to_floats(label: str) -> tuple[float, float]:
    """Parse a Senate Stock Watcher amount string like ``"$15,001 - $50,000"``."""
    if not label:
        return 0.0, 0.0
    import re
    nums = re.findall(r"\d[\d,]*", label)
    parsed = [float(n.replace(",", "")) for n in nums]
    if not parsed:
        return 0.0, 0.0
    if len(parsed) == 1:
        return parsed[0], parsed[0]
    return parsed[0], parsed[-1]


# --- Reporting --------------------------------------------------------------


def _format_report(
    ticker: str,
    trades: List[_Trade],
    lookback_days: int,
    source_label: str,
) -> str:
    purchases = [t for t in trades if t["type"] == "Purchase"]
    sales = [t for t in trades if t["type"] == "Sale"]
    unique_buyers = {t["filer"] for t in purchases}
    unique_sellers = {t["filer"] for t in sales}

    bought_lo = sum(t["amount_min"] for t in purchases)
    bought_hi = sum(t["amount_max"] for t in purchases)
    sold_lo = sum(t["amount_min"] for t in sales)
    sold_hi = sum(t["amount_max"] for t in sales)

    lines = [
        f"## Congressional Trade Disclosures for {ticker} — last {lookback_days} days",
        f"_Source: {source_label}_",
        "",
        f"**Sentiment**: {len(unique_buyers)} unique buyer(s) vs "
        f"{len(unique_sellers)} unique seller(s) "
        f"(net {len(unique_buyers) - len(unique_sellers):+d} buyers).",
        f"**Volume (disclosed range)**: ${_money(bought_lo)} – ${_money(bought_hi)} bought, "
        f"${_money(sold_lo)} – ${_money(sold_hi)} sold.",
        "",
        "### Disclosures",
        "| Trade Date | Filer | Chamber/Party/State | Type | Amount | Filing Lag |",
        "|---|---|---|---|---|---|",
    ]
    for t in sorted(trades, key=lambda x: x.get("date", ""), reverse=True):
        loc = "/".join([t.get("chamber", "—") or "—", t.get("party", "—") or "—", t.get("state", "—") or "—"])
        lag = t.get("filing_lag_days")
        lag_str = f"{lag} days" if lag is not None else "—"
        lines.append(
            f"| {t.get('date', '—')} | {t.get('filer', '—')} | {loc} | "
            f"{t.get('type', '—')} | {t.get('amount_label', '—')} | {lag_str} |"
        )

    lines.append("")
    lines.append(
        "_Note: committee assignments are not surfaced by either upstream source. "
        "Cross-reference filer names against committee rosters manually if needed._"
    )
    return "\n".join(lines)
