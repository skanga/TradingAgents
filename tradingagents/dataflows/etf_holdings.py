"""ETF holdings adapter via yfinance.

For an ETF ticker, surfaces what the underlying fund actually owns —
asset-class mix, sector weights, top holdings, and a concentration
metric — so the Fundamentals analyst has something meaningful to write
about for fund tickers (where the company-level financials a peer
comparison would produce don't apply).

Data is pulled from ``yfinance.Ticker(symbol).funds_data``:

- ``asset_classes``     — dict, ``{"stockPosition": 0.997, ...}``
- ``sector_weightings`` — dict, ``{"technology": 0.336, ...}``
- ``top_holdings``      — DataFrame indexed by ticker, columns
                          ``Name`` and ``Holding Percent`` (fractions).
- ``fund_overview``     — dict with ``categoryName``, ``family``,
                          ``legalType``.

Returns a Markdown report on success and a bracketed ``"[…]"`` string on
any failure (non-ETF ticker, yfinance hiccup, empty data) — same contract
as every other vendor adapter.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

_SOURCE = "etf_holdings"
_CACHE_TTL_SECONDS = 24 * 3600
_TOP_N_HOLDINGS = 10

# Sector names that snake_case-to-title-case mangles. yfinance uses a few
# without underscores (``realestate``) or with non-obvious capitalisation.
_SECTOR_LABELS: Dict[str, str] = {
    "realestate":             "Real Estate",
    "consumer_cyclical":      "Consumer Cyclical",
    "consumer_defensive":     "Consumer Defensive",
    "basic_materials":        "Basic Materials",
    "communication_services": "Communication Services",
    "financial_services":     "Financial Services",
    "technology":             "Technology",
    "industrials":            "Industrials",
    "healthcare":             "Healthcare",
    "energy":                 "Energy",
    "utilities":              "Utilities",
}


# --- Public API -------------------------------------------------------------


def get_etf_holdings(ticker: str) -> str:
    """Fetch and render the holdings breakdown for an ETF ``ticker``."""
    try:
        ticker_safe = safe_ticker_component(ticker)
    except ValueError as e:
        return f"[ETF holdings unavailable for {ticker!r}: {e}]"

    cache_key = {
        "ticker": ticker_safe.upper(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        snapshot = _fetch(ticker_safe)
    except Exception as e:  # noqa: BLE001
        logger.warning("yfinance funds_data failed for %s: %s", ticker_safe, e)
        return (
            f"[ETF holdings unavailable for {ticker_safe.upper()}: {e}. "
            f"Proceed with available data.]"
        )

    if not _looks_like_etf(snapshot):
        return (
            f"[ETF holdings unavailable for {ticker_safe.upper()}: "
            f"yfinance returned no funds_data — likely not an ETF / mutual "
            f"fund ticker. Proceed with available data.]"
        )

    report = _format_report(ticker_safe.upper(), snapshot)
    cache_put(_SOURCE, cache_key, report)
    return report


# --- Fetch ------------------------------------------------------------------


def _fetch(ticker: str) -> Dict[str, Any]:
    """Pull the four ``funds_data`` blocks into a plain dict.

    Returns whatever yfinance gives us; ``_looks_like_etf`` decides
    afterwards whether we have enough to render. Each accessor is
    wrapped individually so a missing field doesn't sink the others.
    """
    import yfinance as yf  # lazy import keeps test envs light

    fd = yf.Ticker(ticker.upper()).funds_data
    snap: Dict[str, Any] = {
        "asset_classes":     None,
        "sector_weightings": None,
        "top_holdings":      None,
        "fund_overview":     None,
    }
    for key in snap:
        try:
            snap[key] = getattr(fd, key, None)
        except Exception as e:  # noqa: BLE001
            logger.debug("funds_data.%s for %s raised: %s", key, ticker, e)
    return snap


def _looks_like_etf(snap: Dict[str, Any]) -> bool:
    """Heuristic: an ETF has *at least one* of sector weights or top
    holdings. Non-ETF tickers usually return empty dicts / DataFrames."""
    sw = snap.get("sector_weightings")
    th = snap.get("top_holdings")
    has_sectors = isinstance(sw, dict) and len(sw) > 0
    has_holdings = th is not None and getattr(th, "empty", True) is False
    return has_sectors or has_holdings


# --- Formatting -------------------------------------------------------------


def _format_report(ticker: str, snap: Dict[str, Any]) -> str:
    overview = snap.get("fund_overview") or {}
    asset_classes = snap.get("asset_classes") or {}
    sectors = snap.get("sector_weightings") or {}
    holdings = snap.get("top_holdings")

    lines: List[str] = []
    lines.append(f"## ETF Holdings for {ticker}")
    lines.append(
        f"_Source: yfinance funds_data. Retrieved "
        f"{datetime.utcnow().strftime('%Y-%m-%d')}._"
    )

    # --- Fund overview (compact line) ---
    overview_bits: List[str] = []
    if isinstance(overview, dict):
        if overview.get("categoryName"):
            overview_bits.append(f"**Category**: {overview['categoryName']}")
        if overview.get("family"):
            overview_bits.append(f"**Family**: {overview['family']}")
        if overview.get("legalType"):
            overview_bits.append(f"**Legal Type**: {overview['legalType']}")
    if overview_bits:
        lines.append("")
        lines.append(" | ".join(overview_bits))

    # --- Asset classes ---
    asset_rows = _format_asset_classes(asset_classes)
    if asset_rows:
        lines.append("")
        lines.append("### Asset-Class Breakdown")
        lines.append("| Asset Class | Weight |")
        lines.append("|---|---|")
        lines.extend(asset_rows)

    # --- Sector weights ---
    sector_rows = _format_sector_weights(sectors)
    if sector_rows:
        lines.append("")
        lines.append("### Sector Weights")
        lines.append("| Sector | Weight |")
        lines.append("|---|---|")
        lines.extend(sector_rows)

    # --- Top holdings + concentration ---
    holdings_block, top_n_pct = _format_top_holdings(holdings, _TOP_N_HOLDINGS)
    if holdings_block:
        lines.append("")
        lines.append(f"### Top {_TOP_N_HOLDINGS} Holdings")
        lines.extend(holdings_block)
        if top_n_pct is not None:
            lines.append("")
            lines.append(
                f"**Concentration**: top {_TOP_N_HOLDINGS} holdings represent "
                f"**{top_n_pct * 100:.1f}%** of fund AUM."
            )

    if len(lines) <= 2:
        # Only the title + source line — nothing useful actually rendered.
        # Treat as unavailable so downstream agents don't see a blank section.
        return (
            f"[ETF holdings unavailable for {ticker}: yfinance returned "
            f"no usable holdings data. Proceed with available data.]"
        )

    return "\n".join(lines)


def _format_asset_classes(ac: Any) -> List[str]:
    if not isinstance(ac, dict) or not ac:
        return []
    label_map = {
        "stockPosition":       "Stocks",
        "bondPosition":        "Bonds",
        "cashPosition":        "Cash",
        "preferredPosition":   "Preferred",
        "convertiblePosition": "Convertibles",
        "otherPosition":       "Other",
    }
    out: List[str] = []
    # Preserve label_map order; drop any zero positions to keep the table tight.
    for key, label in label_map.items():
        v = ac.get(key)
        if not isinstance(v, (int, float)) or v <= 0:
            continue
        out.append(f"| {label} | {v * 100:.1f}% |")
    return out


def _format_sector_weights(sw: Any) -> List[str]:
    if not isinstance(sw, dict) or not sw:
        return []
    rows: List[Tuple[str, float]] = []
    for key, weight in sw.items():
        if not isinstance(weight, (int, float)) or weight <= 0:
            continue
        rows.append((_sector_label(key), float(weight)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return [f"| {label} | {weight * 100:.2f}% |" for label, weight in rows]


def _sector_label(key: str) -> str:
    return _SECTOR_LABELS.get(key, key.replace("_", " ").title())


def _format_top_holdings(
    holdings: Any, top_n: int
) -> Tuple[List[str], Optional[float]]:
    """Render the top-N holdings table and return (rows, top-N total weight).

    ``holdings`` is a pandas DataFrame from yfinance — indexed by ticker
    symbol with columns ``Name`` and ``Holding Percent`` (fractions).
    Returns ``([], None)`` when the DataFrame is missing/empty so the
    caller can skip the section.
    """
    if holdings is None:
        return [], None
    try:
        head = holdings.head(top_n)
        if head.empty:
            return [], None
    except AttributeError:
        return [], None

    rows = ["| # | Symbol | Name | Weight |", "|---|---|---|---|"]
    total_weight = 0.0
    for i, (sym, row) in enumerate(head.iterrows(), start=1):
        name = str(row.get("Name", "—"))
        pct = row.get("Holding Percent")
        if isinstance(pct, (int, float)):
            total_weight += float(pct)
            cell = f"{pct * 100:.2f}%"
        else:
            cell = "—"
        rows.append(f"| {i} | {sym} | {name} | {cell} |")
    return rows, total_weight if total_weight > 0 else None
