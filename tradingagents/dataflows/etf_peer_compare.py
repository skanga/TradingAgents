"""ETF peer-comparison adapter via yfinance.

Compares a primary ETF against 2-6 peer ETFs on a fixed metric set:

- **Profile**: category, AUM, expense ratio, distribution yield,
  inception year, 3-year beta. Pulled from ``Ticker.info``.
- **Returns**: 1-month, 3-month, year-to-date, 1-year. Computed from
  ``Ticker.history(period='1y', auto_adjust=True)`` so dividends are
  baked into the close prices (i.e., total return).
- **Risk**: 1-year annualized volatility, 1-year max drawdown.
  Computed from the same daily-close series.

The metric set is fixed (no ``metrics`` parameter) because picking the
right metrics for ETFs is much less situational than for companies —
analysts virtually always want returns + risk + cost.

Returns a Markdown comparison table on success, a bracketed
``"[…]"`` string on any failure (auth not needed for yfinance, but
network errors, non-ETF tickers, and empty histories all produce a
graceful fallback).
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.utils import safe_ticker_component

logger = logging.getLogger(__name__)

_SOURCE = "etf_peer_compare"

# ETF prices change daily; cache on the day key so a multi-ticker run
# pulls each peer set at most once per day.
_CACHE_TTL_SECONDS = 6 * 3600

# Cap peers so the rendered table stays readable.
_MAX_PEERS = 6

# Trading days per window for return calculations. 252 ≈ trading days/year.
_TRADING_DAYS = {
    "ret_1m": 21,
    "ret_3m": 63,
    "ret_1y": 252,
}


# --- Public API -------------------------------------------------------------


def get_etf_peer_comparison(ticker: str, peers: str) -> str:
    """Compare ``ticker`` against ``peers`` (comma-separated ETFs) on a
    fixed profile + returns + risk metric set."""
    try:
        primary = safe_ticker_component(ticker)
    except ValueError as e:
        return f"[ETF peer comparison unavailable for {ticker!r}: {e}]"

    primary_upper = primary.upper()
    try:
        peer_list = _parse_peers(peers, exclude=primary_upper)
    except ValueError as e:
        return f"[ETF peer comparison unavailable for {primary_upper}: {e}]"
    if not peer_list:
        return (
            f"[ETF peer comparison unavailable for {primary_upper}: "
            f"no peers supplied. Pass a comma-separated list (e.g. "
            f"'QQQ,IWM,DIA').]"
        )

    requested = [primary_upper, *peer_list]
    cache_key = {
        "tickers": ",".join(requested),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=_CACHE_TTL_SECONDS)
    if cached is not None:
        return cached

    try:
        rows = _fetch_metrics(requested)
    except Exception as e:  # noqa: BLE001
        logger.warning("ETF peer compare failed for %s: %s", requested, e)
        return (
            f"[ETF peer comparison unavailable for {primary_upper}: {e}. "
            f"Proceed with available data.]"
        )

    if not any(rows.values()):
        return (
            f"[ETF peer comparison unavailable for {primary_upper}: "
            f"yfinance returned no data for any of {', '.join(requested)}.]"
        )

    report = _format_table(primary_upper, requested, rows)
    cache_put(_SOURCE, cache_key, report)
    return report


# --- Input validation -------------------------------------------------------


def _parse_peers(raw: Any, *, exclude: str) -> List[str]:
    """Split a comma-separated peer list, validate each ticker, dedupe,
    drop the primary if the LLM accidentally re-listed it."""
    if not isinstance(raw, str):
        raise ValueError("peers must be a comma-separated string")
    seen: set[str] = {exclude}
    out: List[str] = []
    for chunk in raw.split(","):
        sym = chunk.strip().upper()
        if not sym or sym in seen:
            continue
        # Raises ValueError for unsafe components — propagate.
        safe_ticker_component(sym)
        out.append(sym)
        seen.add(sym)
        if len(out) >= _MAX_PEERS:
            break
    return out


# --- Data fetching ----------------------------------------------------------


def _fetch_metrics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """Pull profile + price-derived metrics for each ticker.

    Returns ``{TICKER: {metric_key: value}}``. Per-ticker failures are
    swallowed (the ticker just gets a partial dict) so one missing ETF
    doesn't sink the whole comparison.
    """
    import yfinance as yf  # lazy import keeps test envs lighter

    # Batch download is one HTTP request per N tickers vs N requests with
    # sequential calls. ``group_by='ticker'`` makes per-ticker slicing easy.
    # ``15mo`` gives ~315 trading days — a comfortable buffer for the 252-
    # trading-day 1Y lookback (period='1y' would return ~251 days, exactly
    # one short, leaving ret_1y NA on every ETF).
    hist = yf.download(
        tickers,
        period="15mo",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    out: Dict[str, Dict[str, Any]] = {}
    for sym in tickers:
        metrics: Dict[str, Any] = {}

        close = _per_ticker_close(hist, sym, len(tickers))
        if close is not None and not close.empty:
            metrics.update(_compute_price_metrics(close))

        try:
            info = yf.Ticker(sym).info or {}
        except Exception as e:  # noqa: BLE001
            logger.debug("yfinance info for %s failed: %s", sym, e)
            info = {}
        metrics.update(_extract_info_metrics(info))

        out[sym] = metrics
    return out


def _per_ticker_close(hist: Any, sym: str, n_tickers: int) -> Optional[pd.Series]:
    """Slice the per-ticker Close series out of yf.download's output.

    yf.download returns a flat DataFrame for one ticker and a multi-level-
    column DataFrame (tickers as outer level) for many. This helper
    handles both shapes and any missing-ticker edge cases.
    """
    if hist is None or hist.empty:
        return None
    try:
        if n_tickers == 1:
            return hist["Close"].dropna() if "Close" in hist.columns else None
        # multi-ticker: hist[(sym, 'Close')] or hist[sym]['Close']
        if isinstance(hist.columns, pd.MultiIndex):
            level0 = hist.columns.get_level_values(0).unique()
            if sym not in level0:
                return None
            sym_block = hist[sym]
            if "Close" not in sym_block.columns:
                return None
            return sym_block["Close"].dropna()
    except (KeyError, AttributeError) as e:
        logger.debug("Slicing close for %s failed: %s", sym, e)
        return None
    return None


def _compute_price_metrics(close: pd.Series) -> Dict[str, Any]:
    """Returns over 1M/3M/YTD/1Y, annualized vol, max drawdown."""
    if close.empty:
        return {}
    out: Dict[str, Any] = {}
    last = float(close.iloc[-1])

    for label, n in _TRADING_DAYS.items():
        if len(close) > n:
            base = float(close.iloc[-n - 1])
            if base > 0:
                out[label] = (last / base) - 1.0

    # YTD: from January 1 of the current year, in the price series' tz.
    current_year_start = pd.Timestamp(
        f"{datetime.utcnow().year}-01-01", tz=close.index.tz
    )
    ytd_window = close[close.index >= current_year_start]
    if not ytd_window.empty:
        ytd_base = float(ytd_window.iloc[0])
        if ytd_base > 0:
            out["ret_ytd"] = (last / ytd_base) - 1.0

    # Restrict the 1Y volatility / drawdown windows to the last 252 trading
    # days so the column labels stay honest. We fetch ~315 days of history
    # for the 1Y *return* lookback, but vol_1y / max_dd_1y should reflect
    # exactly one trading year, not the full fetch window.
    one_year = close.iloc[-252:] if len(close) >= 252 else close
    daily = one_year.pct_change().dropna()
    if len(daily) > 1:
        out["vol_1y"] = float(daily.std() * (252 ** 0.5))

    # Max drawdown over the same 1Y window. Cumulative product on daily
    # simple returns gives a normalized equity curve starting at 1; then
    # drawdown = (curve - running_max) / running_max.
    if len(daily) > 0:
        cum = (1 + daily).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        if not drawdown.empty:
            out["max_dd_1y"] = float(drawdown.min())

    return out


def _extract_info_metrics(info: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the profile fields we display from yfinance's info dict."""
    metrics: Dict[str, Any] = {}

    aum = info.get("totalAssets")
    if isinstance(aum, (int, float)) and aum > 0:
        metrics["aum"] = float(aum)

    # yfinance quirk: ``yield`` is a fraction (0.0114 = 1.14%) but
    # ``netExpenseRatio`` is already a percent (0.0945 = 0.0945%).
    yld = info.get("yield")
    if isinstance(yld, (int, float)):
        metrics["yield"] = float(yld)

    er = info.get("netExpenseRatio")
    if isinstance(er, (int, float)):
        metrics["expense_ratio"] = float(er)

    inception = info.get("fundInceptionDate")
    if isinstance(inception, (int, float)):
        try:
            metrics["inception_year"] = datetime.utcfromtimestamp(inception).year
        except (ValueError, OSError):
            pass

    cat = info.get("category")
    if isinstance(cat, str) and cat:
        metrics["category"] = cat

    beta = info.get("beta3Year")
    if isinstance(beta, (int, float)):
        metrics["beta_3y"] = float(beta)

    return metrics


# --- Formatting -------------------------------------------------------------


def _format_table(
    primary: str, requested: List[str], rows: Dict[str, Dict[str, Any]]
) -> str:
    """Render the comparison as a Markdown table with ETFs as columns."""
    peer_label = ", ".join(t for t in requested if t != primary)
    lines: List[str] = [
        f"## ETF Peer Comparison — {primary} vs {peer_label}",
        f"_Source: yfinance (prices + info). Retrieved "
        f"{datetime.utcnow().strftime('%Y-%m-%d')}._",
        "",
    ]
    header = "| Metric | " + " | ".join(requested) + " |"
    sep = "|---|" + "|".join(["---"] * len(requested)) + "|"
    lines.append(header)
    lines.append(sep)

    for label, key, formatter in _METRIC_ROWS:
        cells = [formatter(rows.get(t, {}).get(key)) for t in requested]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines.append("")
    missing = [t for t in requested if not rows.get(t)]
    if missing:
        lines.append(
            f"_Note: yfinance returned no data for {', '.join(missing)} "
            f"(rendered as em-dashes above)._"
        )
    else:
        lines.append(f"_All {len(requested)} ETFs had data._")
    return "\n".join(lines)


# --- Cell formatters --------------------------------------------------------


def _fmt_text(v: Any) -> str:
    if v is None or v == "":
        return "—"
    return str(v)


def _fmt_aum(v: Any) -> str:
    if not isinstance(v, (int, float)) or v <= 0:
        return "—"
    if v >= 1_000_000_000_000:
        return f"${v / 1_000_000_000_000:.2f}T"
    if v >= 1_000_000_000:
        return f"${v / 1_000_000_000:.2f}B"
    if v >= 1_000_000:
        return f"${v / 1_000_000:.2f}M"
    return f"${v:,.0f}"


def _fmt_expense_ratio(v: Any) -> str:
    """yfinance's netExpenseRatio is already in percent units (0.0945 == 0.0945%)."""
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.2f}%"


def _fmt_yield(v: Any) -> str:
    """yfinance's yield is a fraction (0.0114 == 1.14%)."""
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v * 100:.2f}%"


def _fmt_return(v: Any) -> str:
    """Signed percentage with explicit sign so positive/negative reads at a glance."""
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v * 100:+.2f}%"


def _fmt_pct(v: Any) -> str:
    """Unsigned percentage for vol / drawdown rows (drawdown comes through
    negative; surface its sign so an analyst can't misread a -25% drop)."""
    if not isinstance(v, (int, float)):
        return "—"
    if v < 0:
        return f"{v * 100:.2f}%"
    return f"{v * 100:.2f}%"


def _fmt_beta(v: Any) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.2f}"


def _fmt_year(v: Any) -> str:
    if not isinstance(v, int) or v <= 0:
        return "—"
    return str(v)


# Display order for the rendered table. Each entry is
# (row_label, metrics_key, cell_formatter).
_METRIC_ROWS: List[Tuple[str, str, Callable[[Any], str]]] = [
    ("Category",              "category",       _fmt_text),
    ("AUM",                   "aum",            _fmt_aum),
    ("Expense Ratio",         "expense_ratio",  _fmt_expense_ratio),
    ("Distribution Yield",    "yield",          _fmt_yield),
    ("Inception",             "inception_year", _fmt_year),
    ("Beta (3Y)",             "beta_3y",        _fmt_beta),
    ("Return — 1M",           "ret_1m",         _fmt_return),
    ("Return — 3M",           "ret_3m",         _fmt_return),
    ("Return — YTD",          "ret_ytd",        _fmt_return),
    ("Return — 1Y",           "ret_1y",         _fmt_return),
    ("Volatility (1Y, ann.)", "vol_1y",         _fmt_pct),
    ("Max Drawdown (1Y)",     "max_dd_1y",      _fmt_pct),
]
