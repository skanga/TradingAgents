"""Sector relative strength and inter-market correlations via yfinance.

- Sector mapping uses the SPDR Select Sector ETFs (XLK/XLF/XLE/...) as proxies
  for the broad sector return. Yahoo's ``info['sector']`` value is the lookup
  key.
- Inter-market basket: GLD (gold), USO (oil), BTC-USD (crypto), UUP (USD proxy
  for DXY), ^VIX. Correlations are Pearson on daily simple returns over the
  lookback window.

All numeric work is done with pandas; daily reports are cached to disk for
12 hours so repeated agent calls inside one analysis don't redownload.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Optional, TypeVar

import pandas as pd
import yfinance as yf

from tradingagents.dataflows._cache import cache_get, cache_put

_T = TypeVar("_T")


def _yf_retry(func: Callable[[], _T], max_retries: int = 3, base_delay: float = 2.0) -> _T:
    """Tiny retry helper for yfinance calls — avoids importing stockstats."""
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:  # noqa: BLE001 — yfinance raises a wide variety
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    assert last_exc is not None
    raise last_exc

logger = logging.getLogger(__name__)

_SOURCE = "sector_analysis"

# Yahoo sector name → SPDR sector ETF ticker.
SECTOR_ETF: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Industrials": "XLI",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Basic Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

_BENCHMARK = "SPY"
_INTERMARKET = {
    "GLD": "gold",
    "USO": "oil",
    "BTC-USD": "crypto",
    "UUP": "USD",
    "^VIX": "VIX",
}
_NOTABLE_CORR = 0.5


# --- Public API -------------------------------------------------------------


def get_sector_relative_strength(ticker: str, lookback_days: int = 63) -> str:
    """Sector ETF return vs SPY, and ``ticker`` vs its sector ETF, over ``lookback_days``."""
    cache_key = {
        "kind": "sector_rs",
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=12 * 3600)
    if cached is not None:
        return cached

    try:
        sector = _resolve_sector(ticker)
        if sector is None or sector not in SECTOR_ETF:
            label = sector or "unknown"
            return (
                f"[Sector RS unavailable: no SPDR sector ETF mapping for "
                f"{ticker} (Yahoo sector = {label!r}). "
                "Proceed with available data.]"
            )
        sector_etf = SECTOR_ETF[sector]

        prices = _download_close([ticker, sector_etf, _BENCHMARK], lookback_days)
        if prices is None or prices.empty:
            return f"[Sector RS unavailable: no price data for {ticker}. Proceed with available data.]"

        ticker_ret = _total_return(prices[ticker])
        sector_ret = _total_return(prices[sector_etf])
        spy_ret = _total_return(prices[_BENCHMARK])
        if ticker_ret is None or sector_ret is None or spy_ret is None:
            return f"[Sector RS unavailable: insufficient overlapping prices for {ticker}.]"

        report = _format_rs_report(
            ticker=ticker,
            sector=sector,
            sector_etf=sector_etf,
            lookback_days=lookback_days,
            ticker_ret=ticker_ret,
            sector_ret=sector_ret,
            spy_ret=spy_ret,
        )
        cache_put(_SOURCE, cache_key, report)
        return report
    except Exception as e:
        logger.exception("sector RS failed for %s", ticker)
        return f"[Sector RS unavailable: {e}. Proceed with available data.]"


def get_intermarket_correlations(ticker: str, lookback_days: int = 63) -> str:
    """Pearson correlations between ``ticker`` daily returns and the inter-market basket."""
    cache_key = {
        "kind": "intermarket",
        "ticker": ticker.upper(),
        "lookback_days": lookback_days,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=12 * 3600)
    if cached is not None:
        return cached

    try:
        symbols = [ticker] + list(_INTERMARKET.keys())
        prices = _download_close(symbols, lookback_days)
        if prices is None or prices.empty or ticker not in prices.columns:
            return f"[Inter-market correlations unavailable: no price data for {ticker}.]"

        returns = prices.pct_change(fill_method=None).dropna(how="all")
        ticker_ret = returns[ticker].dropna()
        if len(ticker_ret) < 10:
            return (
                f"[Inter-market correlations unavailable: only {len(ticker_ret)} "
                f"overlapping days for {ticker}. Proceed with available data.]"
            )

        correlations: dict[str, tuple[str, Optional[float]]] = {}
        for sym, label in _INTERMARKET.items():
            if sym not in returns.columns:
                correlations[sym] = (label, None)
                continue
            joined = pd.concat([ticker_ret, returns[sym]], axis=1, join="inner").dropna()
            if len(joined) < 10:
                correlations[sym] = (label, None)
                continue
            corr = joined.iloc[:, 0].corr(joined.iloc[:, 1])
            correlations[sym] = (label, None if pd.isna(corr) else float(corr))

        report = _format_corr_report(ticker, lookback_days, correlations)
        cache_put(_SOURCE, cache_key, report)
        return report
    except Exception as e:
        logger.exception("inter-market correlations failed for %s", ticker)
        return f"[Inter-market correlations unavailable: {e}. Proceed with available data.]"


# --- Helpers ---------------------------------------------------------------


def _resolve_sector(ticker: str) -> Optional[str]:
    """Return the Yahoo ``sector`` string for ``ticker`` or None if missing."""
    cache_key = {"kind": "sector_lookup", "ticker": ticker.upper()}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=7 * 24 * 3600)
    if cached is not None:
        return cached or None  # cached empty string → None

    try:
        info = _yf_retry(lambda: yf.Ticker(ticker).get_info())
    except Exception as e:
        logger.warning("sector lookup failed for %s: %s", ticker, e)
        return None
    sector = (info or {}).get("sector") or ""
    cache_put(_SOURCE, cache_key, sector)
    return sector or None


def _download_close(symbols: list[str], lookback_days: int) -> Optional[pd.DataFrame]:
    """Download adjusted close prices for ``symbols`` over the recent window.

    Adds a buffer so the requested window's worth of trading days is actually
    present even after weekends/holidays trim the calendar window.
    """
    period_days = lookback_days + 14  # weekend + holiday buffer
    period = f"{period_days}d"
    try:
        data = _yf_retry(lambda: yf.download(
            tickers=symbols,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=False,
        ))
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", symbols, e)
        return None

    if data is None or data.empty:
        return None

    # Multi-symbol download returns a column-MultiIndex (field, symbol);
    # single-symbol returns a flat-column frame.
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"] if "Close" in data.columns.get_level_values(0) else data
    else:
        close = data[["Close"]].rename(columns={"Close": symbols[0]})

    return close.dropna(how="all").tail(lookback_days)


def _total_return(prices: pd.Series) -> Optional[float]:
    s = prices.dropna()
    if len(s) < 2:
        return None
    return float(s.iloc[-1] / s.iloc[0] - 1.0)


# --- Reporting --------------------------------------------------------------


def _format_rs_report(
    *,
    ticker: str,
    sector: str,
    sector_etf: str,
    lookback_days: int,
    ticker_ret: float,
    sector_ret: float,
    spy_ret: float,
) -> str:
    rs_ratio = (1 + sector_ret) / (1 + spy_ret) if spy_ret > -1.0 else None
    sector_alpha = sector_ret - spy_ret
    stock_alpha = ticker_ret - sector_ret

    if sector_alpha > 0.005:
        regime = "TAILWIND (sector outperforming the broad market)"
    elif sector_alpha < -0.005:
        regime = "HEADWIND (sector underperforming the broad market)"
    else:
        regime = "NEUTRAL (sector tracking the broad market)"

    if stock_alpha > 0.005:
        leadership = f"leading its sector (+{stock_alpha * 100:.2f}% vs {sector_etf})"
    elif stock_alpha < -0.005:
        leadership = f"lagging its sector ({stock_alpha * 100:.2f}% vs {sector_etf})"
    else:
        leadership = f"tracking its sector (≈{stock_alpha * 100:+.2f}% vs {sector_etf})"

    rs_line = (
        f"- Sector RS ({sector_etf}/{_BENCHMARK}): {rs_ratio:.3f}"
        if rs_ratio is not None
        else f"- Sector RS ({sector_etf}/{_BENCHMARK}): undefined"
    )

    return "\n".join([
        f"## Sector Relative Strength for {ticker} — last {lookback_days} trading days",
        "",
        f"**Sector**: {sector} (proxy: {sector_etf})",
        f"- {sector_etf} total return: {sector_ret * 100:+.2f}%",
        f"- {_BENCHMARK} total return: {spy_ret * 100:+.2f}%",
        rs_line,
        f"- Sector vs SPY alpha: {sector_alpha * 100:+.2f}% → **{regime}**",
        f"- {ticker} vs {sector_etf}: {ticker_ret * 100:+.2f}% vs {sector_ret * 100:+.2f}% → "
        f"{stock_alpha * 100:+.2f}% stock-specific alpha — {ticker} is {leadership}.",
        "",
        "Source: yfinance (SPDR Select Sector ETFs)",
    ])


def _format_corr_report(
    ticker: str,
    lookback_days: int,
    correlations: dict[str, tuple[str, Optional[float]]],
) -> str:
    lines = [
        f"## Inter-Market Correlations for {ticker} — last {lookback_days} trading days",
        "",
        "| Asset | Label | Correlation |",
        "|---|---|---|",
    ]
    notable = []
    for sym, (label, corr) in correlations.items():
        if corr is None:
            lines.append(f"| {sym} | {label} | n/a |")
            continue
        lines.append(f"| {sym} | {label} | {corr:+.2f} |")
        if abs(corr) >= _NOTABLE_CORR:
            notable.append((sym, label, corr))

    if notable:
        lines.append("")
        lines.append("**Notable sensitivities**:")
        for sym, label, corr in notable:
            direction = "positive" if corr > 0 else "negative"
            lines.append(
                f"- {sym} ({label}, {corr:+.2f}): strong {direction} correlation."
            )
    else:
        lines.append("")
        lines.append("No basket assets exceed |0.5| correlation in this window.")

    lines.append("")
    lines.append("Source: yfinance")
    return "\n".join(lines)
