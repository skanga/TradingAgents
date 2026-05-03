"""Options-flow signals built on yfinance option chains.

``yf.Ticker(ticker).options`` returns expiry dates as strings (YYYY-MM-DD);
``option_chain(exp)`` returns per-expiry calls/puts DataFrames with columns
``strike, lastPrice, volume, openInterest, impliedVolatility``.

Aggregations:
- Put/Call volume + open-interest ratios across the nearest 4 expiries
- Front-month max-pain strike (where most OI expires worthless)
- Call/put walls (largest single-strike OI on each side)
- Unusual flow (any strike where today's volume exceeds 3× its OI)
- ATM implied volatility from the front-month chain

IV Rank uses 30-day annualised realised volatility as the historical baseline
since free-tier yfinance does not expose historical IV per expiry. The rank
is documented as an HV-based proxy, not a true IV/IV percentile.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable, Optional, TypeVar

import numpy as np
import pandas as pd
import yfinance as yf

from tradingagents.dataflows._cache import cache_get, cache_put

logger = logging.getLogger(__name__)

_SOURCE = "options_flow"
_NUM_EXPIRIES = 4
_UNUSUAL_VOL_OI_RATIO = 3.0
_LARGE_PURCHASE_USD = 500_000  # for context only; flagged in the report
_MAX_UNUSUAL_ROWS = 5
_HV_WINDOW_DAYS = 30
_HV_HISTORY_DAYS = 252

_T = TypeVar("_T")


def _yf_retry(func: Callable[[], _T], max_retries: int = 3, base_delay: float = 2.0) -> _T:
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:  # noqa: BLE001
            last_exc = e
            if attempt < max_retries - 1:
                time.sleep(base_delay * (2 ** attempt))
    assert last_exc is not None
    raise last_exc


# --- Public API -------------------------------------------------------------


def get_options_summary(ticker: str) -> str:
    cache_key = {
        "kind": "summary",
        "ticker": ticker.upper(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=4 * 3600)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(ticker)
        expiries = list(_yf_retry(lambda: tk.options))[:_NUM_EXPIRIES]
        if not expiries:
            return f"[Options summary unavailable: no listed expiries for {ticker}.]"

        spot = _spot_price(tk)
        if spot is None:
            return f"[Options summary unavailable: no spot price for {ticker}.]"

        per_expiry = []
        for exp in expiries:
            chain = _yf_retry(lambda e=exp: tk.option_chain(e))
            calls = chain.calls.copy()
            puts = chain.puts.copy()
            per_expiry.append({"exp": exp, "calls": calls, "puts": puts})

        report = _format_summary(ticker, spot, per_expiry)
        cache_put(_SOURCE, cache_key, report)
        return report
    except Exception as e:
        logger.exception("options summary failed for %s", ticker)
        return f"[Options summary unavailable: {e}. Proceed with available data.]"


def get_iv_rank(ticker: str) -> str:
    cache_key = {
        "kind": "ivr",
        "ticker": ticker.upper(),
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=4 * 3600)
    if cached is not None:
        return cached

    try:
        tk = yf.Ticker(ticker)
        expiries = list(_yf_retry(lambda: tk.options))
        if not expiries:
            return f"[IV Rank unavailable: no listed options for {ticker}.]"

        spot = _spot_price(tk)
        if spot is None:
            return f"[IV Rank unavailable: no spot price for {ticker}.]"

        chain = _yf_retry(lambda: tk.option_chain(expiries[0]))
        atm_iv = _atm_iv(chain.calls, chain.puts, spot)
        if atm_iv is None:
            return f"[IV Rank unavailable: no ATM IV available for {ticker}.]"

        hv_series = _historical_vol_series(tk, _HV_HISTORY_DAYS, _HV_WINDOW_DAYS)
        if hv_series is None or hv_series.empty:
            return f"[IV Rank unavailable: insufficient price history for {ticker}.]"

        hv_min = float(hv_series.min())
        hv_max = float(hv_series.max())
        if hv_max <= hv_min:
            return f"[IV Rank unavailable: degenerate HV range for {ticker}.]"

        # Rank current ATM IV against the HV range
        ivr = max(0.0, min(100.0, (atm_iv - hv_min) / (hv_max - hv_min) * 100.0))

        report = _format_ivr_report(ticker, atm_iv, hv_min, hv_max, ivr)
        cache_put(_SOURCE, cache_key, report)
        return report
    except Exception as e:
        logger.exception("IV Rank failed for %s", ticker)
        return f"[IV Rank unavailable: {e}. Proceed with available data.]"


# --- Helpers ---------------------------------------------------------------


def _spot_price(tk: yf.Ticker) -> Optional[float]:
    try:
        fast = tk.fast_info
        price = fast.get("lastPrice") if hasattr(fast, "get") else getattr(fast, "last_price", None)
        if price:
            return float(price)
    except Exception:
        pass
    try:
        hist = _yf_retry(lambda: tk.history(period="5d", auto_adjust=True))
        if hist is None or hist.empty:
            return None
        return float(hist["Close"].dropna().iloc[-1])
    except Exception:
        return None


def _atm_iv(calls: pd.DataFrame, puts: pd.DataFrame, spot: float) -> Optional[float]:
    """Average call+put IV at the strike nearest spot."""
    ivs: list[float] = []
    for df in (calls, puts):
        if df is None or df.empty:
            continue
        d = df.dropna(subset=["strike", "impliedVolatility"]).copy()
        d = d[d["impliedVolatility"] > 0]
        if d.empty:
            continue
        idx = (d["strike"] - spot).abs().idxmin()
        ivs.append(float(d.at[idx, "impliedVolatility"]))
    if not ivs:
        return None
    return sum(ivs) / len(ivs)


def _historical_vol_series(
    tk: yf.Ticker, history_days: int, window: int
) -> Optional[pd.Series]:
    """Trailing rolling annualised realised vol (daily log returns)."""
    period = f"{history_days + window + 14}d"
    hist = _yf_retry(lambda: tk.history(period=period, auto_adjust=True))
    if hist is None or hist.empty:
        return None
    closes = hist["Close"].dropna()
    if len(closes) < window + 5:
        return None
    log_ret = np.log(closes / closes.shift(1)).dropna()
    rolling = log_ret.rolling(window).std(ddof=0) * np.sqrt(252)
    return rolling.dropna().tail(history_days)


def max_pain_strike(calls: pd.DataFrame, puts: pd.DataFrame) -> Optional[float]:
    """Strike where most OI expires worthless (option-writer pain minimum).

    For each candidate strike K* we compute the total intrinsic value paid
    out across all calls and puts at expiry; the writer-friendly strike is
    the K* that minimises that payout.
    """
    calls = calls.dropna(subset=["strike", "openInterest"]) if calls is not None else None
    puts = puts.dropna(subset=["strike", "openInterest"]) if puts is not None else None
    if calls is None or puts is None or calls.empty or puts.empty:
        return None
    strikes = sorted(set(calls["strike"]).union(puts["strike"]))
    if not strikes:
        return None

    call_strikes = calls["strike"].to_numpy()
    call_oi = calls["openInterest"].fillna(0).to_numpy()
    put_strikes = puts["strike"].to_numpy()
    put_oi = puts["openInterest"].fillna(0).to_numpy()

    best_strike, best_pain = None, None
    for k in strikes:
        call_pain = float(np.sum(np.maximum(0.0, k - call_strikes) * call_oi))
        put_pain = float(np.sum(np.maximum(0.0, put_strikes - k) * put_oi))
        total = call_pain + put_pain
        if best_pain is None or total < best_pain:
            best_strike, best_pain = float(k), total
    return best_strike


def aggregate_pc_ratios(per_expiry: list[dict]) -> dict:
    call_vol = put_vol = call_oi = put_oi = 0.0
    for entry in per_expiry:
        c, p = entry["calls"], entry["puts"]
        if c is not None and not c.empty:
            call_vol += float(c["volume"].fillna(0).sum())
            call_oi += float(c["openInterest"].fillna(0).sum())
        if p is not None and not p.empty:
            put_vol += float(p["volume"].fillna(0).sum())
            put_oi += float(p["openInterest"].fillna(0).sum())
    return {
        "call_vol": call_vol,
        "put_vol": put_vol,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "pc_vol": (put_vol / call_vol) if call_vol > 0 else None,
        "pc_oi": (put_oi / call_oi) if call_oi > 0 else None,
    }


def find_walls(df: pd.DataFrame) -> Optional[tuple[float, float]]:
    """Return (strike, OI) of the largest single-strike OI in ``df``."""
    if df is None or df.empty:
        return None
    d = df.dropna(subset=["strike", "openInterest"])
    if d.empty:
        return None
    idx = d["openInterest"].idxmax()
    return float(d.at[idx, "strike"]), float(d.at[idx, "openInterest"])


def find_unusual_flow(per_expiry: list[dict], spot: float) -> list[dict]:
    """Flag strikes where volume > 3× OI; return up to top-N by ratio."""
    out = []
    for entry in per_expiry:
        for side, df in (("C", entry["calls"]), ("P", entry["puts"])):
            if df is None or df.empty:
                continue
            d = df.dropna(subset=["strike", "volume", "openInterest"]).copy()
            d = d[(d["openInterest"] > 0) & (d["volume"] > 0)]
            d["ratio"] = d["volume"] / d["openInterest"]
            unusual = d[d["ratio"] >= _UNUSUAL_VOL_OI_RATIO]
            for _, row in unusual.iterrows():
                out.append({
                    "expiry": entry["exp"],
                    "side": side,
                    "strike": float(row["strike"]),
                    "volume": float(row["volume"]),
                    "open_interest": float(row["openInterest"]),
                    "ratio": float(row["ratio"]),
                    "moneyness": float(row["strike"] / spot - 1),
                })
    out.sort(key=lambda r: r["ratio"], reverse=True)
    return out[:_MAX_UNUSUAL_ROWS]


# --- Reporting --------------------------------------------------------------


def _format_summary(ticker: str, spot: float, per_expiry: list[dict]) -> str:
    pc = aggregate_pc_ratios(per_expiry)
    if pc["pc_vol"] is None and pc["pc_oi"] is None:
        return f"[Options summary unavailable: empty option chains for {ticker}.]"

    pc_vol_label = _pc_label(pc["pc_vol"])
    pc_oi_label = _pc_label(pc["pc_oi"])

    front = per_expiry[0]
    front_atm_iv = _atm_iv(front["calls"], front["puts"], spot)
    front_max_pain = max_pain_strike(front["calls"], front["puts"])
    call_wall = find_walls(front["calls"])
    put_wall = find_walls(front["puts"])
    unusual = find_unusual_flow(per_expiry, spot)

    lines = [
        f"## Options Flow Summary for {ticker} — front {len(per_expiry)} expiries",
        "",
        f"**Spot**: ${spot:,.2f}",
        "",
        f"**Aggregate flow (across {len(per_expiry)} expiries)**:",
        f"- Call volume: {pc['call_vol']:,.0f} | Put volume: {pc['put_vol']:,.0f} | "
        f"P/C volume ratio: {_fmt_ratio(pc['pc_vol'])} ({pc_vol_label})",
        f"- Call OI: {pc['call_oi']:,.0f} | Put OI: {pc['put_oi']:,.0f} | "
        f"P/C OI ratio: {_fmt_ratio(pc['pc_oi'])} ({pc_oi_label})",
        "",
        f"**Front month ({front['exp']})**:",
    ]
    if front_atm_iv is not None:
        lines.append(f"- ATM IV: {front_atm_iv * 100:.1f}%")
    if front_max_pain is not None:
        moneyness = (front_max_pain / spot - 1) * 100
        lines.append(
            f"- Max pain strike: ${front_max_pain:,.2f} ({moneyness:+.2f}% from spot)"
        )
    if call_wall is not None:
        lines.append(f"- Call wall: ${call_wall[0]:,.2f} ({int(call_wall[1]):,} OI)")
    if put_wall is not None:
        lines.append(f"- Put wall: ${put_wall[0]:,.2f} ({int(put_wall[1]):,} OI)")

    if unusual:
        lines.append("")
        lines.append("**Unusual activity** (volume ≥ 3× OI):")
        for u in unusual:
            lines.append(
                f"- {u['expiry']} ${u['strike']:,.2f}{u['side']}: "
                f"vol {int(u['volume']):,} vs OI {int(u['open_interest']):,} "
                f"({u['ratio']:.1f}×, {u['moneyness'] * 100:+.1f}% from spot)"
            )
    else:
        lines.append("")
        lines.append("No unusual flow (no strike with volume ≥ 3× OI).")

    lines.append("")
    lines.append("Source: yfinance option chains")
    return "\n".join(lines)


def _format_ivr_report(
    ticker: str, atm_iv: float, hv_min: float, hv_max: float, ivr: float
) -> str:
    if ivr >= 50:
        interp = "ELEVATED — options are pricing in above-average uncertainty; hedges are expensive but vol-selling has higher premium."
    elif ivr <= 20:
        interp = "COMPLACENCY — options are cheap relative to recent realised vol; cheap to buy hedges, possible vol expansion ahead."
    else:
        interp = "NEUTRAL — IV is mid-range relative to recent realised vol regime."
    return "\n".join([
        f"## Implied Volatility Rank for {ticker}",
        "",
        f"- Current ATM IV (front-month): {atm_iv * 100:.1f}%",
        f"- Trailing {_HV_HISTORY_DAYS}-day realised-vol range "
        f"({_HV_WINDOW_DAYS}-day rolling, annualised): "
        f"{hv_min * 100:.1f}% – {hv_max * 100:.1f}%",
        f"- IV Rank: {ivr:.0f}/100",
        "",
        f"**Interpretation**: {interp}",
        "",
        "_Note: this IVR uses realised volatility as the historical baseline — "
        "free-tier yfinance does not expose historical implied volatility per "
        "expiry. Treat the rank as a relative-uncertainty signal rather than a "
        "true IV percentile._",
    ])


def _fmt_ratio(x: Optional[float]) -> str:
    return f"{x:.2f}" if x is not None else "n/a"


def _pc_label(pc: Optional[float]) -> str:
    if pc is None:
        return "n/a"
    if pc >= 1.5:
        return "EXTREME bearish skew"
    if pc <= 0.5:
        return "EXTREME bullish skew"
    if pc > 1.0:
        return "modest bearish skew"
    if pc < 1.0:
        return "modest bullish skew"
    return "balanced"
