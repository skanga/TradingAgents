"""FRED-backed macroeconomic snapshot.

Endpoint: ``GET https://api.stlouisfed.org/fred/series/observations
?series_id={ID}&api_key={KEY}&file_type=json``

Series of interest:
- ``DGS2``         — 2Y Treasury yield (%, daily)
- ``DGS10``        — 10Y Treasury yield (%, daily)
- ``T10Y2Y``       — 10Y-2Y spread (pp, daily) — inversion = recession risk
- ``DFF``          — Fed Funds effective rate (%, daily)
- ``BAMLH0A0HYM2`` — High-yield credit spread (%, daily) — risk-on/off
- ``DTWEXBGS``     — Broad USD index (level, daily)

(The original spec listed ``DPCREDIT`` for Fed Funds; that series is the
discount-window primary credit rate, not the Fed Funds effective rate. ``DFF``
is the canonical Fed Funds effective series.)

The backdrop classifier scores each signal and aggregates to one of
FAVORABLE / NEUTRAL / UNFAVORABLE for equities. Scoring is intentionally
simple — its job is to give the agent a clear summary, not to replace
proper macro analysis.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from typing import Optional, TypedDict

import requests

from tradingagents.dataflows._cache import cache_get, cache_put
from tradingagents.dataflows.config import get_config

logger = logging.getLogger(__name__)

_SOURCE = "macro_data"
_FRED_URL = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = 30


class _SeriesSnapshot(TypedDict):
    series_id: str
    label: str
    units: str
    current: float
    current_date: str
    prior: Optional[float]
    delta: Optional[float]      # current - prior in raw units
    pct_change: Optional[float] # (current/prior - 1), only meaningful for level series
    score: int                  # -1 / 0 / +1 contribution to the backdrop rating
    interpretation: str


# Ordered so the report reads top-down: rates → curve → policy → credit → FX.
_SERIES: list[tuple[str, str, str]] = [
    ("DGS2",         "2Y Treasury",          "%"),
    ("DGS10",        "10Y Treasury",         "%"),
    ("T10Y2Y",       "10Y-2Y Spread",        "pp"),
    ("DFF",          "Fed Funds Eff",        "%"),
    ("BAMLH0A0HYM2", "HY Credit Spread",     "%"),
    ("DTWEXBGS",     "Broad USD Index",      "level"),
]

# Series where percent change matters more than raw delta (level series).
_PCT_CHANGE_SERIES = {"DTWEXBGS"}


# --- Public API -------------------------------------------------------------


def get_macro_environment(lookback_days: int = 30) -> str:
    cache_key = {
        "kind": "report",
        "lookback_days": lookback_days,
        "date": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=6 * 3600)
    if cached is not None:
        return cached

    fred_key = (get_config().get("fred_api_key") or "").strip()
    if not fred_key:
        return (
            "[Macro environment unavailable: FRED_API_KEY not configured. "
            "Register a free key at fred.stlouisfed.org and set FRED_API_KEY. "
            "Proceed with available data.]"
        )

    snapshots: list[_SeriesSnapshot] = []
    failures: list[str] = []
    for series_id, label, units in _SERIES:
        try:
            snap = _build_snapshot(series_id, label, units, fred_key, lookback_days)
            if snap is not None:
                snapshots.append(snap)
        except Exception as e:
            logger.warning("FRED fetch failed for %s: %s", series_id, e)
            failures.append(f"{series_id} ({e})")

    if not snapshots:
        return (
            "[Macro environment unavailable: every FRED series fetch failed "
            f"({', '.join(failures) if failures else 'unknown error'}). "
            "Proceed with available data.]"
        )

    report = _format_report(snapshots, lookback_days, partial=bool(failures))
    cache_put(_SOURCE, cache_key, report)
    return report


# --- Series fetch + snapshot construction ----------------------------------


def _build_snapshot(
    series_id: str,
    label: str,
    units: str,
    api_key: str,
    lookback_days: int,
) -> Optional[_SeriesSnapshot]:
    obs = _fetch_series(series_id, api_key, lookback_days + 14)
    if not obs:
        return None
    obs.sort(key=lambda r: r[0])  # oldest → newest

    current_date, current = obs[-1]
    prior = _value_n_days_ago(obs, lookback_days)
    delta = (current - prior) if prior is not None else None
    pct_change = (current / prior - 1.0) if (prior and prior != 0) else None

    score, interpretation = _interpret(series_id, current, delta, pct_change)
    return _SeriesSnapshot(
        series_id=series_id,
        label=label,
        units=units,
        current=current,
        current_date=current_date.isoformat(),
        prior=prior,
        delta=delta,
        pct_change=pct_change,
        score=score,
        interpretation=interpretation,
    )


def _fetch_series(series_id: str, api_key: str, recent_n: int) -> list[tuple[date, float]]:
    cache_key = {"kind": "series", "series_id": series_id}
    cached = cache_get(_SOURCE, cache_key, ttl_seconds=6 * 3600)
    if cached is not None:
        return _parse_observations(json.loads(cached))

    resp = requests.get(
        _FRED_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": recent_n,
        },
        timeout=_TIMEOUT,
    )
    resp.raise_for_status()
    body = resp.json()
    cache_put(_SOURCE, cache_key, json.dumps(body))
    return _parse_observations(body)


def _parse_observations(body: dict) -> list[tuple[date, float]]:
    """Filter FRED's '.' missing-value sentinel and parse the rest."""
    out: list[tuple[date, float]] = []
    for row in body.get("observations") or []:
        raw = (row.get("value") or "").strip()
        if raw in ("", "."):
            continue
        try:
            value = float(raw)
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
        except (KeyError, ValueError):
            continue
        out.append((d, value))
    return out


def _value_n_days_ago(
    obs: list[tuple[date, float]], n_days: int
) -> Optional[float]:
    """Return the observation closest to ``n_days`` before the latest point."""
    if not obs:
        return None
    target = obs[-1][0] - timedelta(days=n_days)
    # obs is sorted ascending; pick the largest date <= target
    candidate = None
    for d, v in obs:
        if d <= target:
            candidate = v
        else:
            break
    if candidate is None:
        # Series too short — fall back to the oldest value we have
        return obs[0][1]
    return candidate


# --- Per-series interpretation + scoring -----------------------------------


def _interpret(
    series_id: str,
    current: float,
    delta: Optional[float],
    pct_change: Optional[float],
) -> tuple[int, str]:
    """Return (score, short interpretation string).

    Score convention: +1 = supportive of equities, -1 = headwind, 0 = neutral.
    """
    if series_id == "T10Y2Y":
        # Curve inversion is the headline recession risk signal.
        if current < -0.10:
            return -1, "deeply inverted (recession risk)"
        if current < 0:
            return -1, "mildly inverted"
        if current > 0.50:
            return 1, "comfortably positive"
        return 0, "near-flat (transitional)"

    if series_id == "BAMLH0A0HYM2":
        # Tightening = risk-on; widening = risk-off.
        if delta is None:
            return 0, "no trend data"
        if delta < -0.10:
            return 1, "tightening (risk-on)"
        if delta > 0.30:
            return -1, "widening sharply (risk-off)"
        if delta > 0.10:
            return 0, "drifting wider (modest risk-off)"
        return 0, "stable"

    if series_id in ("DGS10", "DGS2"):
        # Falling long-end yields are typically equity-supportive at the
        # margin (lower discount rate); rising yields are a headwind for
        # duration-sensitive sectors. Threshold of 25 bp over the window.
        if delta is None:
            return 0, "no trend data"
        if delta < -0.25:
            return 1, "falling (eq-supportive)"
        if delta > 0.25:
            return -1, "rising (eq-headwind)"
        return 0, "stable"

    if series_id == "DFF":
        # The level moves only on FOMC decisions; trend over 30d usually 0.
        if delta is None or abs(delta) < 0.05:
            return 0, "stable"
        return (1 if delta < 0 else -1), ("cutting" if delta < 0 else "tightening")

    if series_id == "DTWEXBGS":
        # Strong USD pressures multinationals and commodities; weak USD helps.
        if pct_change is None:
            return 0, "no trend data"
        if pct_change < -0.01:
            return 1, "softening (commodity / multinational tailwind)"
        if pct_change > 0.01:
            return -1, "strengthening (multinational headwind)"
        return 0, "stable"

    return 0, "—"


def _classify_backdrop(score_total: int) -> str:
    if score_total >= 2:
        return "FAVORABLE"
    if score_total <= -2:
        return "UNFAVORABLE"
    return "NEUTRAL"


# --- Reporting --------------------------------------------------------------


def _format_report(
    snapshots: list[_SeriesSnapshot], lookback_days: int, partial: bool
) -> str:
    backdrop = _classify_backdrop(sum(s["score"] for s in snapshots))
    as_of = max(s["current_date"] for s in snapshots)
    lines = [
        f"## Macroeconomic Environment — as of {as_of}",
        "",
        f"**Backdrop**: {backdrop} for equities",
    ]
    if partial:
        lines.append("_(partial — one or more FRED series failed; see logs)_")
    lines += [
        "",
        f"| Indicator | Current | {lookback_days}d Δ | Signal |",
        "|---|---|---|---|",
    ]
    for s in snapshots:
        current_str = _fmt_value(s["current"], s["units"])
        delta_str = _fmt_delta(s, lookback_days)
        lines.append(
            f"| {s['label']} ({s['series_id']}) | {current_str} | {delta_str} | "
            f"{_score_glyph(s['score'])} {s['interpretation']} |"
        )
    lines += [
        "",
        "**Reasoning**:",
    ]
    for s in snapshots:
        lines.append(f"- {s['label']}: {s['interpretation']}")
    lines += [
        "",
        "Source: FRED (Federal Reserve Bank of St. Louis)",
    ]
    return "\n".join(lines)


def _fmt_value(value: float, units: str) -> str:
    if units == "%":
        return f"{value:.2f}%"
    if units == "pp":
        return f"{value:+.2f}"
    return f"{value:,.2f}"


def _fmt_delta(s: _SeriesSnapshot, lookback_days: int) -> str:
    if s["series_id"] in _PCT_CHANGE_SERIES:
        if s["pct_change"] is None:
            return "n/a"
        return f"{s['pct_change'] * 100:+.2f}%"
    if s["delta"] is None:
        return "n/a"
    if s["units"] == "%":
        return f"{s['delta']:+.2f} pp"
    return f"{s['delta']:+.2f}"


def _score_glyph(score: int) -> str:
    if score > 0:
        return "✓"
    if score < 0:
        return "✗"
    return "•"
