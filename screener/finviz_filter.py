"""Finviz screener wrapper with on-disk TTL cache."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def get_candidates(filters: dict, use_cache: bool = True) -> list[str]:
    """Return tickers matching ``filters``, hitting Finviz unless a fresh cache exists.

    On any Finviz error, falls back to the cached list (even if expired);
    if no cache exists, returns an empty list.
    """
    from config import CONFIG  # late import — keeps this module side-effect-free at import
    cache_path = Path(CONFIG["cache_path"])
    ttl_seconds = CONFIG["cache_ttl_hours"] * 3600

    if use_cache:
        cached = _load_cache(cache_path, max_age_seconds=ttl_seconds)
        if cached is not None:
            logger.info("Finviz cache HIT — returning %d cached tickers", len(cached))
            return cached
        logger.info("Finviz cache MISS — querying Finviz")

    try:
        from finvizfinance.screener.overview import Overview
        screener = Overview()
        screener.set_filter(filters_dict=filters)
        df = screener.screener_view()
        tickers = (
            df["Ticker"].astype(str).tolist()
            if df is not None and "Ticker" in getattr(df, "columns", [])
            else []
        )
        logger.info("Finviz returned %d tickers for filters=%s", len(tickers), filters)
        _save_cache(cache_path, tickers)
        return tickers
    except Exception as e:
        logger.warning("Finviz call failed (%s); attempting expired-cache fallback", e)
        cached = _load_cache(cache_path, max_age_seconds=None)
        if cached is not None:
            logger.warning("Returning %d expired-cached tickers", len(cached))
            return cached
        logger.error("Finviz unavailable and no cache present — returning empty list")
        return []


def _load_cache(path: Path, max_age_seconds: Optional[int]) -> Optional[list[str]]:
    """Load cached tickers. ``max_age_seconds=None`` means accept any age."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    tickers = data.get("tickers")
    if not isinstance(tickers, list):
        return None
    ts = data.get("timestamp", 0)
    if max_age_seconds is not None and (time.time() - ts) > max_age_seconds:
        return None
    return [str(t) for t in tickers]


def _save_cache(path: Path, tickers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": time.time(), "tickers": tickers}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
