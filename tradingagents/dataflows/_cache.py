"""File-based JSON cache for external API responses.

Used by the new dataflow modules (sec_insider, congress_trades, options_flow,
macro_data, earnings_transcript, sector_analysis) to avoid hammering external
APIs across repeated backtest runs for the same (ticker, date) combination.

Cache location: ``<data_cache_dir>/api/<source>/<sha1(key)>.json`` where
``data_cache_dir`` is read from the runtime config (defaults to
``~/.tradingagents/cache``). Each entry stores ``{"ts": <epoch>, "value": str}``.

The cache is intentionally simple: filesystem-only, no concurrency control
beyond atomic-rename writes, and no automatic eviction (callers pass a TTL).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional


def _cache_root() -> Path:
    from tradingagents.dataflows.config import get_config
    base = get_config().get("data_cache_dir") or os.path.join(
        os.path.expanduser("~"), ".tradingagents", "cache"
    )
    return Path(base) / "api"


def _key_hash(key: Mapping[str, Any]) -> str:
    canonical = json.dumps(key, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:24]


def _entry_path(source: str, key: Mapping[str, Any]) -> Path:
    return _cache_root() / source / f"{_key_hash(key)}.json"


def cache_get(source: str, key: Mapping[str, Any], ttl_seconds: int) -> Optional[str]:
    """Return cached value if present and not older than ``ttl_seconds``."""
    path = _entry_path(source, key)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            entry = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - entry.get("ts", 0) > ttl_seconds:
        return None
    value = entry.get("value")
    return value if isinstance(value, str) else None


def cache_put(source: str, key: Mapping[str, Any], value: str) -> None:
    """Write ``value`` to cache under ``(source, key)`` atomically."""
    path = _entry_path(source, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps({"ts": time.time(), "value": value})
    fd, tmp = tempfile.mkstemp(prefix=".cache_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
