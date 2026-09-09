"""Append-only markdown decision log for TradingAgents."""

import json
import logging
import re
import tempfile
import threading
import weakref
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from tradingagents.agents.utils.outcomes import format_outcome, score_outcome
from tradingagents.agents.utils.rating import RATING_REVIEW, extract_rating, parse_actionable_rating

logger = logging.getLogger(__name__)


def _valid_learning_entry(entry: dict) -> bool:
    decision = entry.get("decision", "")
    rating = parse_actionable_rating(decision)
    # Older logs used imperative prose ("Buy NVDA.") rather than a label.
    # Retain these only when the initial rating agrees with the stored tag.
    if rating == RATING_REVIEW and not re.search(r"\brating\b", decision, re.IGNORECASE):
        if re.match(r"^(Buy|Overweight|Hold|Underweight|Sell)\s+", decision, re.IGNORECASE):
            words = re.findall(r"\b(Buy|Overweight|Hold|Underweight|Sell)\b", decision, re.IGNORECASE)
            if len({word.lower() for word in words}) == 1:
                rating = extract_rating(decision)
    return rating != RATING_REVIEW and rating == entry.get("rating")


def _namespace(value: str) -> str:
    if value not in ("live", "simulation", "legacy"):
        raise ValueError("memory namespace must be live, simulation, or legacy")
    return value


def outcome_known_by(entry: dict, as_of: str) -> bool:
    """Require valid decision/outcome dates ordered within the run cutoff."""
    dates = (entry.get("date"), entry.get("resolution_date"))
    try:
        for value in dates:
            if datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d") != value:
                return False
    except (TypeError, ValueError):
        return False
    return dates[0] <= dates[1] <= as_of


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections."""

    # HTML comment: cannot appear in LLM prose output, safe as a hard delimiter
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _JSONL_VERSION = 2
    # Precompiled patterns — avoids re-compilation on every load_entries() call
    _DECISION_RE = re.compile(r"DECISION:\n(.*?)(?=\nREFLECTION:|\Z)", re.DOTALL)
    _REFLECTION_RE = re.compile(r"REFLECTION:\n(.*?)$", re.DOTALL)
    _path_locks: weakref.WeakValueDictionary[Path, threading.Lock] = (
        weakref.WeakValueDictionary()
    )
    _path_locks_guard = threading.Lock()

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self._log_path = None
        path = cfg.get("memory_log_path")
        if path:
            self._log_path = Path(path).expanduser()
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
        # Optional cap on resolved entries. None disables rotation.
        self._max_entries = cfg.get("memory_log_max_entries")
        self._entries_cache_mtime_ns: int | None = None
        self._entries_cache: List[dict] | None = None

    # --- Write path (Phase A) ---

    def store_decision(
        self,
        ticker: str,
        trade_date: str,
        final_trade_decision: str,
        *, namespace: str = "legacy",
    ) -> None:
        """Append a namespaced pending entry. Unscoped callers remain legacy."""
        namespace = _namespace(namespace)
        if not self._log_path:
            return
        rating = parse_actionable_rating(final_trade_decision)
        if rating == RATING_REVIEW:
            logger.warning("Not storing invalid decision for %s on %s: REVIEW", ticker, trade_date)
            return
        with self._path_lock():
            entries = self._read_entries_uncached()
            for entry in entries:
                if (
                    entry.get("namespace", "legacy") == namespace
                    and entry["date"] == trade_date
                    and entry["ticker"] == ticker
                    and entry.get("pending")
                    and _valid_learning_entry(entry)
                ):
                    return
            entries.append({
                "namespace": namespace,
                "date": trade_date,
                "ticker": ticker,
                "rating": rating,
                "pending": True,
                "raw": None,
                "alpha": None,
                "holding": None,
                "decision": final_trade_decision,
                "reflection": "",
            })
            self._atomic_write_text(self._serialize_jsonl(entries))

    # --- Read path (Phase A) ---

    def load_entries(self) -> List[dict]:
        """Parse all entries from log. Returns list of dicts."""
        if not self._log_path or not self._log_path.exists():
            self._invalidate_entries_cache()
            return []

        mtime_ns = self._log_path.stat().st_mtime_ns
        if (
            self._entries_cache is not None
            and self._entries_cache_mtime_ns == mtime_ns
        ):
            return deepcopy(self._entries_cache)

        entries = self._read_entries_uncached()
        self._entries_cache_mtime_ns = mtime_ns
        self._entries_cache = deepcopy(entries)
        return entries

    def get_pending_entries(self, *, namespace: str | None = None) -> List[dict]:
        """Return pending entries; unscoped calls are for legacy/audit clients."""
        if namespace is not None:
            _namespace(namespace)
        return [e for e in self.load_entries() if e.get("pending") and _valid_learning_entry(e)
                and (namespace is None or e.get("namespace", "legacy") == namespace)]

    def get_past_context(self, ticker: str, n_same: int = 5, n_cross: int = 3,
                         as_of: str | None = None, *, namespace: str | None = None) -> str:
        """Run queries require a namespace and cutoff; unscoped reads are legacy/audit."""
        if namespace is not None:
            _namespace(namespace)
            if as_of is None:
                raise ValueError("Namespaced memory context requires as_of")
        if as_of is not None:
            if datetime.strptime(as_of, "%Y-%m-%d").strftime("%Y-%m-%d") != as_of:
                raise ValueError("as_of must use YYYY-MM-DD")
        entries = [
            e for e in self.load_entries()
            if not e.get("pending") and _valid_learning_entry(e)
            and (namespace is None or e.get("namespace", "legacy") == namespace)
            and (as_of is None or outcome_known_by(e, as_of))
        ]
        if not entries:
            return ""

        same: list[dict] = []
        cross: list[dict] = []
        for e in reversed(entries):
            if len(same) >= n_same and len(cross) >= n_cross:
                break
            if e["ticker"] == ticker and len(same) < n_same:
                same.append(e)
            elif e["ticker"] != ticker and len(cross) < n_cross:
                cross.append(e)

        if not same and not cross:
            return ""

        parts = []
        if same:
            parts.append(f"Past analyses of {ticker} (most recent first):")
            parts.extend(self._format_full(e) for e in same)
        if cross:
            parts.append("Recent cross-ticker lessons:")
            parts.extend(self._format_reflection_only(e) for e in cross)
        return "\n\n".join(parts)

    def format_entry(self, entry: dict) -> str:
        """Return a human-readable markdown view of a parsed memory entry."""
        scope = f"Memory namespace: {entry.get('namespace', 'legacy')}\n\n"
        if not entry.get("pending"):
            return scope + self._format_full(entry)
        tag = f"[{entry['date']} | {entry['ticker']} | {entry['rating']} | pending]"
        return scope + "\n\n".join([tag, f"DECISION:\n{entry.get('decision', '')}"])

    # --- Update path (Phase B) ---

    def batch_update_with_outcomes(self, updates: List[dict]) -> None:
        """Apply multiple outcome updates in a single read + atomic write.

        Each element of updates must have keys: ticker, trade_date,
        raw_return, alpha_return, holding_days, reflection.
        """
        if not self._log_path or not self._log_path.exists() or not updates:
            return

        with self._path_lock():
            entries = self._read_entries_uncached()

            update_map = {}
            for update in updates:
                key = (_namespace(update.get("namespace", "legacy")), update["trade_date"], update["ticker"])
                if key in update_map:
                    raise ValueError(
                        f"duplicate outcome update for namespace={key[0]!r}, trade_date={key[1]!r}, ticker={key[2]!r}"
                    )
                update_map[key] = update

            updated_entries = []
            for entry in entries:
                if not entry.get("pending") or not _valid_learning_entry(entry):
                    updated_entries.append(entry)
                    continue

                key = (entry.get("namespace", "legacy"), entry["date"], entry["ticker"])
                upd = update_map.get(key)
                if upd is None:
                    updated_entries.append(entry)
                    continue

                outcome = score_outcome(entry["rating"], upd["raw_return"], upd["alpha_return"])
                outcome.update({
                    "evaluation_sessions": upd["holding_days"],
                    "evaluation_start": upd.get("evaluation_start"),
                    "benchmark_name": upd.get("benchmark_name"),
                    "resolution_date": upd.get("resolution_date"),
                })
                entry = entry.copy()
                entry.update({
                    "outcome": outcome,
                    "pending": False,
                    "raw": f"{upd['raw_return']:+.1%}",
                    "alpha": f"{upd['alpha_return']:+.1%}",
                    "holding": f"{upd['holding_days']}d",
                    "reflection": upd["reflection"],
                    "resolution_date": upd.get("resolution_date"),
                })
                updated_entries.append(entry)
                del update_map[key]

            updated_entries = self._apply_rotation_entries(updated_entries)
            self._atomic_write_text(self._serialize_jsonl(updated_entries))

    # --- Helpers ---

    def _invalidate_entries_cache(self) -> None:
        self._entries_cache_mtime_ns = None
        self._entries_cache = None

    def _read_entries_uncached(self) -> List[dict]:
        if not self._log_path or not self._log_path.exists():
            return []

        text = self._log_path.read_text(encoding="utf-8")
        json_entries: List[dict] = []
        legacy_lines: list[str] = []
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                parsed = self._parse_jsonl_entry(stripped)
                if parsed:
                    json_entries.append(parsed)
                    continue
            legacy_lines.append(line)

        legacy_text = "\n".join(legacy_lines)
        legacy_entries = []
        raw_entries = [e.strip() for e in legacy_text.split(self._SEPARATOR) if e.strip()]
        for raw in raw_entries:
            parsed = self._parse_legacy_entry(raw)
            if parsed:
                legacy_entries.append(parsed)
        return legacy_entries + json_entries

    def _serialize_jsonl(self, entries: List[dict]) -> str:
        lines = []
        for entry in entries:
            payload = {
                "version": self._JSONL_VERSION,
                "namespace": entry.get("namespace", "legacy"),
                "date": entry["date"],
                "ticker": entry["ticker"],
                "rating": entry["rating"],
                "pending": bool(entry.get("pending")),
                "raw": entry.get("raw"),
                "alpha": entry.get("alpha"),
                "holding": entry.get("holding"),
                "decision": entry.get("decision", ""),
                "reflection": entry.get("reflection", ""),
                "resolution_date": entry.get("resolution_date"),
                "outcome": entry.get("outcome"),
            }
            lines.append(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        return "\n".join(lines) + ("\n" if lines else "")

    def _parse_jsonl_entry(self, line: str) -> Optional[dict]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        required = ("date", "ticker", "rating", "pending")
        if any(key not in payload for key in required):
            return None

        return {
            "namespace": payload.get("namespace", "legacy"),
            "date": str(payload["date"]),
            "ticker": str(payload["ticker"]),
            "rating": str(payload["rating"]),
            "pending": bool(payload["pending"]),
            "raw": payload.get("raw"),
            "alpha": payload.get("alpha"),
            "holding": payload.get("holding"),
            "decision": str(payload.get("decision", "")),
            "reflection": str(payload.get("reflection", "")),
            "resolution_date": payload.get("resolution_date"),
            "outcome": payload.get("outcome"),
        }

    def _path_lock(self) -> threading.Lock:
        assert self._log_path is not None
        lock_path = self._log_path.resolve()
        with self._path_locks_guard:
            lock = self._path_locks.get(lock_path)
            if lock is None:
                lock = threading.Lock()
                self._path_locks[lock_path] = lock
            return lock

    def _atomic_write_text(self, text: str) -> None:
        """Write the full memory log with an atomic same-directory replace."""
        assert self._log_path is not None
        tmp_name = None
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            delete=False,
            dir=self._log_path.parent,
            prefix=f"{self._log_path.name}.",
            suffix=".tmp",
        ) as tmp:
            tmp.write(text)
            tmp_name = tmp.name

        tmp_path = Path(tmp_name)
        try:
            tmp_path.replace(self._log_path)
        finally:
            self._invalidate_entries_cache()
            # On successful replace() the temp path has been atomically renamed.
            # If replace() fails before the rename, remove the leftover temp file.
            if tmp_path.exists():
                tmp_path.unlink()

    def _apply_rotation_entries(self, entries: List[dict]) -> List[dict]:
        """Drop oldest resolved entries per namespace when count exceeds max_entries.

        Pending entries are always kept (they represent unprocessed work).
        Returns ``entries`` unchanged when rotation is disabled or under cap.
        """
        if not self._max_entries or self._max_entries <= 0:
            return entries

        resolved_counts = Counter(entry.get("namespace", "legacy") for entry in entries
                                  if not entry.get("pending"))
        to_drop = {namespace: max(0, count - self._max_entries)
                   for namespace, count in resolved_counts.items()}
        kept: List[dict] = []
        for entry in entries:
            namespace = entry.get("namespace", "legacy")
            if not entry.get("pending") and to_drop.get(namespace, 0) > 0:
                to_drop[namespace] -= 1
                continue
            kept.append(entry)
        return kept

    def _parse_legacy_entry(self, raw: str) -> Optional[dict]:
        lines = raw.strip().splitlines()
        if not lines:
            return None
        tag_line = lines[0].strip()
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None
        fields = self._parse_legacy_tag(tag_line)
        if not fields:
            return None
        entry = {
            "namespace": "legacy",
            "date": fields["date"],
            "ticker": fields["ticker"],
            "rating": fields["rating"],
            "pending": fields["pending"],
            "raw": fields["raw"],
            "alpha": fields["alpha"],
            "holding": fields["holding"],
            "resolution_date": fields.get("resolution_date"),
        }
        body = "\n".join(lines[1:]).strip()
        decision_match = self._DECISION_RE.search(body)
        reflection_match = self._REFLECTION_RE.search(body)
        entry["decision"] = decision_match.group(1).strip() if decision_match else ""
        entry["reflection"] = reflection_match.group(1).strip() if reflection_match else ""
        return entry

    def _parse_legacy_tag(self, tag_line: str) -> Optional[dict]:
        if not (tag_line.startswith("[") and tag_line.endswith("]")):
            return None

        # Fork logs can contain literal pipes in tickers; prefer their spaced
        # delimiter while also accepting upstream's compact tag syntax.
        delimiter = " | " if " | " in tag_line else "|"
        parts = [part.strip() for part in tag_line[1:-1].split(delimiter)]
        if len(parts) == 4 and parts[3] == "pending":
            return {
                "date": parts[0],
                "ticker": parts[1],
                "rating": parts[2],
                "pending": True,
                "raw": None,
                "alpha": None,
                "holding": None,
            }
        if len(parts) >= 6:
            # Preserve upstream's point-in-time marker when migrating to JSONL.
            resolution_date = None
            for part in parts[6:]:
                if part.startswith("resolved:"):
                    resolution_date = part[len("resolved:"):].strip()
            return {
                "date": parts[0],
                "ticker": parts[1],
                "rating": parts[2],
                "pending": False,
                "raw": parts[3],
                "alpha": parts[4],
                "holding": parts[5],
                "resolution_date": resolution_date,
            }
        return None

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e.get("outcome"):
            parts.append(format_outcome(e["outcome"]))
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e.get("outcome"):
            tag += "\n" + format_outcome(e["outcome"])
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
