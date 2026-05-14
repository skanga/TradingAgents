"""Append-only markdown decision log for TradingAgents."""

from typing import List, Optional
from pathlib import Path
import json
import re
import tempfile
import threading
import weakref

from tradingagents.agents.utils.rating import parse_rating


class TradingMemoryLog:
    """Append-only markdown log of trading decisions and reflections."""

    # HTML comment: cannot appear in LLM prose output, safe as a hard delimiter
    _SEPARATOR = "\n\n<!-- ENTRY_END -->\n\n"
    _JSONL_VERSION = 1
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
    ) -> None:
        """Append pending entry at end of propagate(). No LLM call."""
        if not self._log_path:
            return
        with self._path_lock():
            entries = self._read_entries_uncached()
            for entry in entries:
                if (
                    entry["date"] == trade_date
                    and entry["ticker"] == ticker
                    and entry.get("pending")
                ):
                    return
            rating = parse_rating(final_trade_decision)
            entries.append({
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
            return [entry.copy() for entry in self._entries_cache]

        entries = self._read_entries_uncached()
        self._entries_cache_mtime_ns = mtime_ns
        self._entries_cache = [entry.copy() for entry in entries]
        return entries

    def get_pending_entries(self) -> List[dict]:
        """Return entries with outcome:pending (for Phase B)."""
        return [e for e in self.load_entries() if e.get("pending")]

    def get_past_context(self, ticker: str, n_same: int = 5, n_cross: int = 3) -> str:
        """Return formatted past context string for agent prompt injection."""
        entries = [e for e in self.load_entries() if not e.get("pending")]
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
        if not entry.get("pending"):
            return self._format_full(entry)
        tag = f"[{entry['date']} | {entry['ticker']} | {entry['rating']} | pending]"
        return "\n\n".join([tag, f"DECISION:\n{entry.get('decision', '')}"])

    # --- Update path (Phase B) ---

    def batch_update_with_outcomes(self, updates: List[dict]) -> None:
        """Apply multiple outcome updates in a single read + atomic write.

        Each element of updates must have keys: ticker, trade_date,
        raw_return, alpha_return, holding_days, reflection.
        """
        if not self._log_path or not self._log_path.exists() or not updates:
            return

        entries = self._read_entries_uncached()

        update_map = {}
        for update in updates:
            key = (update["trade_date"], update["ticker"])
            if key in update_map:
                raise ValueError(
                    f"duplicate outcome update for trade_date={key[0]!r}, ticker={key[1]!r}"
                )
            update_map[key] = update

        updated_entries = []
        for entry in entries:
            if not entry.get("pending"):
                updated_entries.append(entry)
                continue

            upd = update_map.get((entry["date"], entry["ticker"]))
            if upd is None:
                updated_entries.append(entry)
                continue

            entry = entry.copy()
            entry.update({
                "pending": False,
                "raw": f"{upd['raw_return']:+.1%}",
                "alpha": f"{upd['alpha_return']:+.1%}",
                "holding": f"{upd['holding_days']}d",
                "reflection": upd["reflection"],
            })
            updated_entries.append(entry)
            del update_map[(entry["date"], entry["ticker"])]

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
                "date": entry["date"],
                "ticker": entry["ticker"],
                "rating": entry["rating"],
                "pending": bool(entry.get("pending")),
                "raw": entry.get("raw"),
                "alpha": entry.get("alpha"),
                "holding": entry.get("holding"),
                "decision": entry.get("decision", ""),
                "reflection": entry.get("reflection", ""),
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
            "date": str(payload["date"]),
            "ticker": str(payload["ticker"]),
            "rating": str(payload["rating"]),
            "pending": bool(payload["pending"]),
            "raw": payload.get("raw"),
            "alpha": payload.get("alpha"),
            "holding": payload.get("holding"),
            "decision": str(payload.get("decision", "")),
            "reflection": str(payload.get("reflection", "")),
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
            if tmp_path.exists():
                tmp_path.unlink()

    def _apply_rotation_entries(self, entries: List[dict]) -> List[dict]:
        """Drop oldest resolved entries when their count exceeds max_entries.

        Pending entries are always kept (they represent unprocessed work).
        Returns ``entries`` unchanged when rotation is disabled or under cap.
        """
        if not self._max_entries or self._max_entries <= 0:
            return entries

        resolved_count = sum(1 for entry in entries if not entry.get("pending"))
        if resolved_count <= self._max_entries:
            return entries

        to_drop = resolved_count - self._max_entries
        kept: List[dict] = []
        for entry in entries:
            if not entry.get("pending") and to_drop > 0:
                to_drop -= 1
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
            "date": fields["date"],
            "ticker": fields["ticker"],
            "rating": fields["rating"],
            "pending": fields["pending"],
            "raw": fields["raw"],
            "alpha": fields["alpha"],
            "holding": fields["holding"],
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

        parts = [part.strip() for part in tag_line[1:-1].split(" | ")]
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
            return {
                "date": parts[0],
                "ticker": parts[1],
                "rating": parts[2],
                "pending": False,
                "raw": parts[3],
                "alpha": parts[4],
                "holding": parts[5],
            }
        return None

    def _format_full(self, e: dict) -> str:
        raw = e["raw"] or "n/a"
        alpha = e["alpha"] or "n/a"
        holding = e["holding"] or "n/a"
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {raw} | {alpha} | {holding}]"
        parts = [tag, f"DECISION:\n{e['decision']}"]
        if e["reflection"]:
            parts.append(f"REFLECTION:\n{e['reflection']}")
        return "\n\n".join(parts)

    def _format_reflection_only(self, e: dict) -> str:
        tag = f"[{e['date']} | {e['ticker']} | {e['rating']} | {e['raw'] or 'n/a'}]"
        if e["reflection"]:
            return f"{tag}\n{e['reflection']}"
        text = e["decision"][:300]
        suffix = "..." if len(e["decision"]) > 300 else ""
        return f"{tag}\n{text}{suffix}"
