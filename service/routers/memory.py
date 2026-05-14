"""Rolling decision memory log."""

from fastapi import APIRouter

from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.default_config import DEFAULT_CONFIG
from gui.log_browser import memory_log_path
from service.schemas import MemoryEntry, MemoryResponse

router = APIRouter(prefix="/memory", tags=["memory"])


@router.get("", response_model=MemoryResponse)
def get_memory() -> MemoryResponse:
    log = TradingMemoryLog(DEFAULT_CONFIG)
    entries: list[MemoryEntry] = []
    for entry in log.load_entries():
        entries.append(
            MemoryEntry(raw=log.format_entry(entry), resolved=not entry.get("pending"))
        )

    resolved = sum(1 for e in entries if e.resolved)
    return MemoryResponse(
        path=str(memory_log_path()),
        entries=entries,
        total=len(entries),
        resolved_count=resolved,
        pending_count=len(entries) - resolved,
    )
