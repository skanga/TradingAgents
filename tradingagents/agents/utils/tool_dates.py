"""Trusted LangGraph state is injected by the executor, never supplied by the LLM."""

from datetime import datetime
from typing import Annotated

from langgraph.prebuilt import InjectedState
from pydantic import Field

from tradingagents.dataflows.utils import get_current_date

RunState = Annotated[dict | None, InjectedState]
# Python 3.10 get_type_hints wraps annotations whose Python default is None
# in Optional, burying InjectedState from the model-schema filter. A Pydantic
# field default preserves the outer annotation and still validates to None.
DEFAULT_RUN_STATE = Field(default=None)


def _date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").strftime("%Y-%m-%d")


def bounded_date(requested: str | None, state: dict | None) -> str | None:
    """Direct Python calls retain explicit date semantics; graph calls are capped."""
    if state is None or state is DEFAULT_RUN_STATE:
        return _date(requested) if requested is not None else None
    cutoff = min(_date(state["trade_date"]), get_current_date())
    return min(_date(requested), cutoff) if requested is not None else cutoff


def bounded_range(start: str, end: str, state: dict | None) -> tuple[str, str]:
    start = _date(start)
    end = bounded_date(end, state)
    if end is None or start > end:
        raise ValueError("Requested date range starts after the analysis cutoff")
    return start, end


def live_only_notice(state: dict | None, label: str) -> str | None:
    cutoff = bounded_date(None, state)
    if cutoff is not None and cutoff < get_current_date():
        return f"DATA_UNAVAILABLE: {label} withheld for {cutoff}; source is live-only with no historical vintage."
    return None
