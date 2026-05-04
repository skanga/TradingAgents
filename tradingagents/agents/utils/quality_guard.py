"""Detect and recover from degenerate analyst outputs.

Free-tier LLMs occasionally emit a near-empty final answer when they
should have produced a structured report. The motivating example came
from a production AAPL run where the Fundamentals analyst's final
response was literally ``"Call correct."`` — two words, no markdown,
nothing usable for the Bull/Bear/Risk debaters that consume the report
downstream.

This module exposes two layers an analyst factory can use:

1. :func:`is_degenerate_report` and :func:`make_unavailable_report` —
   primitives for detection and a fallback placeholder.
2. :func:`invoke_chain_with_quality_retry` — the composed helper that
   each analyst node actually calls: invoke the chain, retry once with
   a stricter user message if the answer is degenerate, substitute a
   placeholder if the retry also fails.
"""

from __future__ import annotations

import logging
import re
from typing import Any, List, Tuple

logger = logging.getLogger(__name__)

# A real analyst report is at least a few hundred chars long and contains
# some markdown structure (heading, bullet, table row, etc.). The threshold
# is conservative — a one-paragraph honest "tools unavailable" fallback is
# already longer than this.
_MIN_REPORT_CHARS = 200

# Markdown-structure tells (any one passing means the report has form).
_STRUCTURE_RE = re.compile(
    r"(?m)^"
    r"(#{1,6}\s"      # heading
    r"|[-*+]\s"       # bullet list
    r"|\d+\.\s"       # numbered list
    r"|\|.*\|)"       # table row
)


def is_degenerate_report(report: object) -> bool:
    """Return ``True`` when ``report`` looks like a failed LLM final answer.

    Flags as degenerate when *all* hold:

    - the value is empty, non-string, or pure whitespace; OR
    - the trimmed text is shorter than the structure threshold AND
      contains no markdown structure (no headings, lists, or tables).

    A genuine short-but-structured response (e.g. a single bullet list)
    or a longer free-form paragraph still passes — the goal is to catch
    truly empty / two-word responses, not to second-guess terse output.
    """
    if not isinstance(report, str):
        return True
    text = report.strip()
    if not text:
        return True
    if len(text) >= _MIN_REPORT_CHARS:
        return False
    return _STRUCTURE_RE.search(text) is None


def make_unavailable_report(*, analyst_label: str, original: str = "") -> str:
    """Build a placeholder report for downstream agents.

    Used when an analyst produces a degenerate response and a single
    retry still fails. The placeholder is markdown-structured so the
    renderer treats it as a real section, and it explicitly tells
    downstream debaters to treat the section as missing.
    """
    snippet = (original or "").strip()
    if len(snippet) > 200:
        snippet = snippet[:200] + "…"
    quoted = f"`{snippet}`" if snippet else "_(empty)_"
    return (
        f"## {analyst_label} — output unavailable\n\n"
        f"The {analyst_label} produced a degenerate response and a single retry "
        f"did not recover. Original output: {quoted}\n\n"
        f"Downstream agents should treat this section as **missing** for this "
        f"run and weight other analyst inputs accordingly.\n"
    )


# Default user-message used to nudge the LLM into producing a real report
# on the retry pass. Analyst factories can override with a more specific
# instruction (e.g. "include the insider-buying table").
RETRY_USER_PROMPT = (
    "Your previous response was too short or lacked the required structure. "
    "Produce the full report now using the tool data already retrieved in "
    "this conversation. Required: at least one section heading, the markdown "
    "summary table at the end, and concrete numbers cited from the tool "
    "outputs (do not invent data — if a tool returned a bracketed "
    "'unavailable' string, say so and continue with what is available)."
)


def invoke_chain_with_quality_retry(
    chain: Any,
    messages: List[Any],
    *,
    analyst_label: str,
    retry_user_prompt: str = RETRY_USER_PROMPT,
) -> Tuple[Any, str]:
    """Invoke ``chain.invoke(messages)`` with one retry on degenerate output.

    Behaviour:

    - If the result still requests tool calls, return it verbatim with
      an empty report — the LangGraph supervisor will route to ToolNode
      and call this analyst again on the next turn.
    - Otherwise, capture ``result.content`` as the report. If it is
      degenerate, append ``retry_user_prompt`` and re-invoke the chain
      once. If the retry also produces a degenerate answer (or returns
      tool calls), substitute :func:`make_unavailable_report` so
      downstream agents see a coherent placeholder.

    Returns ``(message_to_append_to_state, report_string)``. The caller
    is responsible for returning these to the LangGraph state.
    """
    result = chain.invoke(messages)

    if getattr(result, "tool_calls", None):
        # Mid-conversation tool call; not a final answer to evaluate.
        return result, ""

    report = getattr(result, "content", "") or ""
    if not is_degenerate_report(report):
        return result, report

    logger.warning(
        "%s emitted a degenerate response (%d chars); retrying once.",
        analyst_label,
        len(report),
    )
    retry_result = chain.invoke(list(messages) + [("user", retry_user_prompt)])
    retry_content = getattr(retry_result, "content", "") or ""
    retry_has_tools = bool(getattr(retry_result, "tool_calls", None))

    if not retry_has_tools and not is_degenerate_report(retry_content):
        return retry_result, retry_content

    logger.warning(
        "%s retry also failed; substituting unavailable placeholder.",
        analyst_label,
    )
    placeholder = make_unavailable_report(
        analyst_label=analyst_label, original=report
    )
    return result, placeholder
