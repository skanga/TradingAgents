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


# Paraphrase pattern observed on multiple Market Analyst runs: the LLM
# emits "Need <tool_name(s)>." as a stand-in for "I would call more tools
# but I'm out of time" — followed by the actual report. e.g.
# "Need atr." (2026-05-05), "Need boll, atr." (2026-05-05).
# Constrained to short alphanumeric+comma+space content so it can't
# falsely match legitimate report text like "Need to monitor the Fed".
_NEED_TOOLS_RE = re.compile(r"^\s*Need\s+[\w, ]{1,40}\.\s*", re.IGNORECASE)

# Markdown-structure tells (any one passing means the report has form).
_STRUCTURE_RE = re.compile(
    r"(?m)^"
    r"(#{1,6}\s"      # heading
    r"|[-*+]\s"       # bullet list
    r"|\d+\.\s"       # numbered list
    r"|\|.*\|)"       # table row
)


def strip_reasoning_leak(content: str) -> str:
    """Strip a leaked LLM reasoning-trace prefix from ``content``.

    Some LLMs (notably free-tier OpenRouter models) paraphrase the system
    prompt's "knows to stop" instruction at the head of their final answer
    instead of jumping straight into the report. The motivating example
    came from a SPY run where the Market Analyst's output opened with::

        Finally other: choose close_10_ema, macd, rsi, ... .We can stop.
        **FINAL TRANSACTION PROPOSAL: HOLD** – ...

    The phrase ``"We can stop"`` is a direct paraphrase of the analyst
    system prompt's *"so the team knows to stop"* instruction and is the
    most reliable marker — none of our reports legitimately use it. When
    found in the first 1000 characters, this helper drops everything up
    to (and including) that marker plus any leading whitespace.

    No marker → ``content`` unchanged, so the helper is safe to apply
    unconditionally to any analyst output.
    """
    if not isinstance(content, str) or not content:
        return content

    # Strip the "Need <tool_name(s)>." paraphrase first if it leads the
    # content. This pattern was observed on multiple SPY Market Analyst
    # runs (2026-05-05): the LLM emits something like "Need atr." as
    # the head of its final answer before producing the real report.
    # Apply before the literal-marker scan because the trailing period
    # provides a clean strip boundary that the literal scan can't match.
    m = _NEED_TOOLS_RE.match(content)
    if m:
        content = content[m.end():]

    head = content[:1000]
    # Ordered most-specific first. Every marker here is a paraphrase of a
    # system-prompt instruction and never appears in legitimate analyst
    # report content. Adding generic words like "may stop" alone would
    # falsely match phrases like "the Fed may stop hiking" — each marker
    # includes context unique to the meta-instruction ("due to time
    # limit", "knows to stop"). "due to time limit may stop" was
    # emitted verbatim by gpt-oss-20b on the 2026-05-05 SPY run;
    # "We can stop." is the canonical paraphrase of the system prompt's
    # "knows to stop" stop-condition instruction.
    # NOTE: each marker must terminate at a clear sentence boundary
    # (period or end-of-phrase) so the strip ends in a clean place.
    # "Finally other:" was *also* observed paired with "We can stop." in
    # the May 4 SPY run, but "We can stop." appeared later in the same
    # string and provides the right strip boundary; standalone
    # "Finally other:" without a trailing period doesn't terminate
    # cleanly and is intentionally not in this list.
    for marker in (
        "due to time limit may stop.",  # observed verbatim 2026-05-05
        "due to time limit may stop",
        "We can stop.", "we can stop.", "We can stop", "we can stop",
    ):
        idx = head.find(marker)
        if idx == -1:
            continue
        end = idx + len(marker)
        while end < len(content) and content[end] in " \t\n\r":
            end += 1
        return content[end:]
    return content


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
    max_tool_rounds: Any = None,
) -> Tuple[Any, str]:
    """Invoke ``chain.invoke(messages)`` with one retry on degenerate output.

    Behaviour:

    - If the result still requests tool calls AND we have headroom under
      ``max_tool_rounds``, return it verbatim with an empty report — the
      LangGraph supervisor will route to ToolNode and call this analyst
      again on the next turn.
    - If the result still requests tool calls AND we are at or over the
      cap, the conditional logic is about to force-terminate this
      analyst — substitute :func:`make_unavailable_report` so downstream
      sees a coherent placeholder instead of empty content. (Without
      this branch the empty string propagates to the markdown writer
      which silently skips the section, motivating the
      2026-05-05 trial regression where Market Analyst sections went
      missing for both SPY and AAPL.)
    - Otherwise, capture ``result.content`` as the report. If it is
      degenerate, append ``retry_user_prompt`` and re-invoke the chain
      once. If the retry also produces a degenerate answer (or returns
      tool calls), substitute the unavailable placeholder.

    Returns ``(message_to_append_to_state, report_string)``. The caller
    is responsible for returning these to the LangGraph state.

    ``max_tool_rounds=None`` disables the cap-aware substitution path
    (the helper behaves exactly as before for callers that don't want
    that semantics).
    """
    result = chain.invoke(messages)

    if getattr(result, "tool_calls", None):
        # Mid-conversation tool call. Did this round just push us over
        # the cap? Count tool-bearing AIMessages in the input plus this
        # new result; if total >= cap, conditional_logic will route to
        # Msg Clear next, and the empty report we'd otherwise return
        # would propagate to state and silently disappear from the
        # rendered report.
        if isinstance(max_tool_rounds, int) and max_tool_rounds > 0:
            prior_rounds = sum(
                1 for m in messages
                if getattr(m, "tool_calls", None)
            )
            if prior_rounds + 1 >= max_tool_rounds:
                logger.warning(
                    "%s would be force-terminated by tool-round cap "
                    "(%d rounds); substituting unavailable placeholder.",
                    analyst_label, max_tool_rounds,
                )
                placeholder = make_unavailable_report(
                    analyst_label=analyst_label,
                    original=(
                        f"force-terminated after {prior_rounds + 1} tool-"
                        f"calling rounds without producing a final answer"
                    ),
                )
                return result, placeholder
        return result, ""

    report = strip_reasoning_leak(getattr(result, "content", "") or "")
    if not is_degenerate_report(report):
        return result, report

    logger.warning(
        "%s emitted a degenerate response (%d chars); retrying once.",
        analyst_label,
        len(report),
    )
    retry_result = chain.invoke(list(messages) + [("user", retry_user_prompt)])
    retry_content = strip_reasoning_leak(getattr(retry_result, "content", "") or "")
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
