"""Shared helpers for invoking an agent with structured output and a graceful fallback.

The Portfolio Manager, Trader, and Research Manager all follow the same
canonical pattern:

1. At agent creation, wrap the LLM with ``with_structured_output(Schema)``
   so the model returns a typed Pydantic instance. If the provider does
   not support structured output (rare; mostly older Ollama models), the
   wrap is skipped and the agent uses free-text generation instead.
2. At invocation, run the structured call and render the result back to
   markdown. If the structured call itself fails for any reason
   (malformed JSON from a weak model, transient provider issue), fall
   back to a plain ``llm.invoke`` so the pipeline never blocks.

Centralising the pattern here keeps the agent factories small and ensures
all three agents log the same warnings when fallback fires.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional, TypeVar

from pydantic import BaseModel

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def bind_structured(llm: Any, schema: type[T], agent_name: str) -> Optional[Any]:
    """Return ``llm.with_structured_output(schema)`` or ``None`` if unsupported.

    Logs a warning when the binding fails so the user understands the agent
    will use free-text generation for every call instead of one-shot fallback.
    """
    try:
        return llm.with_structured_output(schema)
    except (NotImplementedError, AttributeError) as exc:
        logger.warning(
            "%s: provider does not support with_structured_output (%s); "
            "falling back to free-text generation",
            agent_name, exc,
        )
        return None


def invoke_structured_or_freetext(
    structured_llm: Optional[Any],
    plain_llm: Any,
    prompt: Any,
    render: Callable[[T], str],
    agent_name: str,
) -> str:
    """Run the structured call and render to markdown; fall back to free-text on any failure.

    ``prompt`` is whatever the underlying LLM accepts (a string for chat
    invocations, a list of message dicts for chat models that take that
    shape). The same value is forwarded to the free-text path so the
    fallback sees the same input the structured call did.

    When *both* the structured call and the free-text fallback produce
    empty / whitespace-only content, returns a bracketed placeholder so
    downstream agents and the renderer see a coherent failure signal
    instead of an empty section. This was the silent failure mode that
    sank the SPY 2026-05-05 Portfolio Manager run: the structured call
    raised a Pydantic JSON-validation error (LLM emitted malformed JSON
    stuck in a ``"\\n"`` loop), the free-text retry returned empty
    content, and the empty string propagated to the markdown writer
    which correctly skipped the section — but the user lost the
    verdict entirely with no indication of why.
    """
    if structured_llm is not None:
        try:
            result = structured_llm.invoke(prompt)
            rendered = render(result)
            if rendered and rendered.strip():
                return rendered
            logger.warning(
                "%s: structured-output succeeded but render produced empty "
                "content; falling back to free text",
                agent_name,
            )
        except Exception as exc:
            logger.warning(
                "%s: structured-output invocation failed (%s); retrying once as free text",
                agent_name, exc,
            )

    response = plain_llm.invoke(prompt)
    content = getattr(response, "content", "") or ""
    if content.strip():
        return content

    logger.error(
        "%s: free-text fallback returned empty content; substituting "
        "placeholder so downstream sees the failure.",
        agent_name,
    )
    return (
        f"[{agent_name} could not produce a verdict on this run: structured "
        f"output failed and the free-text retry returned no content. "
        f"Treat this section as missing.]"
    )
