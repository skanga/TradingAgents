"""Validate completion metadata before generated prose loses its message envelope."""

from collections.abc import Mapping
from typing import Any


class IncompleteResponseError(ValueError):
    """The provider returned no usable, complete deliverable."""


def check_completion(response: Any) -> None:
    """Reject known incomplete/blocked generations; absent metadata is inconclusive.

    Structured tool responses legitimately have empty content, so text validation
    belongs to the prose path, not this metadata check.
    """
    metadata = getattr(response, "response_metadata", None)
    if not isinstance(metadata, Mapping):
        return
    incomplete_reasons = {
        "length", "max_tokens", "max_output_tokens", "content_filter", "safety",
        "recitation", "blocklist", "prohibited_content", "malformed_function_call",
    }
    for key in ("finish_reason", "stop_reason"):
        reason = metadata.get(key)
        if isinstance(reason, str) and reason.lower() in incomplete_reasons:
            raise IncompleteResponseError(f"generation did not complete: {reason}")
    if metadata.get("status") in ("incomplete", "failed", "cancelled"):
        raise IncompleteResponseError("generation did not complete")


def _announces_tools(response: Any) -> bool:
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("finish_reason", "stop_reason"):
            reason = metadata.get(key)
            if isinstance(reason, str) and reason.lower() in {"tool_calls", "tool_use", "function_call"}:
                return True
    extra = getattr(response, "additional_kwargs", None)
    if isinstance(extra, Mapping) and (extra.get("tool_calls") or extra.get("function_call")):
        return True
    content = getattr(response, "content", None)
    return isinstance(content, list) and any(
        isinstance(block, dict) and block.get("type") in {"tool_use", "tool_call", "function_call"}
        for block in content
    )


def response_text(response: Any) -> str:
    """Extract answer text only, never reasoning blocks or pending tool requests."""
    check_completion(response)
    for field in ("tool_calls", "invalid_tool_calls"):
        calls = getattr(response, field, None)
        if isinstance(calls, list) and calls:
            raise IncompleteResponseError("generation returned tool calls instead of prose")
    if _announces_tools(response):
        raise IncompleteResponseError("generation announced tool use instead of final prose")
    content = getattr(response, "content", None)
    if isinstance(content, list):
        content = "\n".join(
            block if isinstance(block, str) else block["text"]
            for block in content
            if isinstance(block, str) or (
                isinstance(block, dict)
                and block.get("type") in ("text", "output_text")
                and isinstance(block.get("text"), str)
            )
        )
    if not isinstance(content, str) or not content.strip():
        raise IncompleteResponseError("generation returned empty answer text")
    return content


def _check_tool_or_text(response: Any, allowed_tools: set[str]) -> None:
    check_completion(response)
    invalid = getattr(response, "invalid_tool_calls", None)
    if isinstance(invalid, list) and invalid:
        raise IncompleteResponseError("generation returned malformed tool calls")
    calls = getattr(response, "tool_calls", [])
    if not isinstance(calls, list):
        raise IncompleteResponseError("generation returned malformed tool calls")
    if not calls:
        response_text(response)
        return
    seen = set()
    for call in calls:
        if not isinstance(call, Mapping):
            raise IncompleteResponseError("generation returned malformed tool calls")
        name, call_id = call.get("name"), call.get("id")
        if not isinstance(name, str) or name not in allowed_tools:
            raise IncompleteResponseError("generation requested an unavailable tool")
        if not isinstance(call_id, str) or not call_id.strip() or call_id in seen:
            raise IncompleteResponseError("generation returned missing or duplicate tool-call IDs")
        if not isinstance(call.get("args"), dict):
            raise IncompleteResponseError("generation returned malformed tool arguments")
        seen.add(call_id)


def invoke_complete_tool_or_text(llm: Any, prompt: Any, allowed_tools: set[str]) -> Any:
    """Retry unusable generation once before tool execution or report publication.

    Preserve the accepted message envelope verbatim. Argument schemas and trusted
    state injection remain the ToolNode's responsibility; no tools run here.
    Transport exceptions are not retried by this layer.
    """
    response = llm.invoke(prompt)
    try:
        _check_tool_or_text(response, allowed_tools)
    except IncompleteResponseError:
        response = llm.invoke(prompt)
        _check_tool_or_text(response, allowed_tools)
    return response


def invoke_complete_text(llm: Any, prompt: Any) -> str:
    """Retry unusable prose once; transport exceptions remain the caller's concern."""
    response = llm.invoke(prompt)
    try:
        return response_text(response)
    except IncompleteResponseError:
        return response_text(llm.invoke(prompt))
