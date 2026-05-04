"""Pin the default ``method`` chosen by NormalizedChatOpenAI.with_structured_output.

Background: a SPY pipeline run logged ``Unknown tool type: 'researchPlan'.
Available tools: ResearchPlan`` from the Research Manager's structured-
output call. Some OpenRouter free-tier models emit lowercased tool names
when responding to a function-calling-style structured-output request,
which langchain's case-sensitive parser rejects.

The fix routes Chat-Completions endpoints (OpenRouter / xAI / Qwen /
GLM / Ollama) through ``json_schema`` instead, bypassing tool-name
matching. Native OpenAI continues to use ``function_calling`` through
the Responses API to avoid Pydantic serialization warnings on its
json_schema parse path.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from tradingagents.llm_clients.openai_client import NormalizedChatOpenAI


@pytest.fixture
def captured() -> dict:
    return {}


@pytest.fixture
def patched_parent(captured):
    """Patch ChatOpenAI.with_structured_output so we can inspect the
    ``method`` kwarg our subclass forwards without making a network call."""

    def _capture(self, schema, *, method=None, **kwargs):
        captured["method"] = method
        return None

    with patch.object(ChatOpenAI, "with_structured_output", _capture):
        yield captured


def test_chat_completions_provider_defaults_to_json_schema(patched_parent):
    """OpenRouter / xAI / Qwen / GLM / Ollama path — use_responses_api=False
    (the framework's default for non-OpenAI providers)."""
    llm = NormalizedChatOpenAI(model="x", api_key="test", use_responses_api=False)
    llm.with_structured_output(dict)
    assert patched_parent["method"] == "json_schema"


def test_responses_api_defaults_to_function_calling(patched_parent):
    """Native OpenAI uses Responses API; keep function_calling there to
    avoid the noisy Pydantic warnings on the json_schema parse path."""
    llm = NormalizedChatOpenAI(model="gpt-4o", api_key="test", use_responses_api=True)
    llm.with_structured_output(dict)
    assert patched_parent["method"] == "function_calling"


def test_explicit_method_kwarg_wins_over_default(patched_parent):
    """Users (or future code) can still override by passing method explicitly."""
    llm = NormalizedChatOpenAI(model="x", api_key="test", use_responses_api=False)
    llm.with_structured_output(dict, method="function_calling")
    assert patched_parent["method"] == "function_calling"

    llm.with_structured_output(dict, method="json_mode")
    assert patched_parent["method"] == "json_mode"
