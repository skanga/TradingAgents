"""OpenAI-compatible client applies a framework retry default and lets users override it.

Backstory: a dry-run of the screener pipeline failed at the first Market
Analyst tool call with ``openai.RateLimitError 429`` because the
``qwen3-next:free`` model on OpenRouter was upstream-throttled by Venice.
The openai SDK's default of 2 retries wasn't enough to ride it out.
"""

from unittest.mock import patch


def _captured_kwargs(provider: str, model: str, **client_kwargs) -> dict:
    """Construct an OpenAIClient and return the kwargs handed to ChatOpenAI."""
    captured: dict = {}

    class _FakeChat:
        def __init__(self, **kw):
            captured.update(kw)

    # Patch BOTH chat classes — DeepSeekChatOpenAI is selected for provider=deepseek.
    with patch("tradingagents.llm_clients.openai_client.NormalizedChatOpenAI", _FakeChat), \
         patch("tradingagents.llm_clients.openai_client.DeepSeekChatOpenAI", _FakeChat):
        from tradingagents.llm_clients.openai_client import OpenAIClient
        OpenAIClient(model=model, provider=provider, **client_kwargs).get_llm()
    return captured


def test_default_max_retries_applied_to_openrouter():
    kw = _captured_kwargs("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free")
    assert kw.get("max_retries") == 6, kw


def test_default_max_retries_applied_to_native_openai():
    kw = _captured_kwargs("openai", "gpt-5.4-mini")
    assert kw.get("max_retries") == 6, kw
    # Native OpenAI also uses the Responses API
    assert kw.get("use_responses_api") is True


def test_default_max_retries_applied_to_deepseek_subclass():
    """DeepSeekChatOpenAI is its own subclass; the retry default still propagates."""
    kw = _captured_kwargs("deepseek", "deepseek-chat")
    assert kw.get("max_retries") == 6, kw


def test_user_override_wins_over_framework_default():
    kw = _captured_kwargs(
        "openrouter",
        "anything/whatever:free",
        max_retries=12,
    )
    assert kw.get("max_retries") == 12, kw


def test_user_override_to_zero_disables_retries():
    """Setting max_retries=0 explicitly should disable retries (no setdefault clobber)."""
    kw = _captured_kwargs(
        "openai",
        "gpt-5.4-mini",
        max_retries=0,
    )
    assert kw.get("max_retries") == 0, kw
