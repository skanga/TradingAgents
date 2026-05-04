"""Unit tests for FallbackChatModel.

Verifies the wrapper falls through to the next model on recoverable
errors (the kind we'd see when an OpenRouter free-tier upstream throws
a 429 on a CSCO analysis), preserves the fallback chain through
``bind_tools`` and ``with_structured_output``, and does *not* fall
through on non-recoverable errors (which would just fail again on
the next model and mask the real bug).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tradingagents.llm_clients import fallback as fallback_module
from tradingagents.llm_clients.fallback import FallbackChatModel


# A recoverable error stand-in. Patching ``_RECOVERABLE`` to include this
# class lets us test fall-through semantics without constructing a real
# ``openai.RateLimitError`` (which requires a fully-formed httpx Response).
class _Recoverable(Exception):
    pass


# A non-recoverable error stand-in (e.g. auth, schema validation).
class _Fatal(Exception):
    pass


@pytest.fixture(autouse=True)
def _patch_recoverable(monkeypatch):
    monkeypatch.setattr(fallback_module, "_RECOVERABLE", (_Recoverable,))


def _model(name: str, *, invoke_side_effects=None) -> MagicMock:
    """Build a fake chat model. ``invoke_side_effects`` is a list of either
    return values or exceptions to raise on each successive ``.invoke`` call.
    Exceptions are raised; everything else is returned."""
    m = MagicMock(name=name, model_name=name)
    m.invoke.side_effect = invoke_side_effects or [f"ok:{name}"]
    return m


# --- Direct invoke fall-through ----------------------------------------


def test_invoke_primary_success_skips_fallbacks():
    p = _model("primary", invoke_side_effects=["ok:primary"])
    f1 = _model("fb1")
    f2 = _model("fb2")
    chat = FallbackChatModel(p, [f1, f2], role="quick")

    assert chat.invoke("Analyze CSCO") == "ok:primary"
    p.invoke.assert_called_once()
    f1.invoke.assert_not_called()
    f2.invoke.assert_not_called()


def test_invoke_primary_recoverable_promotes_to_first_fallback():
    p = _model("primary", invoke_side_effects=[_Recoverable("429")])
    f1 = _model("fb1", invoke_side_effects=["ok:fb1"])
    f2 = _model("fb2")
    chat = FallbackChatModel(p, [f1, f2], role="quick")

    assert chat.invoke("Analyze CSCO") == "ok:fb1"
    p.invoke.assert_called_once()
    f1.invoke.assert_called_once()
    f2.invoke.assert_not_called()


def test_invoke_walks_through_to_last_fallback():
    p = _model("primary", invoke_side_effects=[_Recoverable("429")])
    f1 = _model("fb1", invoke_side_effects=[_Recoverable("429")])
    f2 = _model("fb2", invoke_side_effects=["ok:fb2"])
    chat = FallbackChatModel(p, [f1, f2], role="quick")

    assert chat.invoke("Analyze CSCO") == "ok:fb2"
    assert p.invoke.call_count == f1.invoke.call_count == f2.invoke.call_count == 1


def test_invoke_all_recoverable_raises_last_exception():
    p = _model("primary", invoke_side_effects=[_Recoverable("429-primary")])
    f1 = _model("fb1", invoke_side_effects=[_Recoverable("429-fb1")])
    f2 = _model("fb2", invoke_side_effects=[_Recoverable("429-fb2")])
    chat = FallbackChatModel(p, [f1, f2], role="quick")

    with pytest.raises(_Recoverable, match="429-fb2"):
        chat.invoke("Analyze CSCO")


def test_invoke_falls_through_on_openrouter_upstream_5xx_value_error():
    """OpenRouter wraps upstream 5xx in a 200 body; langchain raises a
    bare ValueError({'message': ..., 'code': 502}). Detect by code shape
    so the run survives an upstream provider hiccup the same way it
    survives a 429."""
    upstream_502 = ValueError(
        {"message": "Upstream error from OpenInference: ...", "code": 502}
    )
    p = _model("primary", invoke_side_effects=[upstream_502])
    f1 = _model("fb1", invoke_side_effects=["ok:fb1"])
    chat = FallbackChatModel(p, [f1], role="quick")

    assert chat.invoke("Analyze MSFT") == "ok:fb1"


def test_invoke_falls_through_on_nested_error_value_error():
    """Some langchain versions pass {'error': {...}} instead of the inner
    dict directly. Both shapes should be recognised."""
    nested = ValueError({"error": {"message": "rate limited", "code": 429}})
    p = _model("primary", invoke_side_effects=[nested])
    f1 = _model("fb1", invoke_side_effects=["ok:fb1"])
    chat = FallbackChatModel(p, [f1], role="quick")

    assert chat.invoke("Analyze MSFT") == "ok:fb1"


def test_invoke_does_not_fall_through_on_unrelated_value_error():
    """Pydantic / schema / user-code ValueErrors must propagate so the
    real bug surfaces. Only dicts with 429/5xx ``code`` count."""
    p = _model("primary", invoke_side_effects=[ValueError("bad schema")])
    f1 = _model("fb1", invoke_side_effects=["ok:fb1"])
    chat = FallbackChatModel(p, [f1], role="quick")

    with pytest.raises(ValueError, match="bad schema"):
        chat.invoke("Analyze MSFT")
    f1.invoke.assert_not_called()


def test_invoke_does_not_fall_through_on_4xx_value_error():
    """A 400/401/403 wrapped in a body still indicates a request bug, not
    an upstream provider issue — don't retry."""
    auth_400 = ValueError({"message": "auth fail", "code": 400})
    p = _model("primary", invoke_side_effects=[auth_400])
    f1 = _model("fb1", invoke_side_effects=["ok:fb1"])
    chat = FallbackChatModel(p, [f1], role="quick")

    with pytest.raises(ValueError):
        chat.invoke("Analyze MSFT")
    f1.invoke.assert_not_called()


def test_invoke_non_recoverable_does_not_fall_through():
    """Auth/schema errors would fail again on the next model — the wrapper
    must let them propagate immediately so the actual bug surfaces."""
    p = _model("primary", invoke_side_effects=[_Fatal("schema error")])
    f1 = _model("fb1", invoke_side_effects=["ok:fb1"])
    chat = FallbackChatModel(p, [f1], role="quick")

    with pytest.raises(_Fatal, match="schema error"):
        chat.invoke("Analyze CSCO")
    f1.invoke.assert_not_called()


# --- bind_tools chain preservation -------------------------------------


def test_bind_tools_preserves_fallback_chain():
    """After ``bind_tools``, the returned wrapper still falls through
    to the next bound model on recoverable errors."""
    p_bound = _model("primary-bound", invoke_side_effects=[_Recoverable("429")])
    f1_bound = _model("fb1-bound", invoke_side_effects=["ok:fb1-bound"])

    p = MagicMock(name="primary")
    p.bind_tools.return_value = p_bound
    f1 = MagicMock(name="fb1")
    f1.bind_tools.return_value = f1_bound

    chat = FallbackChatModel(p, [f1], role="market")
    bound = chat.bind_tools(["fake-tool"])

    assert isinstance(bound, FallbackChatModel)
    assert bound.invoke([{"role": "user", "content": "CSCO"}]) == "ok:fb1-bound"
    p.bind_tools.assert_called_once_with(["fake-tool"])
    f1.bind_tools.assert_called_once_with(["fake-tool"])


# --- with_structured_output chain --------------------------------------


def test_with_structured_output_preserves_chain_when_all_supported():
    p_struct = _model("primary-struct", invoke_side_effects=[_Recoverable("429")])
    f1_struct = _model("fb1-struct", invoke_side_effects=["ok:fb1-struct"])

    p = MagicMock(name="primary")
    p.with_structured_output.return_value = p_struct
    f1 = MagicMock(name="fb1")
    f1.with_structured_output.return_value = f1_struct

    chat = FallbackChatModel(p, [f1], role="structured_output")
    structured = chat.with_structured_output(object)

    assert isinstance(structured, FallbackChatModel)
    assert structured.invoke("Analyze CSCO") == "ok:fb1-struct"


def test_with_structured_output_drops_unsupported_fallbacks():
    """A fallback that raises NotImplementedError when wrapping (e.g. the
    deepseek-reasoner case) gets dropped from the chain so the run still
    succeeds — at the cost of less safety net for that branch."""
    p_struct = _model("primary-struct", invoke_side_effects=["ok:primary-struct"])
    p = MagicMock(name="primary")
    p.with_structured_output.return_value = p_struct

    f1 = MagicMock(name="fb1-no-struct")
    f1.with_structured_output.side_effect = NotImplementedError("unsupported")

    chat = FallbackChatModel(p, [f1], role="structured_output")
    structured = chat.with_structured_output(object)

    # Primary alone — no fallbacks survived, so we drop the wrapper layer.
    assert structured is p_struct


def test_with_structured_output_keeps_supported_fallbacks_only():
    """Mix of supported + unsupported: the supported ones stay in the chain."""
    p_struct = _model("primary-struct", invoke_side_effects=[_Recoverable("429")])
    p = MagicMock(name="primary")
    p.with_structured_output.return_value = p_struct

    f1 = MagicMock(name="fb1-no-struct")
    f1.with_structured_output.side_effect = NotImplementedError("unsupported")

    f2_struct = _model("fb2-struct", invoke_side_effects=["ok:fb2-struct"])
    f2 = MagicMock(name="fb2")
    f2.with_structured_output.return_value = f2_struct

    chat = FallbackChatModel(p, [f1, f2], role="structured_output")
    structured = chat.with_structured_output(object)

    assert isinstance(structured, FallbackChatModel)
    assert structured.invoke("Analyze CSCO") == "ok:fb2-struct"


# --- Misc --------------------------------------------------------------


def test_empty_fallbacks_rejected():
    with pytest.raises(ValueError, match="at least one fallback"):
        FallbackChatModel(_model("primary"), [])


def test_attribute_fallthrough_to_primary():
    """Non-overridden attribute access (e.g. ``model_name``) reaches the
    primary so existing introspection keeps working."""
    p = MagicMock(name="primary", temperature=0.7)
    p.model_name = "my-primary"
    f1 = MagicMock(name="fb1")
    chat = FallbackChatModel(p, [f1], role="quick")

    assert chat.model_name == "my-primary"
    assert chat.temperature == 0.7
