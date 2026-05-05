"""Provider-aware fallback wrapper for LangChain chat models.

When the configured primary model returns a recoverable upstream error —
``429`` rate limit, ``5xx``, request timeout, transport-level connection
failure — :class:`FallbackChatModel` retries the call against an ordered
list of fallback models. Auth, schema, and other 4xx errors are *not*
caught (they would fail again on the next model and mask the real bug).

The wrapper is intentionally duck-typed against the chat-model surface
the agent factories actually use (``invoke``, ``bind_tools``,
``with_structured_output``). Unknown attribute access falls through to
the primary model so existing introspection (``llm.model_name`` etc.)
keeps working.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Sequence, Tuple, Type

from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


def _recoverable_exceptions() -> Tuple[Type[BaseException], ...]:
    """Build the tuple of exception types that should trigger fallback.

    Imports each provider SDK lazily so this module stays importable even
    when only a subset of providers is installed. Falls back to ``Exception``
    if no provider exception types resolve, which keeps the wrapper useful
    in test environments where the real SDKs aren't present.
    """
    candidates = (
        "openai.RateLimitError",
        "openai.APITimeoutError",
        "openai.APIConnectionError",
        "openai.InternalServerError",
        # 404 from OpenRouter typically means "model was delisted" (e.g.
        # google/gemini-2.0-flash-exp:free was dropped on 2026-05-05).
        # Semantically "try a different model", same as a transient 5xx,
        # even though it's a 4xx code. Without this the entire run dies
        # whenever a single fallback in the chain becomes invalid.
        "openai.NotFoundError",
        "anthropic.RateLimitError",
        "anthropic.APITimeoutError",
        "anthropic.APIConnectionError",
        "anthropic.InternalServerError",
        "anthropic.NotFoundError",
        "google.api_core.exceptions.ResourceExhausted",
        "google.api_core.exceptions.ServiceUnavailable",
        "google.api_core.exceptions.DeadlineExceeded",
        "google.api_core.exceptions.NotFound",
    )
    excs: List[Type[BaseException]] = []
    for dotted in candidates:
        module_name, _, class_name = dotted.rpartition(".")
        try:
            module = __import__(module_name, fromlist=[class_name])
        except ImportError:
            continue
        cls = getattr(module, class_name, None)
        if isinstance(cls, type) and issubclass(cls, BaseException):
            excs.append(cls)
    return tuple(excs) if excs else (Exception,)


_RECOVERABLE = _recoverable_exceptions()


def _is_recoverable_upstream_error(exc: BaseException) -> bool:
    """Detect ``ValueError`` instances that wrap a provider-side 429/5xx.

    OpenRouter responds ``200 OK`` even when the upstream provider it
    proxies to fails — the error is delivered as a JSON ``error`` block
    in the response body. ``langchain_openai`` raises a bare
    ``ValueError(response_dict.get("error"))`` for those. The HTTP layer
    never sees a non-2xx so ``openai.APIStatusError`` doesn't fire, but
    the underlying problem is still upstream rate-limit / outage and a
    different model is the right next step.

    Matching on ``args[0]`` being a dict with a 429 or 5xx ``code`` is
    specific enough to avoid catching legitimate ``ValueError``s from
    Pydantic validation, schema mismatches, or user logic.
    """
    if not isinstance(exc, ValueError) or not exc.args:
        return False
    payload = exc.args[0]
    if not isinstance(payload, dict):
        return False
    # Some langchain versions pass the inner error dict directly; others
    # pass {"error": {...}}. Accept either.
    code = payload.get("code")
    if not isinstance(code, int):
        nested = payload.get("error")
        code = nested.get("code") if isinstance(nested, dict) else None
    return isinstance(code, int) and (code == 429 or 500 <= code < 600)


def _model_name(model: Any) -> str:
    """Best-effort identifier for log messages."""
    return (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or model.__class__.__name__
    )


class FallbackChatModel(Runnable):
    """LangChain-compatible chat-model facade with ordered fallback.

    On any recoverable upstream error (see ``_recoverable_exceptions``),
    the next model in the chain is tried. The last exception is re-raised
    when every model is exhausted so the caller still gets a meaningful
    traceback. Each fallback transition is logged at WARNING level with
    the role label, the failed model, and the exception type.
    """

    def __init__(
        self,
        primary: Any,
        fallbacks: Sequence[Any],
        *,
        role: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not fallbacks:
            raise ValueError("FallbackChatModel requires at least one fallback")
        self._primary = primary
        self._fallbacks = list(fallbacks)
        self._role = role

    def __repr__(self) -> str:
        prim = _model_name(self._primary)
        fbs = ", ".join(_model_name(m) for m in self._fallbacks)
        prefix = f"role={self._role!r} " if self._role else ""
        return f"FallbackChatModel({prefix}primary={prim!r}, fallbacks=[{fbs}])"

    def _try_each(self, op_name: str, fn) -> Any:
        """Run ``fn(model)`` against primary then each fallback in order.

        Returns the first successful result. Re-raises the last exception
        if every model fails on a recoverable error.
        """
        chain: List[Tuple[str, Any]] = [("primary", self._primary)]
        chain.extend(("fallback", m) for m in self._fallbacks)
        last_exc: Optional[BaseException] = None
        for kind, model in chain:
            try:
                return fn(model)
            except _RECOVERABLE as e:
                last_exc = e
                logger.warning(
                    "FallbackChatModel[%s]: %s on %s '%s' failed (%s); "
                    "trying next.",
                    self._role or "?",
                    op_name,
                    kind,
                    _model_name(model),
                    type(e).__name__,
                )
                continue
            except ValueError as e:
                # langchain raises ValueError when an OpenRouter response
                # wraps an upstream 5xx/429 in a 200 body. Match conservatively.
                if not _is_recoverable_upstream_error(e):
                    raise
                last_exc = e
                logger.warning(
                    "FallbackChatModel[%s]: %s on %s '%s' returned an "
                    "upstream error (%s); trying next.",
                    self._role or "?",
                    op_name,
                    kind,
                    _model_name(model),
                    e.args[0] if e.args else "?",
                )
                continue
        assert last_exc is not None
        logger.error(
            "FallbackChatModel[%s]: all %d models exhausted on %s; "
            "raising last exception.",
            self._role or "?",
            len(chain),
            op_name,
        )
        raise last_exc

    # --- Runnable interface ------------------------------------------------

    def invoke(self, input: Any, config: Any = None, **kwargs: Any) -> Any:
        return self._try_each("invoke", lambda m: m.invoke(input, config, **kwargs))

    # --- Chat-model facade -------------------------------------------------

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FallbackChatModel":
        """Bind ``tools`` to every model and return a new wrapper.

        The result still falls back on invoke; tool-bound LLMs are agnostic
        to the binding mechanism so the chain stays uniform.
        """
        primary_bound = self._primary.bind_tools(tools, **kwargs)
        fallbacks_bound = [m.bind_tools(tools, **kwargs) for m in self._fallbacks]
        return FallbackChatModel(primary_bound, fallbacks_bound, role=self._role)

    def with_structured_output(
        self, schema: Any, **kwargs: Any
    ) -> Any:
        """Wrap each model with provider-native structured output.

        Fallbacks that don't support structured output (e.g. deepseek-reasoner)
        are dropped from the chain with a warning rather than failing the
        whole run. If *no* fallback supports it, returns the primary's
        structured wrapper alone — the caller still gets structured output,
        just without the fallback safety net for this branch.
        """
        primary_struct = self._primary.with_structured_output(schema, **kwargs)
        fallbacks_struct: List[Any] = []
        for i, model in enumerate(self._fallbacks):
            try:
                fallbacks_struct.append(model.with_structured_output(schema, **kwargs))
            except (NotImplementedError, ValueError, TypeError) as e:
                logger.warning(
                    "FallbackChatModel[%s]: fallback %d ('%s') does not "
                    "support with_structured_output (%s); dropping.",
                    self._role or "?",
                    i + 1,
                    _model_name(model),
                    e,
                )
        if not fallbacks_struct:
            return primary_struct
        return FallbackChatModel(primary_struct, fallbacks_struct, role=self._role)

    # --- Introspection convenience -----------------------------------------

    @property
    def model_name(self) -> Optional[str]:
        return _model_name(self._primary)

    def __getattr__(self, name: str) -> Any:
        # Class-defined attributes resolve before __getattr__, so this only
        # fires for attributes the wrapper itself doesn't expose. Fall
        # through to the primary model for read-only introspection
        # (e.g. ``llm.model``, ``llm.temperature``).
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._primary, name)
