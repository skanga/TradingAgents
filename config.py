"""Screener pipeline configuration.

Lives at the repo root alongside the ``tradingagents`` package — no
sys.path manipulation needed; the fork's ``DEFAULT_CONFIG`` imports directly.

The ``tradingagents_config`` key is a copy of ``DEFAULT_CONFIG`` with
screener-level overrides applied so batch runs can use a different
(typically cheaper / faster) LLM mix than ad-hoc fork usage without
mutating the fork's own defaults.
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent

# DEFAULT_CONFIG resolves several env vars at import time
# (FRED_API_KEY, FINNHUB_API_KEY, SEC_USER_AGENT, ...). Load .env first.
load_dotenv(_PROJECT_ROOT / ".env")

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402


def _build_tradingagents_config() -> dict:
    """Copy the fork's defaults and apply screener-level overrides.

    LLM choices live here so the screener can run on a different model mix
    than ad-hoc fork usage without editing fork files.
    """
    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = "openrouter"
    cfg["backend_url"] = "https://openrouter.ai/api/v1"
    cfg["deep_think_llm"] = "nvidia/nemotron-3-super-120b-a12b:free"
    cfg["quick_think_llm"] = "openai/gpt-oss-20b:free"
    # Per-role model keys established in the fork. Currently reserved
    # config keys — agent factories will start consuming them in a
    # follow-up; setting them here is a no-op until then.
    cfg["structured_output_llm"] = "openai/gpt-oss-120b:free"
    cfg["quant_llm"] = "qwen/qwen3-next-80b-a3b-instruct:free"
    cfg["light_llm"] = "meta-llama/llama-3.3-70b-instruct:free"
    return cfg


CONFIG: dict = {
    # Finviz filter labels match the keys finvizfinance expects (which mirror
    # the Finviz UI). The "+" prefix on Market Cap means "this bucket and above".
    "finviz_filters": {
        "Market Cap.":     "+Mid (over $2bln)",
        "Average Volume":  "Over 500K",
        "Country":         "USA",
        "Option/Short":    "Optionable",       # Finviz exposes optionable via the combined Option/Short filter
        "Price":           "Over $5",
    },
    "max_tickers_per_run": 15,
    "output_dir":          _PROJECT_ROOT / "results",
    "cache_path":          _PROJECT_ROOT / "screener" / "cache" / "finviz_cache.json",
    "cache_ttl_hours":     8,
    "tradingagents_config": _build_tradingagents_config(),
    # Exposed for diagnostics; not consumed by pipeline logic.
    "_project_root":       _PROJECT_ROOT,
}
