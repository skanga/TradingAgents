import os

_TRADINGAGENTS_HOME = os.path.join(os.path.expanduser("~"), ".tradingagents")

DEFAULT_CONFIG = {
    "project_dir": os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
    "results_dir": os.getenv("TRADINGAGENTS_RESULTS_DIR", os.path.join(_TRADINGAGENTS_HOME, "logs")),
    "data_cache_dir": os.getenv("TRADINGAGENTS_CACHE_DIR", os.path.join(_TRADINGAGENTS_HOME, "cache")),
    "memory_log_path": os.getenv("TRADINGAGENTS_MEMORY_LOG_PATH", os.path.join(_TRADINGAGENTS_HOME, "memory", "trading_memory.md")),
    # Optional cap on the number of resolved memory log entries. When set,
    # the oldest resolved entries are pruned once this limit is exceeded.
    # Pending entries are never pruned. None disables rotation entirely.
    "memory_log_max_entries": None,

    # --- LLM settings ---
    "llm_provider":   "openrouter",
    "deep_think_llm": "nvidia/nemotron-3-super-120b-a12b:free",
    "quick_think_llm": "openai/gpt-oss-20b:free",
    # Provider-specific endpoint. None lets each provider's client pick its
    # own default; setting it here pins the OpenAI-compatible base URL used
    # by the OpenAI/xAI/DeepSeek/Qwen/GLM/Ollama/OpenRouter clients.
    "backend_url": "https://openrouter.ai/api/v1",

    # Per-role LLM overrides. Empty string falls back to deep_think_llm or
    # quick_think_llm via _build_role_llms. See graph/trading_graph.py for
    # the role-to-agent mapping.
    "structured_output_llm": "openai/gpt-oss-120b:free",   # Research Mgr, Portfolio Mgr
    "quant_llm":             "qwen/qwen3-next-80b-a3b-instruct:free",  # Market, Options, Risk
    "light_llm":             "meta-llama/llama-3.3-70b-instruct:free", # Social, Sector, Form4

    # Per-role fallback chains. Each entry is either a model string (uses the
    # run's ``llm_provider``) or a ``(provider, model)`` tuple / dict for
    # cross-provider fallback (e.g. OpenRouter free → paid OpenAI key). On
    # any recoverable upstream error (429, 5xx, timeout, transport drop) the
    # next entry is tried. Auth/schema/4xx errors are not caught — they
    # would fail again on the next model. Empty list disables fallback for
    # that role and preserves zero-config behaviour.
    "deep_think_llm_fallbacks":        [],
    "quick_think_llm_fallbacks":       [],
    "structured_output_llm_fallbacks": [],
    "quant_llm_fallbacks":             [],
    "light_llm_fallbacks":             [],

    # Provider-specific thinking configuration
    "google_thinking_level": None,      # "high", "minimal", etc.
    "openai_reasoning_effort": None,    # "medium", "high", "low"
    "anthropic_effort": None,           # "high", "medium", "low"

    # Checkpoint/resume: when True, LangGraph saves state after each node
    # so a crashed run can resume from the last successful step.
    "checkpoint_enabled": False,

    # Output language for analyst reports and final decision.
    # Internal agent debate stays in English for reasoning quality.
    "output_language": "English",

    # Debate and discussion settings
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "max_recur_limit": 100,

    # --- Data vendor configuration ---
    # Category-level configuration (default vendor for all tools in category)
    "data_vendors": {
        "core_stock_apis":    "yfinance",     # Options: alpha_vantage, yfinance
        "technical_indicators": "yfinance",   # Options: alpha_vantage, yfinance
        "fundamental_data":   "yfinance",     # Options: alpha_vantage, yfinance
        "news_data":          "yfinance",     # Options: alpha_vantage, yfinance, sec
        "political_data":     "finnhub",      # Options: finnhub (primary, House+Senate via FINNHUB_API_KEY); falls back to Senate Stock Watcher
        "options_data":       "yfinance",     # Options: yfinance
        "macro_data":         "fred",         # Options: fred
        "transcript_data":    "motley_fool",  # Options: motley_fool
        "sector_data":        "yfinance",     # Options: yfinance
    },
    # Tool-level configuration (takes precedence over category-level).
    # Use this to override the vendor for a single tool without changing
    # the whole category default.
    "tool_vendors": {
        # Route insider transactions to SEC EDGAR Form 4 by default; the
        # news_data category default (yfinance) still applies to news.
        "get_insider_transactions": "sec",
    },

    # --- Enhanced data sources: API keys / identifiers ---
    "fred_api_key":    os.environ.get("FRED_API_KEY", ""),
    "sec_user_agent":  os.environ.get("SEC_USER_AGENT", "TradingAgents contact@example.com"),
    "finnhub_api_key": os.environ.get("FINNHUB_API_KEY", ""),
    "lambda_finance_api_key": os.environ.get("LAMBDA_FINANCE_API_KEY", ""),

    # --- Feature flags for enhanced data sources ---
    # Each flag gates whether the framework auto-wires the corresponding
    # capability. The underlying tools are always registered with the LLM;
    # the flag controls whether the new options analyst joins the analyst
    # chain. (System-prompt enrichment for the existing analysts is added
    # during the per-section fill-in phase.)
    "enable_options_analyst":      True,
    "enable_congressional_trades": True,
    "enable_macro_data":           True,
    "enable_transcript_sentiment": True,
    "enable_sector_analysis":      True,
    # Use a local FinBERT model for transcript sentiment scoring.
    # When False (default), transcript sentiment is scored by the configured
    # quick_thinking_llm — no transformers/torch dependency required.
    # Setting True is currently a no-op until the optional FinBERT path is
    # implemented (would require ~2 GB of `transformers` + `torch`).
    "transcript_use_local_finbert": False,
}
