# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

TradingAgents is a multi-agent LLM trading research framework built on LangGraph. A pipeline of specialized agents (analysts → researchers → research manager → trader → risk debators → portfolio manager) produces a Buy/Hold/Sell-style decision for a given ticker + date.

This repo also hosts a thin **screener pipeline** at the project root (`pipeline.py`, `config.py`, `screener/`) that pulls candidates from Finviz and feeds each through `TradingAgentsGraph` in a batch.

## Common commands

Install (editable, into a Python ≥ 3.10 env):
```bash
pip install -e .
```

Run the interactive Typer/Rich CLI:
```bash
tradingagents analyze                    # installed entrypoint
python -m cli.main analyze               # equivalent
tradingagents analyze --checkpoint       # resume after crash on next invocation with same ticker+date
tradingagents analyze --clear-checkpoints
```

Run as a library: see `main.py` (one-shot `TradingAgentsGraph(...).propagate(ticker, date)`).

Run the screener pipeline (batch run over Finviz candidates):
```bash
python pipeline.py
```
Knobs live in `config.py` at the repo root: `finviz_filters`, `max_tickers_per_run`, `cache_ttl_hours`, plus a `tradingagents_config` block that copies `DEFAULT_CONFIG` and applies screener-level overrides (LLM provider, per-role models). The screener writes one JSON per analysis to `results/by_ticker/{TICKER}/` with a relative symlink under `results/by_date/{YYYYMMDD}/` (Windows-safe stub fallback).

Tests (pytest, configured in `pyproject.toml`):
```bash
pytest                                   # full suite
pytest -k test_signal_processing         # single file/test
pytest -m unit                           # by marker (unit / integration / smoke)
pytest -m integration                    # live-network tests; some auto-skip on missing API keys
```
`tests/conftest.py` autouses a fixture that injects placeholder values for every supported LLM/data API key, so tests do not hang on missing credentials. The `mock_llm_client` fixture patches `tradingagents.llm_clients.factory.create_llm_client`.

There is no configured linter/formatter; do not invent one.

Docker:
```bash
docker compose run --rm tradingagents
docker compose --profile ollama run --rm tradingagents-ollama
```

## Architecture

### Pipeline shape (`tradingagents/graph/`)

`TradingAgentsGraph` (in `graph/trading_graph.py`) is the single orchestration entry point. On `__init__` it:
1. Calls `set_config(self.config)` so `dataflows.config` (a process-wide singleton) reflects the run's settings — agent tools read this lazily.
2. Optionally appends `"options"` to `selected_analysts` when `config["enable_options_analyst"]` is true.
3. Builds deep- and quick-thinking `BaseLLMClient`s via `llm_clients.factory.create_llm_client`.
4. Constructs a **role-keyed LLM map** (`self.role_llms`) via `_build_role_llms` — see "Per-role LLM routing" below.
5. Builds five `ToolNode`s keyed by analyst type (`market`, `social`, `news`, `fundamentals`, `options`).
6. Hands these to `GraphSetup.setup_graph(selected_analysts)` which wires the LangGraph `StateGraph` (see `graph/setup.py`). The state schema is `AgentState` from `agents/utils/agent_states.py`.

`propagate(ticker, date)` does several things in order before invoking the graph:
- **Resolves pending memory-log entries** for the ticker via `TradingMemoryLog` (fetches realised return + alpha vs SPY for prior decisions, generates reflections via `Reflector`, batch-writes back).
- **Pre-fetches macro and IV snapshots** (`_safe_macro_snapshot()` and `_safe_iv_snapshot(ticker)`), which the prompt-only risk debaters reference from `state["macro_snapshot"]` / `state["iv_snapshot"]` — they have no ToolNode of their own. Both fetches swallow errors and fall back to empty strings.
- **Optionally recompiles with a checkpointer** (`graph/checkpointer.py`) when `config["checkpoint_enabled"]` is true — per-ticker SQLite DBs live under `<data_cache_dir>/checkpoints/<TICKER>.db`. `thread_id` is `sha256(TICKER:date)[:16]`, so the same ticker+date resumes and a different date starts fresh.
- **Injects past_context, macro_snapshot, iv_snapshot** into the initial state via `Propagator.create_initial_state`.

After the graph returns, the final state is logged as JSON under `<results_dir>/<TICKER>/TradingAgentsStrategy_logs/full_states_log_<date>.json`, the decision is appended (pending) to the memory log, and the checkpoint is cleared.

### Agents (`tradingagents/agents/`)

Each subpackage exposes a `create_<role>(llm)` factory used by `GraphSetup`. The roles are:
- `analysts/` — market, social, news, fundamentals, **options** (call tools, produce a section report).
- `researchers/` — bull and bear (debate `max_debate_rounds` rounds).
- `managers/research_manager.py` — synthesises the debate into an investment plan.
- `trader/trader.py` — turns the plan into a Buy/Hold/Sell transaction proposal.
- `risk_mgmt/` — aggressive, neutral, conservative debators.
- `managers/portfolio_manager.py` — final decision; receives `past_context` from the memory log.

Analyst chain runs **sequentially** (not in parallel): `market → social → news → fundamentals → options → bull/bear debate → research_mgr → trader → risk debate → portfolio_mgr`. Each analyst's prompt now includes the dataflow tools relevant to its mandate (see "Data flow" below).

The Research Manager, Trader, and Portfolio Manager use **provider-native structured output** (json_schema for OpenAI/xAI, response_schema for Gemini, tool-use for Anthropic). Schemas live in `agents/schemas.py`; `agents/utils/structured.py` renders the parsed Pydantic instance back to the markdown shape the rest of the pipeline already consumes. Other agents stay free-form.

Internal debate agents always run in English (for reasoning quality). User-facing agents respect `config["output_language"]` via `agent_utils.get_language_instruction()`.

### Per-role LLM routing

`TradingAgentsGraph._build_role_llms` constructs `self.role_llms`, a dict of role-keyed LLMs that `GraphSetup` hands to each agent factory. Role keys:

| Role | Config key | Used by |
|---|---|---|
| `deep` | `deep_think_llm` | always populated (fallback for `structured_output`) |
| `quick` | `quick_think_llm` | fundamentals, bull, bear, trader; fallback for `quant`/`light` |
| `structured_output` | `structured_output_llm` | research manager, portfolio manager |
| `quant` | `quant_llm` | market, options, all 3 risk debaters |
| `light` | `light_llm` | social, news |

Empty role-config strings fall back: `structured_output → deep`, `quant → quick`, `light → quick`. The map is memoised by model string so two roles pointing at the same model share a single client.

### Data flow (`tradingagents/dataflows/`)

`@tool`-decorated wrappers in `agents/utils/*_tools.py` delegate to `dataflows.interface.route_to_vendor(method, *args)`. Routing reads `config["tool_vendors"]` first (per-tool override), then `config["data_vendors"]` (per-category default).

Active categories and vendors:

| Category | Vendors | Module(s) |
|---|---|---|
| `core_stock_apis` | yfinance, alpha_vantage | `y_finance.py`, `alpha_vantage_*.py` |
| `technical_indicators` | yfinance, alpha_vantage | same |
| `fundamental_data` | yfinance, alpha_vantage | same |
| `news_data` | yfinance, alpha_vantage | `yfinance_news.py`, `alpha_vantage_news.py` |
| `news_data` (insider) | yfinance, alpha_vantage, **sec** | `sec_insider.py` (Form 4 from SEC EDGAR) |
| `political_data` | **finnhub** (with Senate Stock Watcher fallback inside) | `congress_trades.py` |
| `options_data` | yfinance | `options_flow.py` (P/C ratios, max pain, walls, IVR) |
| `macro_data` | fred | `macro_data.py` (yields, curve, HY spread, USD) |
| `transcript_data` | motley_fool | `earnings_transcript.py` (LLM-scored sentiment) |
| `sector_data` | yfinance | `sector_analysis.py` (RS vs SPY, inter-market correlations) |

`route_to_vendor` falls through to the next vendor only on `AlphaVantageRateLimitError` — other exceptions bubble up. Vendor adapters return strings starting with `[` to indicate graceful failure (so the LLM can keep going); they should never raise.

When adding a new data tool: write the vendor adapter in `dataflows/`, register it in `VENDOR_METHODS` and category in `TOOLS_CATEGORIES` in `dataflows/interface.py`, write a tool wrapper in `agents/utils/<topic>_tools.py` that calls `route_to_vendor`, and re-export from `agents/utils/agent_utils.py` so analyst factories can import it.

API-response cache helper at `dataflows/_cache.py` (file-based JSON, keyed by source + arbitrary dict, explicit TTL per call site).

### LLM clients (`tradingagents/llm_clients/`)

`factory.create_llm_client(provider, model, base_url, **kwargs)` lazily imports the right backend. Providers in `_OPENAI_COMPATIBLE` (`openai`, `xai`, `deepseek`, `qwen`, `glm`, `ollama`, `openrouter`) all go through `OpenAIClient` with a provider-specific base URL. `anthropic`, `google`, and `azure` have dedicated clients. Provider-specific reasoning/thinking knobs are passed through from config: `google_thinking_level`, `openai_reasoning_effort`, `anthropic_effort`.

`OpenAIClient` defaults `max_retries=6` (overridable via `config["max_retries"]`). The openai SDK retries 429 and transient 5xx with exponential-backoff jitter — needed because OpenRouter free-tier models share upstream-provider quota pools that throttle aggressively.

`config["backend_url"]` defaults to **`None`** intentionally — each client falls back to its own provider default. Do not put an OpenAI URL in `DEFAULT_CONFIG`; it would leak into other providers (see commit `4016fd4`). The CLI sets it per-provider when the user picks one.

`model_catalog.py` is the single source of truth for which models appear in the CLI selector and is used by validators.

### Persistence

Everything user-state-shaped lives under `~/.tradingagents/` by default:
- `logs/` — full run state JSON (override: `TRADINGAGENTS_RESULTS_DIR`).
- `cache/checkpoints/<TICKER>.db` — LangGraph SqliteSaver (override base: `TRADINGAGENTS_CACHE_DIR`).
- `cache/api/<source>/<hash>.json` — file-based API-response cache (`dataflows/_cache.py`).
- `memory/trading_memory.md` — append-only decision log (override: `TRADINGAGENTS_MEMORY_LOG_PATH`).

The memory log uses an HTML-comment separator (`<!-- ENTRY_END -->`) as a hard delimiter that LLM prose cannot accidentally produce. Entries start as `pending` and are resolved in-place once price data is available.

The screener writes its own JSON output under `./results/by_ticker/<TICKER>/<YYYYMMDD_HHMMSS>_<TICKER>.json` with a relative symlink under `./results/by_date/<YYYYMMDD>/`.

## Coding conventions specific to this repo

- **Always validate ticker symbols** with `tradingagents.dataflows.utils.safe_ticker_component(ticker)` before interpolating into a filesystem path. Tickers come from CLI input *and* LLM tool calls, so they are attacker-influenced (commit `2c97bad`). The regex allows letters, digits, `.`, `-`, `_`, `^`.
- **Configuration is read through `dataflows.config.get_config()`**, not by passing the dict around. `TradingAgentsGraph.__init__` calls `set_config()` once; tools and agents read it lazily so the call site does not need to know about config.
- **Preserve exchange-qualified tickers verbatim** through tool calls (`CNC.TO`, `7203.T`, `0700.HK`). `agent_utils.build_instrument_context` is the canonical instruction; reuse it rather than rephrasing.
- **Don't hardcode provider URLs** in `DEFAULT_CONFIG` or share `base_url` across provider clients (see "LLM clients" above).
- **Vendor adapters never raise.** Wrap your pipeline in `try/except`, log via `logging.getLogger(__name__).warning`, and return `f"[<source> unavailable: {e}. Proceed with available data.]"`. The LLM tolerates these strings; an unhandled exception aborts the agent run.
- **Risk debaters are prompt-only** — no ToolNodes. To give them new context, pre-fetch in `_run_graph` and inject via `AgentState` (see how `macro_snapshot` / `iv_snapshot` are wired).
- The CLI loads `.env` and then `.env.enterprise` (without override), in that order. `main.py` and the screener `pipeline.py` only load `.env`.
- **Env vars at import time**: `default_config.py` resolves `FRED_API_KEY`, `FINNHUB_API_KEY`, `SEC_USER_AGENT` etc. at module-load time. Anything that imports it must `load_dotenv` first, or rely on the env being set externally. The screener `config.py` does this in the right order.

## Where things live (quick map)

```
pipeline.py                          # screener entry point: Finviz → TradingAgentsGraph → JSON
config.py                            # screener config (CONFIG dict + tradingagents_config overrides)
screener/
  finviz_filter.py                   # get_candidates() with TTL cache
  queue_manager.py                   # already_run_today / build_queue / mark_complete
  cache/                             # finviz_cache.json
results/                             # by_ticker/ + by_date/ — populated at runtime

main.py                              # minimal library-usage example
cli/main.py                          # `tradingagents analyze` Typer app + Rich UI loop
tradingagents/default_config.py      # DEFAULT_CONFIG + env-var overrides + per-role model keys
tradingagents/graph/
  trading_graph.py                   # TradingAgentsGraph (entry point) + _build_role_llms + macro/IV pre-fetch
  setup.py                           # GraphSetup — routes role_llms to agent factories
  checkpointer.py                    # per-ticker SQLite resume
  reflection.py / signal_processing.py / propagation.py / conditional_logic.py
tradingagents/agents/
  schemas.py                         # Pydantic structured-output schemas
  analysts/{market,social,news,fundamentals,options}_analyst.py
  researchers/{bull,bear}_researcher.py
  managers/{research,portfolio}_manager.py
  trader/trader.py
  risk_mgmt/{aggressive,neutral,conservative}_debator.py
  utils/agent_states.py              # AgentState (TypedDict) — graph state shape
  utils/agent_utils.py               # tool re-exports, language instr, ticker context
  utils/memory.py                    # TradingMemoryLog (append-only markdown log)
  utils/structured.py                # Pydantic → markdown render helpers
  utils/{core_stock,technical_indicators,fundamental_data,news_data}_tools.py
  utils/{political,options,macro,sector,transcript}_tools.py   # newer tool wrappers
tradingagents/dataflows/
  interface.py                       # route_to_vendor dispatcher + tool catalog
  _cache.py                          # file-based JSON API-response cache
  y_finance.py / alpha_vantage*.py   # original vendor adapters
  sec_insider.py                     # SEC EDGAR Form 4
  congress_trades.py                 # Finnhub + Senate Stock Watcher chain
  options_flow.py                    # yfinance option chains
  macro_data.py                      # FRED yields / curve / credit / USD
  earnings_transcript.py             # Motley Fool scrape + quick_thinking_llm sentiment scoring
  sector_analysis.py                 # SPDR sector RS vs SPY + inter-market correlations
  utils.py                           # safe_ticker_component, etc.
tradingagents/llm_clients/
  factory.py                         # create_llm_client (lazy provider import)
  model_catalog.py                   # MODEL_OPTIONS for the CLI
  openai_client.py                   # default max_retries=6 to ride OpenRouter throttles
  {anthropic,google,azure}_client.py
tests/                               # pytest, with conftest.py setting placeholder API keys
```
