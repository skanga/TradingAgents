# TradingAgents/graph/trading_graph.py

import logging
import os
from pathlib import Path
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional

import yfinance as yf

logger = logging.getLogger(__name__)


def _hashable(value: Any) -> Any:
    """Coerce nested lists/dicts to a hashable shape for use as dict keys.

    Fallback config entries can be plain strings, ``(provider, model)``
    tuples, or ``{"provider": ..., "model": ...}`` dicts. The role-LLM
    cache keys these tuples so identical chains reuse one client; this
    helper normalises the shape so dicts (mutable, unhashable) still work.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(v) for v in value)
    return value

from langgraph.prebuilt import ToolNode

from tradingagents.llm_clients import FallbackChatModel, create_llm_client

from tradingagents.agents import *
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.agents.utils.memory import TradingMemoryLog
from tradingagents.dataflows.utils import safe_ticker_component
from tradingagents.agents.utils.agent_states import (
    AgentState,
    InvestDebateState,
    RiskDebateState,
)
from tradingagents.dataflows.config import set_config

# Import the new abstract tool methods from agent_utils
from tradingagents.agents.utils.agent_utils import (
    get_stock_data,
    get_indicators,
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement,
    get_news,
    get_insider_transactions,
    get_global_news,
    get_congress_trades,
    get_options_summary,
    get_iv_rank,
    get_macro_environment,
    get_sector_relative_strength,
    get_intermarket_correlations,
    get_earnings_transcript_sentiment,
    get_peer_comparison,
    get_etf_holdings,
    get_etf_peer_comparison,
)

from .checkpointer import checkpoint_step, clear_checkpoint, get_checkpointer, thread_id
from .conditional_logic import ConditionalLogic
from .setup import GraphSetup
from .propagation import Propagator
from .reflection import Reflector
from .signal_processing import SignalProcessor


class TradingAgentsGraph:
    """Main class that orchestrates the trading agents framework."""

    def __init__(
        self,
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=False,
        config: Dict[str, Any] = None,
        callbacks: Optional[List] = None,
    ):
        """Initialize the trading agents graph and components.

        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary. If None, uses default config
            callbacks: Optional list of callback handlers (e.g., for tracking LLM/tool stats)
        """
        self.debug = debug
        self.config = config or DEFAULT_CONFIG
        self.callbacks = callbacks or []

        # Update the interface's config
        set_config(self.config)

        # Auto-include the options analyst when its feature flag is enabled.
        # Callers passing an explicit selected_analysts list can still opt out
        # by leaving the flag False; opting in is centralised on the flag.
        if (
            self.config.get("enable_options_analyst")
            and "options" not in selected_analysts
        ):
            selected_analysts = list(selected_analysts) + ["options"]

        # Create necessary directories
        os.makedirs(self.config["data_cache_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        # Initialize LLMs with provider-specific thinking configuration
        llm_kwargs = self._get_provider_kwargs()

        # Add callbacks to kwargs if provided (passed to LLM constructor)
        if self.callbacks:
            llm_kwargs["callbacks"] = self.callbacks

        # Deep/quick LLMs honour their own *_fallbacks lists. With no fallback
        # configured, ``_build_chat_with_fallbacks`` returns the bare chat
        # model so the existing zero-config behaviour is preserved.
        self.deep_thinking_llm = self._build_chat_with_fallbacks(
            primary=self.config["deep_think_llm"],
            fallbacks=self.config.get("deep_think_llm_fallbacks") or [],
            llm_kwargs=llm_kwargs,
            role="deep",
        )
        self.quick_thinking_llm = self._build_chat_with_fallbacks(
            primary=self.config["quick_think_llm"],
            fallbacks=self.config.get("quick_think_llm_fallbacks") or [],
            llm_kwargs=llm_kwargs,
            role="quick",
        )

        # Build the role-keyed LLM map. Empty role-specific config values
        # fall back to the deep/quick pair so this is safe regardless of
        # how much per-role customisation the user has configured.
        self.role_llms = self._build_role_llms(llm_kwargs)

        self.memory_log = TradingMemoryLog(self.config)

        # Create tool nodes
        self.tool_nodes = self._create_tool_nodes()

        # Initialize components
        self.conditional_logic = ConditionalLogic(
            max_debate_rounds=self.config["max_debate_rounds"],
            max_risk_discuss_rounds=self.config["max_risk_discuss_rounds"],
            max_tool_rounds_per_analyst=self.config.get(
                "max_tool_rounds_per_analyst", 12
            ),
        )
        self.graph_setup = GraphSetup(
            role_llms=self.role_llms,
            tool_nodes=self.tool_nodes,
            conditional_logic=self.conditional_logic,
        )

        self.propagator = Propagator()
        self.reflector = Reflector(self.quick_thinking_llm)
        self.signal_processor = SignalProcessor(self.quick_thinking_llm)

        # State tracking
        self.curr_state = None
        self.ticker = None
        self.log_states_dict = {}  # date to full state dict

        # Set up the graph: keep the workflow for recompilation with a checkpointer.
        self.workflow = self.graph_setup.setup_graph(selected_analysts)
        self.graph = self.workflow.compile()
        self._checkpointer_ctx = None

    def _get_provider_kwargs(self) -> Dict[str, Any]:
        """Get provider-specific kwargs for LLM client creation."""
        kwargs = {}
        provider = self.config.get("llm_provider", "").lower()

        if provider == "google":
            thinking_level = self.config.get("google_thinking_level")
            if thinking_level:
                kwargs["thinking_level"] = thinking_level

        elif provider == "openai":
            reasoning_effort = self.config.get("openai_reasoning_effort")
            if reasoning_effort:
                kwargs["reasoning_effort"] = reasoning_effort

        elif provider == "anthropic":
            effort = self.config.get("anthropic_effort")
            if effort:
                kwargs["effort"] = effort

        return kwargs

    def _create_tool_nodes(self) -> Dict[str, ToolNode]:
        """Create tool nodes for different data sources using abstract methods."""
        return {
            "market": ToolNode(
                [
                    # Core stock data tools
                    get_stock_data,
                    # Technical indicators
                    get_indicators,
                    # Sector/inter-market context
                    get_sector_relative_strength,
                    get_intermarket_correlations,
                ]
            ),
            "social": ToolNode(
                [
                    # News tools for social media analysis
                    get_news,
                ]
            ),
            "news": ToolNode(
                [
                    # News and insider information
                    get_news,
                    get_global_news,
                    get_insider_transactions,
                    # Macro and policy-adjacent signals
                    get_macro_environment,
                    get_congress_trades,
                ]
            ),
            "fundamentals": ToolNode(
                [
                    # Fundamental analysis tools
                    get_fundamentals,
                    get_balance_sheet,
                    get_cashflow,
                    get_income_statement,
                    # Insider, congressional, and earnings-call complements
                    get_insider_transactions,
                    get_congress_trades,
                    get_earnings_transcript_sentiment,
                    # Cross-ticker comparisons (must be in this list — the
                    # LLM is bound with these tools in the analyst factory,
                    # so any tool the LLM can pick must also live here or
                    # ToolNode rejects the call as "not a valid tool"
                    # — observed regression on the 2026-05-05 SPY trial).
                    get_peer_comparison,         # SEC fundamentals (single-company)
                    get_etf_holdings,            # ETF sector weights + top-10
                    get_etf_peer_comparison,     # ETF profile + returns + risk
                ]
            ),
            "options": ToolNode(
                [
                    get_options_summary,
                    get_iv_rank,
                ]
            ),
        }

    def _resolve_provider_model(
        self, entry: Any, default_provider: str,
    ) -> Optional[Tuple[str, str]]:
        """Normalise a fallback config entry to ``(provider, model)``.

        Accepts either a plain string (uses the run's ``llm_provider``) or
        a ``(provider, model)`` tuple/list/dict for cross-provider fallback.
        Returns ``None`` for empty/whitespace strings so they can be
        skipped without surprising behaviour.
        """
        if isinstance(entry, str):
            model = entry.strip()
            return (default_provider, model) if model else None
        if isinstance(entry, dict):
            provider = (entry.get("provider") or default_provider).strip()
            model = (entry.get("model") or "").strip()
            return (provider, model) if model else None
        if isinstance(entry, (tuple, list)) and len(entry) == 2:
            provider = (entry[0] or default_provider).strip()
            model = (entry[1] or "").strip()
            return (provider, model) if model else None
        logger.warning("Ignoring unrecognised fallback entry: %r", entry)
        return None

    def _build_chat_with_fallbacks(
        self,
        *,
        primary: Any,
        fallbacks: List[Any],
        llm_kwargs: Dict[str, Any],
        role: str,
    ) -> Any:
        """Build a chat model, optionally wrapped with fallback retry.

        With an empty ``fallbacks`` list this returns the bare LangChain
        chat model so the call site is indistinguishable from the
        previous ``create_llm_client(...).get_llm()`` pattern. With one
        or more fallbacks it returns a :class:`FallbackChatModel` that
        retries against each fallback on recoverable upstream errors.

        Cross-provider entries (``("openai", "gpt-5-mini")``) are honoured
        so a free-tier OpenRouter primary can fall back to a paid OpenAI
        key when the free pool is rate-limited.
        """
        provider = self.config["llm_provider"]
        base_url = self.config.get("backend_url")

        primary_pm = self._resolve_provider_model(primary, provider)
        if primary_pm is None:
            raise ValueError(f"Empty primary model for role '{role}'")
        primary_provider, primary_model = primary_pm
        primary_llm = create_llm_client(
            provider=primary_provider,
            model=primary_model,
            base_url=base_url if primary_provider == provider else None,
            **llm_kwargs,
        ).get_llm()

        fallback_llms: List[Any] = []
        for entry in fallbacks:
            pm = self._resolve_provider_model(entry, provider)
            if pm is None:
                continue
            fb_provider, fb_model = pm
            try:
                fallback_llms.append(
                    create_llm_client(
                        provider=fb_provider,
                        model=fb_model,
                        base_url=base_url if fb_provider == provider else None,
                        **llm_kwargs,
                    ).get_llm()
                )
            except Exception as e:
                # Don't fail the whole run if a fallback is misconfigured —
                # log and skip so the primary still works.
                logger.warning(
                    "Skipping %s fallback %s/%s (build failed: %s)",
                    role, fb_provider, fb_model, e,
                )
        if not fallback_llms:
            return primary_llm
        return FallbackChatModel(primary_llm, fallback_llms, role=role)

    def _build_role_llms(self, llm_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Construct a role-keyed LLM map.

        Roles:
          - ``deep`` / ``quick`` — the existing two-tier pair, always present.
          - ``structured_output`` — for the Research Manager and Portfolio
            Manager (they emit provider-native structured output).
          - ``quant`` — for the Market analyst, Options analyst, and the
            three risk debaters (technical / quantitative reasoning).
          - ``light`` — for the Social and News analysts (surface-level
            reading, no heavy reasoning).

        Empty role-specific config strings fall back to ``deep`` (for
        ``structured_output``) or ``quick`` (for ``quant`` and ``light``).
        Identical model strings across roles share a client to avoid
        duplicate provider connections.
        """
        deep_model = self.config.get("deep_think_llm")
        quick_model = self.config.get("quick_think_llm")

        # Memo by (primary_model, fallback_tuple); primes with the already-built
        # deep/quick LLMs so a role pointing at the same model + fallback list
        # reuses a single client. Fallback lists become hashable tuples for keying.
        deep_fbs = tuple(self.config.get("deep_think_llm_fallbacks") or [])
        quick_fbs = tuple(self.config.get("quick_think_llm_fallbacks") or [])
        cache: Dict[Tuple[Any, Any], Any] = {
            (deep_model, _hashable(deep_fbs)): self.deep_thinking_llm,
            (quick_model, _hashable(quick_fbs)): self.quick_thinking_llm,
        }

        def get_or_build(
            model: Any, fallbacks: List[Any], default: Any, role: str,
        ) -> Any:
            model_str = (model or "").strip() if isinstance(model, str) else ""
            if not model_str:
                return default
            key = (model_str, _hashable(tuple(fallbacks or [])))
            if key in cache:
                return cache[key]
            llm = self._build_chat_with_fallbacks(
                primary=model_str,
                fallbacks=fallbacks or [],
                llm_kwargs=llm_kwargs,
                role=role,
            )
            cache[key] = llm
            return llm

        return {
            "deep": self.deep_thinking_llm,
            "quick": self.quick_thinking_llm,
            "structured_output": get_or_build(
                self.config.get("structured_output_llm"),
                self.config.get("structured_output_llm_fallbacks") or [],
                default=self.deep_thinking_llm,
                role="structured_output",
            ),
            "quant": get_or_build(
                self.config.get("quant_llm"),
                self.config.get("quant_llm_fallbacks") or [],
                default=self.quick_thinking_llm,
                role="quant",
            ),
            "light": get_or_build(
                self.config.get("light_llm"),
                self.config.get("light_llm_fallbacks") or [],
                default=self.quick_thinking_llm,
                role="light",
            ),
        }

    def _safe_macro_snapshot(self) -> str:
        """Best-effort pre-fetch of the macro backdrop for the risk debate.

        Returns the empty string if the call raises so the debaters can
        condition on ``state['macro_snapshot']`` being truthy.
        """
        try:
            from tradingagents.dataflows.interface import route_to_vendor
            return route_to_vendor("get_macro_environment") or ""
        except Exception as e:
            logger.warning("macro snapshot pre-fetch failed: %s", e)
            return ""

    def _safe_iv_snapshot(self, ticker: str) -> str:
        """Best-effort pre-fetch of the IV-rank snapshot for the risk debate."""
        try:
            from tradingagents.dataflows.interface import route_to_vendor
            return route_to_vendor("get_iv_rank", ticker) or ""
        except Exception as e:
            logger.warning("IV snapshot pre-fetch failed: %s", e)
            return ""

    def _fetch_returns(
        self, ticker: str, trade_date: str, holding_days: int = 5
    ) -> Tuple[Optional[float], Optional[float], Optional[int]]:
        """Fetch raw and alpha return for ticker over holding_days from trade_date.

        Returns (raw_return, alpha_return, actual_holding_days) or
        (None, None, None) if price data is unavailable (too recent, delisted,
        or network error).
        """
        try:
            start = datetime.strptime(trade_date, "%Y-%m-%d")
            end = start + timedelta(days=holding_days + 7)  # buffer for weekends/holidays
            end_str = end.strftime("%Y-%m-%d")

            stock = yf.Ticker(ticker).history(start=trade_date, end=end_str)
            spy = yf.Ticker("SPY").history(start=trade_date, end=end_str)

            if len(stock) < 2 or len(spy) < 2:
                return None, None, None

            actual_days = min(holding_days, len(stock) - 1, len(spy) - 1)
            raw = float(
                (stock["Close"].iloc[actual_days] - stock["Close"].iloc[0])
                / stock["Close"].iloc[0]
            )
            spy_ret = float(
                (spy["Close"].iloc[actual_days] - spy["Close"].iloc[0])
                / spy["Close"].iloc[0]
            )
            alpha = raw - spy_ret
            return raw, alpha, actual_days
        except Exception as e:
            logger.warning(
                "Could not resolve outcome for %s on %s (will retry next run): %s",
                ticker, trade_date, e,
            )
            return None, None, None

    def _resolve_pending_entries(self, ticker: str) -> None:
        """Resolve pending log entries for ticker at the start of a new run.

        Fetches returns for each same-ticker pending entry, generates reflections,
        then writes all updates in a single atomic batch write to avoid redundant I/O.
        Skips entries whose price data is not yet available (too recent or delisted).

        Trade-off: only same-ticker entries are resolved per run.  Entries for
        other tickers accumulate until that ticker is run again.
        """
        pending = [e for e in self.memory_log.get_pending_entries() if e["ticker"] == ticker]
        if not pending:
            return

        updates = []
        for entry in pending:
            raw, alpha, days = self._fetch_returns(ticker, entry["date"])
            if raw is None:
                continue  # price not available yet — try again next run
            reflection = self.reflector.reflect_on_final_decision(
                final_decision=entry.get("decision", ""),
                raw_return=raw,
                alpha_return=alpha,
            )
            updates.append({
                "ticker": ticker,
                "trade_date": entry["date"],
                "raw_return": raw,
                "alpha_return": alpha,
                "holding_days": days,
                "reflection": reflection,
            })

        if updates:
            self.memory_log.batch_update_with_outcomes(updates)

    def propagate(self, company_name, trade_date):
        """Run the trading agents graph for a company on a specific date.

        When ``checkpoint_enabled`` is set in config, the graph is recompiled
        with a per-ticker SqliteSaver so a crashed run can resume from the last
        successful node on a subsequent invocation with the same ticker+date.
        """
        self.ticker = company_name

        # Resolve any pending memory-log entries for this ticker before the pipeline runs.
        self._resolve_pending_entries(company_name)

        # Recompile with a checkpointer if the user opted in.
        if self.config.get("checkpoint_enabled"):
            self._checkpointer_ctx = get_checkpointer(
                self.config["data_cache_dir"], company_name
            )
            saver = self._checkpointer_ctx.__enter__()
            self.graph = self.workflow.compile(checkpointer=saver)

            step = checkpoint_step(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )
            if step is not None:
                logger.info(
                    "Resuming from step %d for %s on %s", step, company_name, trade_date
                )
            else:
                logger.info("Starting fresh for %s on %s", company_name, trade_date)

        try:
            return self._run_graph(company_name, trade_date)
        finally:
            if self._checkpointer_ctx is not None:
                self._checkpointer_ctx.__exit__(None, None, None)
                self._checkpointer_ctx = None
                self.graph = self.workflow.compile()

    def _run_graph(self, company_name, trade_date):
        """Execute the graph and write the resulting state to disk and memory log."""
        # Initialize state — inject memory log context for PM.
        past_context = self.memory_log.get_past_context(company_name)

        # Section 8: pre-fetch macro + IV snapshots once so the prompt-only
        # risk debaters can reference them without needing tool-calling
        # plumbing of their own. Both helpers return graceful fallback
        # strings on failure, so the run is never blocked by network errors.
        macro_snapshot = self._safe_macro_snapshot()
        iv_snapshot = self._safe_iv_snapshot(company_name)

        init_agent_state = self.propagator.create_initial_state(
            company_name,
            trade_date,
            past_context=past_context,
            macro_snapshot=macro_snapshot,
            iv_snapshot=iv_snapshot,
        )
        args = self.propagator.get_graph_args()

        # Inject thread_id so same ticker+date resumes, different date starts fresh.
        if self.config.get("checkpoint_enabled"):
            tid = thread_id(company_name, str(trade_date))
            args.setdefault("config", {}).setdefault("configurable", {})["thread_id"] = tid

        if self.debug:
            trace = []
            for chunk in self.graph.stream(init_agent_state, **args):
                if len(chunk["messages"]) == 0:
                    pass
                else:
                    chunk["messages"][-1].pretty_print()
                    trace.append(chunk)
            final_state = trace[-1]
        else:
            final_state = self.graph.invoke(init_agent_state, **args)

        # Store current state for reflection.
        self.curr_state = final_state

        # Log state to disk.
        self._log_state(trade_date, final_state)

        # Store decision for deferred reflection on the next same-ticker run.
        self.memory_log.store_decision(
            ticker=company_name,
            trade_date=trade_date,
            final_trade_decision=final_state["final_trade_decision"],
        )

        # Clear checkpoint on successful completion to avoid stale state.
        if self.config.get("checkpoint_enabled"):
            clear_checkpoint(
                self.config["data_cache_dir"], company_name, str(trade_date)
            )

        return final_state, self.process_signal(final_state["final_trade_decision"])

    def _log_state(self, trade_date, final_state):
        """Log the final state to a JSON file."""
        self.log_states_dict[str(trade_date)] = {
            "company_of_interest": final_state["company_of_interest"],
            "trade_date": final_state["trade_date"],
            "macro_snapshot": final_state.get("macro_snapshot", ""),
            "iv_snapshot": final_state.get("iv_snapshot", ""),
            "market_report": final_state["market_report"],
            "sentiment_report": final_state["sentiment_report"],
            "news_report": final_state["news_report"],
            "fundamentals_report": final_state["fundamentals_report"],
            "options_report": final_state.get("options_report", ""),
            "investment_debate_state": {
                "bull_history": final_state["investment_debate_state"]["bull_history"],
                "bear_history": final_state["investment_debate_state"]["bear_history"],
                "history": final_state["investment_debate_state"]["history"],
                "current_response": final_state["investment_debate_state"][
                    "current_response"
                ],
                "judge_decision": final_state["investment_debate_state"][
                    "judge_decision"
                ],
            },
            "trader_investment_decision": final_state["trader_investment_plan"],
            "risk_debate_state": {
                "aggressive_history": final_state["risk_debate_state"]["aggressive_history"],
                "conservative_history": final_state["risk_debate_state"]["conservative_history"],
                "neutral_history": final_state["risk_debate_state"]["neutral_history"],
                "history": final_state["risk_debate_state"]["history"],
                "judge_decision": final_state["risk_debate_state"]["judge_decision"],
            },
            "investment_plan": final_state["investment_plan"],
            "final_trade_decision": final_state["final_trade_decision"],
        }

        # Save to file. Reject ticker values that would escape the
        # results directory when joined as a path component.
        safe_ticker = safe_ticker_component(self.ticker)
        directory = Path(self.config["results_dir"]) / safe_ticker / "TradingAgentsStrategy_logs"
        directory.mkdir(parents=True, exist_ok=True)

        log_path = directory / f"full_states_log_{trade_date}.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(self.log_states_dict[str(trade_date)], f, indent=4)

    def process_signal(self, full_signal):
        """Process a signal to extract the core decision."""
        return self.signal_processor.process_signal(full_signal)
