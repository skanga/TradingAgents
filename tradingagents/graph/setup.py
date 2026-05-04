# TradingAgents/graph/setup.py

import logging
import time
from typing import Any, Callable, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import *
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic

logger = logging.getLogger(__name__)


def _with_checkpoint(name: str, node_fn: Callable) -> Callable:
    """Wrap a graph node to emit ENTER/EXIT checkpoint log lines.

    Default httpx INFO output drowns the run log in HTTP-request traces with
    no visible boundaries between agents. Wrapping each node here surfaces
    a single ``ENTER <name> | <ticker>`` line on entry and an
    ``EXIT  <name> | <ticker> | <elapsed>s`` line on exit, giving a grep-
    able outline of pipeline progress without per-agent code edits.

    Tool nodes and Msg-Clear nodes are wrapped too — every entry into an
    analyst's node typically follows a tool-node round-trip, so those
    checkpoints make tool-calling activity visible as well.
    """

    def wrapped(state, *args, **kwargs):
        ticker = "?"
        if isinstance(state, dict):
            ticker = state.get("company_of_interest") or "?"
        logger.info("ENTER %s | %s", name, ticker)
        t0 = time.monotonic()
        try:
            result = node_fn(state, *args, **kwargs)
        except Exception as e:
            logger.error(
                "FAIL  %s | %s | %.1fs | %s: %s",
                name,
                ticker,
                time.monotonic() - t0,
                type(e).__name__,
                e,
            )
            raise
        logger.info(
            "EXIT  %s | %s | %.1fs", name, ticker, time.monotonic() - t0
        )
        return result

    return wrapped


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        role_llms: Dict[str, Any],
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with the role-keyed LLM map and graph dependencies.

        ``role_llms`` must contain at minimum ``deep`` and ``quick``; the
        ``structured_output`` / ``quant`` / ``light`` keys fall back to
        ``deep`` / ``quick`` / ``quick`` respectively when absent.
        """
        self.role_llms = role_llms
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic
        # Convenience aliases for the common pair (used by analyst defaults
        # that don't fit any specialised role).
        self.quick_thinking_llm = role_llms["quick"]
        self.deep_thinking_llm = role_llms["deep"]

    def _llm(self, role: str, fallback_role: str = "quick") -> Any:
        """Resolve a role to its LLM, falling back if not configured."""
        return self.role_llms.get(role) or self.role_llms[fallback_role]

    def setup_graph(
        self, selected_analysts=["market", "social", "news", "fundamentals"]
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        # Per-role LLM routing (see TradingAgentsGraph._build_role_llms):
        #   quant  → market, options, risk debaters
        #   light  → social, news (surface-level reading)
        #   quick  → fundamentals, bull/bear, trader (mid-tier reasoning)
        #   structured_output → research mgr, portfolio mgr (structured output)
        quant_llm = self._llm("quant")
        light_llm = self._llm("light")
        quick_llm = self._llm("quick")
        structured_llm = self._llm("structured_output", fallback_role="deep")

        if "market" in selected_analysts:
            analyst_nodes["market"] = create_market_analyst(quant_llm)
            delete_nodes["market"] = create_msg_delete()
            tool_nodes["market"] = self.tool_nodes["market"]

        if "social" in selected_analysts:
            analyst_nodes["social"] = create_social_media_analyst(light_llm)
            delete_nodes["social"] = create_msg_delete()
            tool_nodes["social"] = self.tool_nodes["social"]

        if "news" in selected_analysts:
            analyst_nodes["news"] = create_news_analyst(light_llm)
            delete_nodes["news"] = create_msg_delete()
            tool_nodes["news"] = self.tool_nodes["news"]

        if "fundamentals" in selected_analysts:
            analyst_nodes["fundamentals"] = create_fundamentals_analyst(quick_llm)
            delete_nodes["fundamentals"] = create_msg_delete()
            tool_nodes["fundamentals"] = self.tool_nodes["fundamentals"]

        if "options" in selected_analysts:
            analyst_nodes["options"] = create_options_analyst(quant_llm)
            delete_nodes["options"] = create_msg_delete()
            tool_nodes["options"] = self.tool_nodes["options"]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(quick_llm)
        bear_researcher_node = create_bear_researcher(quick_llm)
        research_manager_node = create_research_manager(structured_llm)
        trader_node = create_trader(quick_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(quant_llm)
        neutral_analyst = create_neutral_debator(quant_llm)
        conservative_analyst = create_conservative_debator(quant_llm)
        portfolio_manager_node = create_portfolio_manager(structured_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph. Every node is wrapped with
        # _with_checkpoint so the run log carries grep-able ENTER/EXIT
        # boundaries between agents — see helper docstring at module top.
        for analyst_type, node in analyst_nodes.items():
            analyst_name = f"{analyst_type.capitalize()} Analyst"
            clear_name = f"Msg Clear {analyst_type.capitalize()}"
            tools_name = f"tools_{analyst_type}"
            workflow.add_node(analyst_name, _with_checkpoint(analyst_name, node))
            workflow.add_node(
                clear_name, _with_checkpoint(clear_name, delete_nodes[analyst_type])
            )
            workflow.add_node(
                tools_name, _with_checkpoint(tools_name, tool_nodes[analyst_type])
            )

        # Add other nodes
        workflow.add_node(
            "Bull Researcher", _with_checkpoint("Bull Researcher", bull_researcher_node)
        )
        workflow.add_node(
            "Bear Researcher", _with_checkpoint("Bear Researcher", bear_researcher_node)
        )
        workflow.add_node(
            "Research Manager", _with_checkpoint("Research Manager", research_manager_node)
        )
        workflow.add_node("Trader", _with_checkpoint("Trader", trader_node))
        workflow.add_node(
            "Aggressive Analyst", _with_checkpoint("Aggressive Analyst", aggressive_analyst)
        )
        workflow.add_node(
            "Neutral Analyst", _with_checkpoint("Neutral Analyst", neutral_analyst)
        )
        workflow.add_node(
            "Conservative Analyst",
            _with_checkpoint("Conservative Analyst", conservative_analyst),
        )
        workflow.add_node(
            "Portfolio Manager", _with_checkpoint("Portfolio Manager", portfolio_manager_node)
        )

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, f"{first_analyst.capitalize()} Analyst")

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            current_analyst = f"{analyst_type.capitalize()} Analyst"
            current_tools = f"tools_{analyst_type}"
            current_clear = f"Msg Clear {analyst_type.capitalize()}"

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                getattr(self.conditional_logic, f"should_continue_{analyst_type}"),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = f"{selected_analysts[i+1].capitalize()} Analyst"
                workflow.add_edge(current_clear, next_analyst)
            else:
                workflow.add_edge(current_clear, "Bull Researcher")

        # Add remaining edges
        workflow.add_conditional_edges(
            "Bull Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bear Researcher": "Bear Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_conditional_edges(
            "Bear Researcher",
            self.conditional_logic.should_continue_debate,
            {
                "Bull Researcher": "Bull Researcher",
                "Research Manager": "Research Manager",
            },
        )
        workflow.add_edge("Research Manager", "Trader")
        workflow.add_edge("Trader", "Aggressive Analyst")
        workflow.add_conditional_edges(
            "Aggressive Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Conservative Analyst": "Conservative Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Conservative Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Neutral Analyst": "Neutral Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )
        workflow.add_conditional_edges(
            "Neutral Analyst",
            self.conditional_logic.should_continue_risk_analysis,
            {
                "Aggressive Analyst": "Aggressive Analyst",
                "Portfolio Manager": "Portfolio Manager",
            },
        )

        workflow.add_edge("Portfolio Manager", END)

        return workflow
