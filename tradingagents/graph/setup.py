# TradingAgents/graph/setup.py

from dataclasses import dataclass
from typing import Any, Callable, Dict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.utils.agent_states import AgentState

from .conditional_logic import ConditionalLogic


@dataclass(frozen=True)
class AnalystSpec:
    node_name: str
    clear_name: str
    tool_name: str
    create_node: Callable[[Any], Callable[..., dict]]
    continue_fn: Callable[[ConditionalLogic], Callable[..., str]]


ANALYST_SPECS: dict[str, AnalystSpec] = {
    "market": AnalystSpec(
        node_name="Market Analyst",
        clear_name="Msg Clear Market",
        tool_name="tools_market",
        create_node=create_market_analyst,
        continue_fn=lambda logic: logic.should_continue_market,
    ),
    "social": AnalystSpec(
        node_name="Social Analyst",
        clear_name="Msg Clear Social",
        tool_name="tools_social",
        create_node=create_sentiment_analyst,
        continue_fn=lambda logic: logic.should_continue_social,
    ),
    "news": AnalystSpec(
        node_name="News Analyst",
        clear_name="Msg Clear News",
        tool_name="tools_news",
        create_node=create_news_analyst,
        continue_fn=lambda logic: logic.should_continue_news,
    ),
    "fundamentals": AnalystSpec(
        node_name="Fundamentals Analyst",
        clear_name="Msg Clear Fundamentals",
        tool_name="tools_fundamentals",
        create_node=create_fundamentals_analyst,
        continue_fn=lambda logic: logic.should_continue_fundamentals,
    ),
}


class GraphSetup:
    """Handles the setup and configuration of the agent graph."""

    def __init__(
        self,
        quick_thinking_llm: Any,
        deep_thinking_llm: Any,
        tool_nodes: Dict[str, ToolNode],
        conditional_logic: ConditionalLogic,
    ):
        """Initialize with required components."""
        self.quick_thinking_llm = quick_thinking_llm
        self.deep_thinking_llm = deep_thinking_llm
        self.tool_nodes = tool_nodes
        self.conditional_logic = conditional_logic

    def setup_graph(
        self, selected_analysts=None
    ):
        """Set up and compile the agent workflow graph.

        Args:
            selected_analysts (list): List of analyst types to include. Options are:
                - "market": Market analyst
                - "social": Social media analyst
                - "news": News analyst
                - "fundamentals": Fundamentals analyst
        """
        if selected_analysts is None:
            selected_analysts = [
                "market",
                "social",
                "news",
                "fundamentals",
            ]

        if len(selected_analysts) == 0:
            raise ValueError("Trading Agents Graph Setup Error: no analysts selected!")

        unknown_analysts = [
            analyst for analyst in selected_analysts if analyst not in ANALYST_SPECS
        ]
        if unknown_analysts:
            unknown = unknown_analysts[0]
            allowed = ", ".join(ANALYST_SPECS)
            raise ValueError(
                f"Trading Agents Graph Setup Error: unknown analyst key {unknown!r}. "
                f"Allowed analyst keys: {allowed}"
            )

        # Create analyst nodes
        analyst_nodes = {}
        delete_nodes = {}
        tool_nodes = {}

        for analyst_type in selected_analysts:
            spec = ANALYST_SPECS[analyst_type]
            analyst_nodes[analyst_type] = spec.create_node(self.quick_thinking_llm)
            delete_nodes[analyst_type] = create_msg_delete()
            tool_nodes[analyst_type] = self.tool_nodes[analyst_type]

        # Create researcher and manager nodes
        bull_researcher_node = create_bull_researcher(self.quick_thinking_llm)
        bear_researcher_node = create_bear_researcher(self.quick_thinking_llm)
        research_manager_node = create_research_manager(self.deep_thinking_llm)
        trader_node = create_trader(self.quick_thinking_llm)

        # Create risk analysis nodes
        aggressive_analyst = create_aggressive_debator(self.quick_thinking_llm)
        neutral_analyst = create_neutral_debator(self.quick_thinking_llm)
        conservative_analyst = create_conservative_debator(self.quick_thinking_llm)
        portfolio_manager_node = create_portfolio_manager(self.deep_thinking_llm)

        # Create workflow
        workflow = StateGraph(AgentState)

        # Add analyst nodes to the graph
        for analyst_type, node in analyst_nodes.items():
            spec = ANALYST_SPECS[analyst_type]
            workflow.add_node(spec.node_name, node)
            workflow.add_node(spec.clear_name, delete_nodes[analyst_type])
            workflow.add_node(spec.tool_name, tool_nodes[analyst_type])

        # Add other nodes
        workflow.add_node("Bull Researcher", bull_researcher_node)
        workflow.add_node("Bear Researcher", bear_researcher_node)
        workflow.add_node("Research Manager", research_manager_node)
        workflow.add_node("Trader", trader_node)
        workflow.add_node("Aggressive Analyst", aggressive_analyst)
        workflow.add_node("Neutral Analyst", neutral_analyst)
        workflow.add_node("Conservative Analyst", conservative_analyst)
        workflow.add_node("Portfolio Manager", portfolio_manager_node)

        # Define edges
        # Start with the first analyst
        first_analyst = selected_analysts[0]
        workflow.add_edge(START, ANALYST_SPECS[first_analyst].node_name)

        # Connect analysts in sequence
        for i, analyst_type in enumerate(selected_analysts):
            spec = ANALYST_SPECS[analyst_type]
            current_analyst = spec.node_name
            current_tools = spec.tool_name
            current_clear = spec.clear_name

            # Add conditional edges for current analyst
            workflow.add_conditional_edges(
                current_analyst,
                spec.continue_fn(self.conditional_logic),
                [current_tools, current_clear],
            )
            workflow.add_edge(current_tools, current_analyst)

            # Connect to next analyst or to Bull Researcher if this is the last analyst
            if i < len(selected_analysts) - 1:
                next_analyst = ANALYST_SPECS[selected_analysts[i + 1]].node_name
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
