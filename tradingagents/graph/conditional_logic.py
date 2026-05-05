# TradingAgents/graph/conditional_logic.py

import logging

from tradingagents.agents.utils.agent_states import AgentState

logger = logging.getLogger(__name__)


class ConditionalLogic:
    """Handles conditional logic for determining graph flow."""

    def __init__(
        self,
        max_debate_rounds=1,
        max_risk_discuss_rounds=1,
        max_tool_rounds_per_analyst=12,
    ):
        """Initialize with configuration parameters."""
        self.max_debate_rounds = max_debate_rounds
        self.max_risk_discuss_rounds = max_risk_discuss_rounds
        self.max_tool_rounds_per_analyst = max_tool_rounds_per_analyst

    def _should_continue_analyst(
        self, state: AgentState, analyst_name: str, tools_node: str, clear_node: str
    ) -> str:
        """Shared loop-termination logic for all analyst nodes.

        Routes to ``tools_node`` when the LLM's last message requested tools
        AND the per-analyst tool-call cap has not been hit; otherwise routes
        to ``clear_node`` to surrender whatever final answer the LLM has
        produced.

        The cap matters because each ``Msg Clear`` between analysts wipes
        the message buffer (see ``create_msg_delete``), so AIMessages with
        ``tool_calls`` in ``state["messages"]`` are exactly the tool rounds
        for the current analyst. Without a cap a flaky free-tier model can
        loop indefinitely until LangGraph's global recursion limit kills
        the entire run (observed: 33 rounds on AAPL Market Analyst).
        """
        messages = state["messages"]
        last_message = messages[-1]
        if not getattr(last_message, "tool_calls", None):
            return clear_node

        tool_rounds = sum(
            1 for m in messages
            if getattr(m, "tool_calls", None)
        )
        if tool_rounds >= self.max_tool_rounds_per_analyst:
            logger.warning(
                "%s hit tool-round cap (%d); forcing termination at %s.",
                analyst_name, self.max_tool_rounds_per_analyst, clear_node,
            )
            return clear_node
        return tools_node

    def should_continue_market(self, state: AgentState):
        return self._should_continue_analyst(
            state, "Market Analyst", "tools_market", "Msg Clear Market",
        )

    def should_continue_social(self, state: AgentState):
        return self._should_continue_analyst(
            state, "Social Analyst", "tools_social", "Msg Clear Social",
        )

    def should_continue_news(self, state: AgentState):
        return self._should_continue_analyst(
            state, "News Analyst", "tools_news", "Msg Clear News",
        )

    def should_continue_fundamentals(self, state: AgentState):
        return self._should_continue_analyst(
            state, "Fundamentals Analyst",
            "tools_fundamentals", "Msg Clear Fundamentals",
        )

    def should_continue_options(self, state: AgentState):
        return self._should_continue_analyst(
            state, "Options Analyst", "tools_options", "Msg Clear Options",
        )

    def should_continue_debate(self, state: AgentState) -> str:
        """Determine if debate should continue."""

        if (
            state["investment_debate_state"]["count"] >= 2 * self.max_debate_rounds
        ):  # 3 rounds of back-and-forth between 2 agents
            return "Research Manager"
        if state["investment_debate_state"]["current_response"].startswith("Bull"):
            return "Bear Researcher"
        return "Bull Researcher"

    def should_continue_risk_analysis(self, state: AgentState) -> str:
        """Determine if risk analysis should continue."""
        if (
            state["risk_debate_state"]["count"] >= 3 * self.max_risk_discuss_rounds
        ):  # 3 rounds of back-and-forth between 3 agents
            return "Portfolio Manager"
        if state["risk_debate_state"]["latest_speaker"].startswith("Aggressive"):
            return "Conservative Analyst"
        if state["risk_debate_state"]["latest_speaker"].startswith("Conservative"):
            return "Neutral Analyst"
        return "Aggressive Analyst"
