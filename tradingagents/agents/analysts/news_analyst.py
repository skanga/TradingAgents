from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from tradingagents.agents.utils.agent_utils import (
    get_global_news,
    get_instrument_context_from_state,
    get_language_instruction,
    get_macro_indicators,
    get_news,
    get_prediction_markets,
)
from tradingagents.agents.utils.prompt_boundaries import (
    UNTRUSTED_CONTENT_INSTRUCTION,
    evidence_block,
    evidence_history,
)
from tradingagents.agents.utils.response_integrity import (
    invoke_complete_tool_or_text,
    response_text,
)
from tradingagents.dataflows.news_evidence import SHARED_NEWS_INSTRUCTION


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        asset_type = state.get("asset_type", "stock")
        asset_label = "company" if asset_type == "stock" else "asset"
        instrument_context = get_instrument_context_from_state(state)

        tools = [
            get_news,
            get_global_news,
            get_macro_indicators,
            get_prediction_markets,
        ]

        system_message = (
            f"You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(ticker, start_date, end_date) for {asset_label}-specific news by ticker symbol, get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news, get_macro_indicators(indicator, curr_date, look_back_days) to ground macro commentary in actual data from FRED (e.g. 'cpi', 'core_pce', 'unemployment', 'fed_funds_rate', '10y_treasury', 'yield_curve'), and get_prediction_markets(topic, limit) for live market-implied probabilities of forward-looking events (e.g. 'Fed rate cut', 'recession 2026', geopolitical or sector events). Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + " " + SHARED_NEWS_INSTRUCTION
            + get_language_instruction()
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " Produce an evidence report, not a transaction proposal. Report supportive, adverse and neutral findings; do not force a directional conclusion."
                    " You have access to the following tools: {tool_names}."
                    " Today's date is {current_date}; treat it as 'now' for all analysis and tool-call date ranges.\n"
                    "{system_message}",
                ),
                ("human", "Instrument context:\n{instrument_context}"),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=str(system_message) + "\n" + UNTRUSTED_CONTENT_INSTRUCTION)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=evidence_block(instrument_context))

        chain = prompt | llm.bind_tools(tools)
        result = invoke_complete_tool_or_text(
            chain, evidence_history(state["messages"]), {tool.name for tool in tools}
        )

        report = ""

        if len(result.tool_calls) == 0:
            report = response_text(result)

        return {
            "messages": [result],
            "news_report": report,
        }

    return news_analyst_node
