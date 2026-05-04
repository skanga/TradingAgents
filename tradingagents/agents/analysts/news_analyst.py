from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_congress_trades,
    get_global_news,
    get_language_instruction,
    get_macro_environment,
    get_news,
)
from tradingagents.agents.utils.quality_guard import invoke_chain_with_quality_retry
from tradingagents.dataflows.config import get_config


def create_news_analyst(llm):
    def news_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_news,
            get_global_news,
            get_congress_trades,
            get_macro_environment,
        ]

        system_message = (
            "You are a news researcher tasked with analyzing recent news and trends over the past week. Please write a comprehensive report of the current state of the world that is relevant for trading and macroeconomics. Use the available tools: get_news(query, start_date, end_date) for company-specific or targeted news searches, and get_global_news(curr_date, look_back_days, limit) for broader macroeconomic news. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Also call `get_congress_trades` to surface recent STOCK Act disclosures from members of Congress. Congressional purchases can reflect policy-level information that intersects with your macro mandate (committee oversight, upcoming regulation, contract awards). Flag any cluster of legislator buys or sells, and note when filers serve on committees with jurisdiction over the company's sector."
            + " Always call `get_macro_environment` once per analysis to anchor your write-up in the current rates / yield-curve / credit-spread / USD regime. Lead with the FAVORABLE/NEUTRAL/UNFAVORABLE backdrop classification it returns, then connect the individual signals (e.g. curve inversion, HY spread widening, dollar strengthening) to specific company-level implications when relevant — multinationals are USD-sensitive, financials are curve-sensitive, leveraged credits are HY-spread sensitive, and so on. Do not contradict the tool's data; if you disagree with its classification, say so explicitly with reasoning."
            + """ Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."""
            + " REQUIRED OUTPUT STRUCTURE — your final report MUST include all of the following: (1) at least one section heading;"
            " (2) the macro backdrop classification (FAVORABLE/NEUTRAL/UNFAVORABLE) from get_macro_environment, surfaced near the top;"
            " (3) at least 3 specific events or stories with concrete dates pulled from the news tools (cite which tool produced each — e.g. 'Per get_news: ...');"
            " (4) the closing markdown summary table. If a news tool returned a bracketed-failure string ('[News unavailable: ...]'), state that explicitly and continue with what you have — never fall silent or emit a one-line response."
            " Source citations are mandatory: every event you cite must reference the tool that produced it; do not invent stories that no tool returned."
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
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. {instrument_context}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(instrument_context=instrument_context)

        chain = prompt | llm.bind_tools(tools)

        # Retry once on degenerate LLM output; substitute an "unavailable"
        # placeholder if the retry also fails. See quality_guard for details.
        final_message, report = invoke_chain_with_quality_retry(
            chain, state["messages"], analyst_label="News Analyst"
        )

        return {
            "messages": [final_message],
            "news_report": report,
        }

    return news_analyst_node
