from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_iv_rank,
    get_language_instruction,
    get_options_summary,
)


def create_options_analyst(llm):

    def options_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_options_summary,
            get_iv_rank,
        ]

        system_message = (
            "You are an options market analyst. Your job is to interpret institutional"
            " positioning signals from options-flow data. Analyze put/call ratios"
            " (volume and open-interest), open-interest call/put walls, max-pain"
            " strikes, unusual volume spikes (volume > 3x OI), and Implied Volatility"
            " Rank to assess whether smart money is positioned bullishly or bearishly"
            " into the upcoming expirations. Flag any unusual activity that may"
            " precede a significant price move. Contextualise IV Rank: > 50 means"
            " elevated fear (options expensive, hedges costly); < 20 means complacency"
            " (cheap hedges, possible vol expansion ahead). Your report feeds into the"
            " bull/bear research debate alongside the technical, fundamentals, news,"
            " and sentiment reports — be specific and actionable, not generic."
            " Append a Markdown table at the end summarising the key positioning"
            " signals."
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

        result = chain.invoke(state["messages"])

        report = ""

        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "options_report": report,
        }

    return options_analyst_node
