from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_congress_trades,
    get_earnings_transcript_sentiment,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
)
from tradingagents.dataflows.config import get_config


def create_fundamentals_analyst(llm):
    def fundamentals_analyst_node(state):
        current_date = state["trade_date"]
        instrument_context = build_instrument_context(state["company_of_interest"])

        tools = [
            get_fundamentals,
            get_balance_sheet,
            get_cashflow,
            get_income_statement,
            get_insider_transactions,
            get_congress_trades,
            get_earnings_transcript_sentiment,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, and `get_insider_transactions` for SEC Form 4 insider buying/selling activity."
            + " When interpreting insider activity, treat **cluster insider buying** (multiple distinct C-suite filers purchasing within the same window, especially any single purchase ≥ $500k) as a strong bullish conviction signal — executives are putting personal capital at risk. Treat **cluster insider selling** with more nuance: routine 10b5-1 plan sales are noise, but a sudden cluster of large discretionary sales is a notable risk flag worth surfacing. Always cite the specific filers, dollar amounts, and dates from the tool's report rather than speaking in generalities."
            + " Also call `get_congress_trades` for STOCK Act disclosures from members of Congress. Treat congressional purchases — especially clusters of buyers across both chambers — as a noteworthy informational signal: legislators sit on committees with policy oversight that can move sectors. Surface filers who chair or sit on committees relevant to the company's business (e.g. Armed Services for defense, Energy & Commerce for healthcare/telecom, Financial Services for banks). Disclosed amounts are ranges, so quote the range; do not invent point estimates."
            + " Call `get_earnings_transcript_sentiment` to read the qualitative tone of management's most recent earnings call. The numeric financial statements tell you WHAT the company reported; the transcript sentiment tells you HOW management is positioning what's coming next. Pay particular attention to: (a) divergence between Prepared Remarks and Q&A — a positive scripted message paired with a guarded Q&A is a yellow flag; (b) elevated hedge-word density (≥ 8 per 1k words is meaningful); (c) elevated Q&A deflection rate (≥ 4 per 1k words means management is dodging specifics). Do NOT downgrade a strong fundamental picture purely on hedging language — but DO surface the divergence so the trader can weigh it."
            + get_language_instruction(),
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
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
