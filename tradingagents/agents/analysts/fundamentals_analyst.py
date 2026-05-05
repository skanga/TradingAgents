from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import (
    build_instrument_context,
    get_balance_sheet,
    get_cashflow,
    get_congress_trades,
    get_earnings_transcript_sentiment,
    get_etf_holdings,
    get_etf_peer_comparison,
    get_fundamentals,
    get_income_statement,
    get_insider_transactions,
    get_language_instruction,
    get_peer_comparison,
)
from tradingagents.agents.utils.quality_guard import invoke_chain_with_quality_retry
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
            get_peer_comparison,
            get_etf_holdings,
            get_etf_peer_comparison,
        ]

        system_message = (
            "You are a researcher tasked with analyzing fundamental information over the past week about a company. Please write a comprehensive report of the company's fundamental information such as financial documents, company profile, basic company financials, and company financial history to gain a full view of the company's fundamental information to inform traders. Make sure to include as much detail as possible. Provide specific, actionable insights with supporting evidence to help traders make informed decisions."
            + " Make sure to append a Markdown table at the end of the report to organize key points in the report, organized and easy to read."
            + " Use the available tools: `get_fundamentals` for comprehensive company analysis, `get_balance_sheet`, `get_cashflow`, and `get_income_statement` for specific financial statements, and `get_insider_transactions` for SEC Form 4 insider buying/selling activity."
            + " When interpreting insider activity, treat **cluster insider buying** (multiple distinct C-suite filers purchasing within the same window, especially any single purchase ≥ $500k) as a strong bullish conviction signal — executives are putting personal capital at risk. Treat **cluster insider selling** with more nuance: routine 10b5-1 plan sales are noise, but a sudden cluster of large discretionary sales is a notable risk flag worth surfacing. Always cite the specific filers, dollar amounts, and dates from the tool's report rather than speaking in generalities."
            + " Also call `get_congress_trades` for STOCK Act disclosures from members of Congress. Treat congressional purchases — especially clusters of buyers across both chambers — as a noteworthy informational signal: legislators sit on committees with policy oversight that can move sectors. Surface filers who chair or sit on committees relevant to the company's business (e.g. Armed Services for defense, Energy & Commerce for healthcare/telecom, Financial Services for banks). Disclosed amounts are ranges, so quote the range; do not invent point estimates."
            + " Call `get_earnings_transcript_sentiment` to read the qualitative tone of management's most recent earnings call. The numeric financial statements tell you WHAT the company reported; the transcript sentiment tells you HOW management is positioning what's coming next. Pay particular attention to: (a) divergence between Prepared Remarks and Q&A — a positive scripted message paired with a guarded Q&A is a yellow flag; (b) elevated hedge-word density (≥ 8 per 1k words is meaningful); (c) elevated Q&A deflection rate (≥ 4 per 1k words means management is dodging specifics). Do NOT downgrade a strong fundamental picture purely on hedging language — but DO surface the divergence so the trader can weigh it."
            + " For large-cap equity tickers (single companies), call `get_peer_comparison` once to benchmark against 2-4 sector peers on revenue, net income, gross profit, and operating income for the most recent complete fiscal year. Pick peers from the same sector and roughly comparable scale (e.g. AAPL → MSFT, GOOGL, AMZN; JPM → BAC, WFC, C; NVDA → AMD, INTC, AVGO). Surface the resulting table verbatim in your report under a 'Peer Comparison' subsection and add 1-2 sentences interpreting where the company leads or lags. SKIP this tool for ETFs and index trackers (SPY, QQQ, etc.) — peer comparison is not meaningful when the 'company' is itself a basket. If the tool returns a bracketed unavailable string, note it and continue."
            + " For ETFs, index trackers, or mutual fund tickers (SPY, QQQ, IWM, VTI, etc.), call `get_etf_holdings` once instead of `get_peer_comparison`. It returns asset-class breakdown, sector weights, top-10 holdings with concentration metric, and category/family metadata. Surface the resulting tables verbatim in your report under an 'ETF Holdings' subsection and add 1-2 sentences interpreting concentration risk (top-10 weight) and sector tilt vs. a broad-market benchmark. Only call `get_etf_holdings` when the ticker is clearly a fund — never for single-company stocks; it will return an unavailable string in that case."
            + " ALSO for ETFs, call `get_etf_peer_comparison` once with 2-4 peer ETFs to benchmark profile (AUM, expense ratio, yield, beta), returns (1M / 3M / YTD / 1Y), and risk (1Y vol + max drawdown). Pick peers with overlapping mandate but different tilt: SPY → QQQ, IWM, DIA (broad-market siblings); QQQ → XLK, VGT, SPYG (tech tilts); IWM → IJR, VB (small-cap siblings); sector ETFs → other major sector ETFs. Surface the resulting table verbatim under an 'ETF Peer Comparison' subsection and add 1-2 sentences calling out where the primary leads or lags on cost, returns, and risk-adjusted performance. Skip this for single-company stocks (use `get_peer_comparison` instead)."
            + " REQUIRED OUTPUT STRUCTURE — your final report MUST include all of the following, even if some sections are short:"
            " (1) at least one section heading; (2) numeric values pulled from the tool outputs (revenue, margins, FCF, EPS — cite the tool that supplied each figure, e.g. 'Per get_income_statement: revenue $X.XB FY2025'); (3) a closing markdown summary table with the key bull/bear factors. If a tool returned a bracketed-failure string (e.g. '[fundamental data unavailable: ...]'), explicitly state that source was unavailable and continue with what you have — never fall silent or emit a one-line response."
            " Source citations are mandatory: every numeric claim must reference the tool that produced it; do not invent figures that no tool returned."
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

        # Retry once on degenerate LLM output (the motivating case: a free-tier
        # model that returned literally "Call correct." instead of a real
        # Fundamentals report). If retry also fails, the helper substitutes an
        # honest "unavailable" placeholder so downstream debaters see coherent
        # signal instead of garbage.
        final_message, report = invoke_chain_with_quality_retry(
            chain, state["messages"], analyst_label="Fundamentals Analyst"
        )

        return {
            "messages": [final_message],
            "fundamentals_report": report,
        }

    return fundamentals_analyst_node
