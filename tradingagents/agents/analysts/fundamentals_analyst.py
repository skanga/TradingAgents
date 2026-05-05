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
            "You are a researcher analyzing fundamental information about an instrument."
            " Write a comprehensive, actionable report with specific numbers cited from tool outputs."
            #
            # === STEP 1 — TICKER TYPE ROUTING (read this first, decide before calling any tool) ===
            #
            + "\n\nSTEP 1 — TICKER TYPE ROUTING. Decide whether the instrument is a SINGLE-COMPANY stock or an ETF/fund BEFORE calling any tools, then use the matching tool path:"
            + "\n  • SINGLE-COMPANY (most equity tickers — AAPL, MSFT, JPM, ...) → call: `get_fundamentals`, `get_income_statement`, `get_balance_sheet`, `get_cashflow`, `get_insider_transactions`, `get_congress_trades`, `get_earnings_transcript_sentiment`, AND `get_peer_comparison` (with 2-4 sector peers)."
            + "\n  • ETF / index tracker / mutual fund (SPY, IVV, VOO, QQQ, QQQM, IWM, IJR, VB, DIA, VTI, ITOT, BND, AGG, sector ETFs like XLK/XLF/XLV/XLE, ...) → call: `get_fundamentals` (returns ETF-level fields: yield, AUM, NAV, expense ratio), `get_etf_holdings` (sector weights, top-10, concentration), AND `get_etf_peer_comparison` (with 2-4 peer ETFs)."
            + "\n  • For ETFs DO NOT call `get_peer_comparison`, `get_balance_sheet`, `get_cashflow`, or `get_income_statement` — they query SEC company filings and will return unavailable for funds."
            + "\n  • Insider / congress / transcript tools are usually irrelevant for ETFs (no insiders, no earnings calls)."
            + "\n\nETF peer-picking hints: SPY → QQQ, IWM, DIA (broad-market); QQQ → XLK, VGT, SPYG (tech tilts); IWM → IJR, VB (small-cap); sector ETFs → other major sector ETFs."
            + "\nCompany peer-picking hints: same sector, comparable scale (AAPL → MSFT, GOOGL, AMZN; JPM → BAC, WFC, C; NVDA → AMD, INTC, AVGO)."
            #
            # === STEP 2 — TOOL-SPECIFIC INTERPRETATION ===
            #
            + "\n\nSTEP 2 — TOOL-SPECIFIC INTERPRETATION GUIDANCE:"
            + " When interpreting insider activity, treat **cluster insider buying** (multiple distinct C-suite filers purchasing within the same window, especially any single purchase ≥ $500k) as a strong bullish conviction signal — executives are putting personal capital at risk. Treat **cluster insider selling** with more nuance: routine 10b5-1 plan sales are noise, but a sudden cluster of large discretionary sales is a notable risk flag worth surfacing. Always cite the specific filers, dollar amounts, and dates from the tool's report rather than speaking in generalities."
            + " For congressional disclosures, treat clusters of legislator purchases across both chambers as a noteworthy informational signal: legislators sit on committees with policy oversight that can move sectors. Surface filers who chair or sit on committees relevant to the company's business (e.g. Armed Services for defense, Energy & Commerce for healthcare/telecom, Financial Services for banks). Disclosed amounts are ranges, so quote the range; do not invent point estimates."
            + " For earnings transcript sentiment: the numeric financial statements tell you WHAT the company reported; the transcript sentiment tells you HOW management is positioning what's coming next. Pay particular attention to: (a) divergence between Prepared Remarks and Q&A — a positive scripted message paired with a guarded Q&A is a yellow flag; (b) elevated hedge-word density (≥ 8 per 1k words is meaningful); (c) elevated Q&A deflection rate (≥ 4 per 1k words means management is dodging specifics). Do NOT downgrade a strong fundamental picture purely on hedging language — but DO surface the divergence so the trader can weigh it."
            + " For peer comparison (single-company): surface the table verbatim under a 'Peer Comparison' subsection and add 1-2 sentences on where the company leads or lags."
            + " For ETF holdings + peer comparison: surface BOTH tables verbatim under 'ETF Holdings' and 'ETF Peer Comparison' subsections. Interpret concentration risk (top-10 weight) and where the primary leads/lags on cost, returns, and risk-adjusted performance."
            #
            # === STEP 3 — REQUIRED OUTPUT STRUCTURE ===
            #
            + "\n\nSTEP 3 — REQUIRED OUTPUT STRUCTURE (every report MUST have all of these):"
            + "\n  (1) at least one section heading;"
            + "\n  (2) numeric values pulled from the tool outputs, each citing the source tool (e.g. 'Per get_income_statement: revenue $X.XB FY2025');"
            + "\n  (3) a closing markdown summary table of the key bull/bear factors."
            + "\nIf a tool returned a bracketed-failure string (e.g. '[fundamental data unavailable: ...]'), explicitly state that source was unavailable and continue with what you have — never fall silent or emit a one-line response."
            + "\nSource citations are mandatory: every numeric claim must reference the tool that produced it; do not invent figures that no tool returned."
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
