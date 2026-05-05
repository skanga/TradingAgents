from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from tradingagents.agents.utils.agent_utils import build_instrument_context, get_language_instruction, get_news
from tradingagents.agents.utils.prompts import load_prompt_template, render_prompt_template

The agent is now ``sentiment_analyst`` and aggregates Yahoo Finance news,
StockTwits cashtag streams, and Reddit posts into a single sentiment
report. Import from ``tradingagents.agents.analysts.sentiment_analyst``
going forward; this module will be removed in a future release.

See: https://github.com/TauricResearch/TradingAgents/issues/557
"""

import warnings as _warnings

        system_message = render_prompt_template(
            "social_media_analyst.md",
            {"language_instruction": get_language_instruction()},
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    load_prompt_template("tool_collaboration_system.md"),
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
            "sentiment_report": report,
        }

    return social_media_analyst_node
