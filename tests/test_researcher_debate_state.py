from types import SimpleNamespace

from tradingagents.agents.researchers.bear_researcher import create_bear_researcher
from tradingagents.agents.researchers.bull_researcher import create_bull_researcher


def _state():
    return {
        "investment_debate_state": {
            "history": "",
            "bull_history": "",
            "bear_history": "",
            "current_response": "",
            "count": 0,
            "last_debater": None,
        },
        "market_report": "market",
        "sentiment_report": "sentiment",
        "news_report": "news",
        "fundamentals_report": "fundamentals",
    }


def test_bull_researcher_marks_last_debater():
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="buy thesis"))

    result = create_bull_researcher(llm)(_state())

    assert result["investment_debate_state"]["last_debater"] == "bull"


def test_bear_researcher_marks_last_debater():
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="sell thesis"))

    result = create_bear_researcher(llm)(_state())

    assert result["investment_debate_state"]["last_debater"] == "bear"
