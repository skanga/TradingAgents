# TradingAgents/graph/reflection.py

from typing import Any

from tradingagents.agents.utils.outcomes import format_outcome, score_outcome
from tradingagents.agents.utils.prompt_boundaries import (
    UNTRUSTED_CONTENT_INSTRUCTION,
    evidence_block,
)
from tradingagents.agents.utils.rating import parse_actionable_rating
from tradingagents.agents.utils.response_integrity import invoke_complete_text


class Reflector:
    """Handles reflection on trading decisions."""

    def __init__(self, quick_thinking_llm: Any):
        """Initialize the reflector with an LLM."""
        self.quick_thinking_llm = quick_thinking_llm
        self.log_reflection_prompt = self._get_log_reflection_prompt()

    def _get_log_reflection_prompt(self) -> str:
        """Concise prompt for reflect_on_final_decision (Phase B log entries).

        Produces 2-4 sentences of plain prose — compact enough to be re-injected
        into future agent prompts without bloating the context window.
        """
        return (
            "You are a trading analyst reviewing your own past decision now that the outcome is known.\n"
            "Write exactly 2-4 sentences of plain prose (no bullets, no headers, no markdown).\n\n"
            "Cover in order:\n"
            "1. Report the supplied directional accuracy and relative assessment separately; "
            "cite raw return for direction and alpha for relative performance. Do not override the classifications.\n"
            "2. Which part of the investment thesis held or failed? Returns alone cannot prove causes; "
            "say 'insufficient evidence' when no outcome evidence tests the thesis.\n"
            "3. One concrete lesson to apply to the next similar analysis.\n\n"
            "Be specific and terse. Your output will be stored verbatim in a decision log "
            "and re-read by future analysts, so every word must earn its place."
        )

    def reflect_on_final_decision(
        self,
        final_decision: str,
        raw_return: float,
        alpha_return: float,
        benchmark_name: str = "SPY",
        holding_days: int | None = None,
        evaluation_start: str | None = None,
        resolution_date: str | None = None,
    ) -> str:
        """Single reflection call on the final trade decision with outcome context.

        Used by Phase B deferred reflection. The decision supplies the original
        thesis, but returns alone cannot establish which causal claims held.
        ``benchmark_name`` is the label used for the alpha line (e.g. ``"SPY"``
        for US tickers, ``"^N225"`` for ``.T`` listings); defaults to SPY for
        callers that haven't been updated to thread the benchmark through.
        """
        outcome = score_outcome(parse_actionable_rating(final_decision), raw_return, alpha_return)
        outcome.update({
            "evaluation_sessions": holding_days,
            "evaluation_start": evaluation_start,
            "resolution_date": resolution_date,
            "benchmark_name": benchmark_name,
        })
        messages = [
            ("system", self.log_reflection_prompt + "\n" + UNTRUSTED_CONTENT_INSTRUCTION),
            (
                "human",
                (
                    format_outcome(outcome) + "\n"
                    + f"Raw return: {raw_return:+.1%}\n"
                    f"Alpha vs {benchmark_name}: {alpha_return:+.1%}\n\n"
                    f"Final Decision:\n{evidence_block(final_decision)}"
                ),
            ),
        ]
        return invoke_complete_text(self.quick_thinking_llm, messages)
