"""Deterministic exposure-call evaluation, distinct from executed trade P&L."""

import math

from tradingagents.agents.utils.rating import RATINGS_5_TIER


def score_outcome(rating: str, asset_return: float, excess_return: float) -> dict:
    """Score direction and relative exposure separately using unrounded returns.

    Sell/Underweight favor reducing exposure, not short-selling. Hold has no
    directional target. Relative assessment concerns asset vs benchmark exposure,
    not the return on cash proceeds. Flat uses only a floating-point tolerance.
    """
    if rating not in RATINGS_5_TIER:
        raise ValueError("Cannot score an invalid rating")
    benchmark_return = asset_return - excess_return
    if not all(math.isfinite(value) for value in (asset_return, excess_return, benchmark_return)):
        raise ValueError("Outcome returns must be finite")
    direction = 1 if rating in ("Buy", "Overweight") else -1
    directional = relative = "not_applicable"
    if rating != "Hold":
        directional = (
            "flat" if math.isclose(asset_return, 0.0, abs_tol=1e-12)
            else "correct" if direction * asset_return > 0 else "incorrect"
        )
        relative = (
            "flat" if math.isclose(excess_return, 0.0, abs_tol=1e-12)
            else "favorable" if direction * excess_return > 0 else "unfavorable"
        )
    return {
        "scoring_version": 1,
        "asset_return": asset_return,
        "benchmark_return": benchmark_return,
        "excess_return": excess_return,
        "directional_accuracy": directional,
        "relative_assessment": relative,
    }


def format_outcome(outcome: dict) -> str:
    """Keep deterministic metrics visible alongside (not hidden in) LLM lessons."""
    return (
        f"Directional accuracy: {outcome['directional_accuracy']}\n"
        f"Relative assessment: {outcome['relative_assessment']}\n"
        f"Benchmark return: {outcome['benchmark_return']:+.1%}\n"
        f"Evaluation: {outcome.get('evaluation_start') or 'unknown start'} to "
        f"{outcome.get('resolution_date') or 'unknown end'}; "
        f"{outcome.get('evaluation_sessions') or 'unknown'} asset trading sessions; "
        f"benchmark {outcome.get('benchmark_name') or 'unspecified'}.\n"
        "These are exposure-call assessments, not realized trade P&L. "
        "Sell/Underweight means reduce or avoid exposure, not an executed short. "
        "Hold has no directional target."
    )
