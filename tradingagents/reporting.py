"""Reusable Markdown report-tree writer for CLI and programmatic runs."""

from datetime import datetime
from pathlib import Path


def _metadata_block(metadata: dict[str, str] | None) -> str:
    if not metadata:
        return ""
    return "\n".join(f"**{key}**: {value}" for key, value in metadata.items()) + "\n\n"


def write_report_tree(
    final_state: dict,
    ticker: str,
    save_path,
    *,
    report_metadata: dict[str, str] | None = None,
    leading_sections: list[str] | None = None,
) -> Path:
    """Write per-team Markdown files and return the consolidated report path."""
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    sections = list(leading_sections or [])

    analysts_dir = save_path / "1_analysts"
    analyst_parts = []
    for key, filename, label in (
        ("market_report", "market.md", "Market Analyst"),
        ("sentiment_report", "sentiment.md", "Sentiment Analyst"),
        ("news_report", "news.md", "News Analyst"),
        ("fundamentals_report", "fundamentals.md", "Fundamentals Analyst"),
    ):
        if text := final_state.get(key):
            analysts_dir.mkdir(exist_ok=True)
            (analysts_dir / filename).write_text(text, encoding="utf-8")
            analyst_parts.append((label, text))
    if analyst_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in analyst_parts)
        sections.append(f"## I. Analyst Team Reports\n\n{content}")

    debate = final_state.get("investment_debate_state") or {}
    research_parts = []
    research_dir = save_path / "2_research"
    for key, filename, label in (
        ("bull_history", "bull.md", "Bull Researcher"),
        ("bear_history", "bear.md", "Bear Researcher"),
        ("judge_decision", "manager.md", "Research Manager"),
    ):
        if text := debate.get(key):
            research_dir.mkdir(exist_ok=True)
            (research_dir / filename).write_text(text, encoding="utf-8")
            research_parts.append((label, text))
    if research_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in research_parts)
        sections.append(f"## II. Research Team Decision\n\n{content}")

    if trader_plan := final_state.get("trader_investment_plan"):
        trading_dir = save_path / "3_trading"
        trading_dir.mkdir(exist_ok=True)
        (trading_dir / "trader.md").write_text(trader_plan, encoding="utf-8")
        sections.append(f"## III. Trading Team Plan\n\n### Trader\n{trader_plan}")

    risk = final_state.get("risk_debate_state") or {}
    risk_parts = []
    risk_dir = save_path / "4_risk"
    for key, filename, label in (
        ("aggressive_history", "aggressive.md", "Aggressive Analyst"),
        ("conservative_history", "conservative.md", "Conservative Analyst"),
        ("neutral_history", "neutral.md", "Neutral Analyst"),
    ):
        if text := risk.get(key):
            risk_dir.mkdir(exist_ok=True)
            (risk_dir / filename).write_text(text, encoding="utf-8")
            risk_parts.append((label, text))
    if risk_parts:
        content = "\n\n".join(f"### {name}\n{text}" for name, text in risk_parts)
        sections.append(f"## IV. Risk Management Team Decision\n\n{content}")

    if portfolio_decision := risk.get("judge_decision"):
        portfolio_dir = save_path / "5_portfolio"
        portfolio_dir.mkdir(exist_ok=True)
        (portfolio_dir / "decision.md").write_text(portfolio_decision, encoding="utf-8")
        sections.append(
            f"## V. Portfolio Manager Decision\n\n### Portfolio Manager\n{portfolio_decision}"
        )

    header = (
        f"# Trading Analysis Report: {ticker}\n\n"
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    output = save_path / "complete_report.md"
    output.write_text(
        header + _metadata_block(report_metadata) + "\n\n".join(sections),
        encoding="utf-8",
    )
    return output
