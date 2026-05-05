"""Markdown report emitter: BLUF + per-step sections, graceful on partial state."""

from screener.markdown_writer import _normalize_headings, render_markdown_report


def _full_state() -> dict:
    return {
        "company_of_interest": "AAPL",
        "trade_date": "2026-05-03",
        "macro_snapshot": "## Macroeconomic Environment\n**Backdrop**: FAVORABLE",
        "iv_snapshot": "## Implied Volatility Rank\n- IV Rank: 42/100",
        "market_report": "Market: bullish chart, RSI 58, MACD positive crossover.",
        "sentiment_report": "Sentiment: positive across X and Reddit.",
        "news_report": "News: solid Q earnings, raised guidance.",
        "fundamentals_report": "Fundamentals: 18% revenue growth, FCF expanding.",
        "options_report": "Options: P/C ratio 0.62 (bullish), IVR 42.",
        "investment_debate_state": {
            "bull_history": "Bull Analyst: Growth thesis intact.",
            "bear_history": "Bear Analyst: Margin pressure ahead.",
            "judge_decision": "Research Manager: Buy with size discipline.",
        },
        "investment_plan": "Invest 1% of book; entry on intraday pullback.",
        "trader_investment_decision": "FINAL TRANSACTION PROPOSAL: **BUY**.",
        "risk_debate_state": {
            "aggressive_history": "Aggressive Analyst: Up to 2% sizing.",
            "conservative_history": "Conservative Analyst: 0.5% only; HY widening.",
            "neutral_history": "Neutral Analyst: 1% is the right balance.",
            "judge_decision": "Portfolio Manager: Approve at 1%.",
        },
        "final_trade_decision": (
            "FINAL DECISION: **BUY** with 1% portfolio weight. "
            "Setup is constructive across the technical, fundamental, and "
            "options-flow lenses; macro backdrop is supportive; risk debate "
            "converged on moderate sizing."
        ),
    }


def test_bluf_at_top_with_signal_and_excerpt():
    md = render_markdown_report(
        ticker="AAPL", trade_date="2026-05-03",
        state=_full_state(), decision="BUY",
    )
    head = md.splitlines()[:8]
    assert head[0] == "# AAPL — Trading Agent Report"
    assert "Bottom Line Up Front" in md.split("##", 2)[1]  # first ## is BLUF
    assert "`BUY`" in md
    assert "Portfolio Manager rationale (excerpt)" in md
    # Excerpt is a blockquote
    assert "> FINAL DECISION" in md


def test_every_pipeline_step_rendered_in_order():
    md = render_markdown_report(
        ticker="AAPL", trade_date="2026-05-03",
        state=_full_state(), decision="BUY",
    )
    expected_order = [
        "Macro Backdrop",
        "Implied Volatility Context",
        "Market Analyst",
        "Social Media Analyst",
        "News Analyst",
        "Fundamentals Analyst",
        "Options Analyst",
        "Bull Researcher",
        "Bear Researcher",
        "Research Manager Verdict",
        "Investment Plan",
        "Trader Proposal",
        "Aggressive Risk Analyst",
        "Conservative Risk Analyst",
        "Neutral Risk Analyst",
        "Portfolio Manager Verdict",
    ]
    positions = [md.find(f"## {h}") for h in expected_order]
    assert all(p > 0 for p in positions), dict(zip(expected_order, positions))
    assert positions == sorted(positions), "sections must appear in pipeline order"
    # The risk-debate verdict must NOT be emitted as a separate section in
    # addition to "Portfolio Manager Verdict" — they're the same content.
    assert "Portfolio Manager Verdict (risk debate)" not in md
    # The legacy "Final Trade Decision" label is gone — the section is now
    # named after the agent that produced it.
    assert "## Final Trade Decision" not in md


def test_portfolio_verdict_falls_back_to_risk_verdict_when_canonical_missing():
    """If final_trade_decision is empty but the risk debate produced a
    judge_decision, render that under 'Portfolio Manager Verdict' so the
    user still sees a verdict — never both."""
    state = dict(_full_state())
    state["final_trade_decision"] = ""
    md = render_markdown_report(
        ticker="AAPL", trade_date="2026-05-03",
        state=state, decision="HOLD",
    )
    assert "## Portfolio Manager Verdict" in md
    assert "Portfolio Manager: Approve at 1%" in md
    # Still no duplicate "(risk debate)" section
    assert md.count("## Portfolio Manager Verdict") == 1


def test_no_decision_section_when_both_sources_missing():
    state = dict(_full_state())
    state["final_trade_decision"] = ""
    state["risk_debate_state"] = dict(state["risk_debate_state"])
    state["risk_debate_state"]["judge_decision"] = ""
    md = render_markdown_report(
        ticker="AAPL", trade_date="2026-05-03",
        state=state, decision="HOLD",
    )
    assert "## Portfolio Manager Verdict" not in md


def test_normalize_headings_shifts_to_min_level():
    body = "# Top\n## Sub\nbody"
    out = _normalize_headings(body, min_level=3)
    assert out == "### Top\n#### Sub\nbody"


def test_normalize_headings_no_op_when_already_at_or_below_target():
    body = "### Already deep\n#### Deeper\nbody"
    assert _normalize_headings(body, min_level=3) == body


def test_normalize_headings_skips_lines_in_fenced_code_blocks():
    """Python comments and shell prompts inside ``` fences must not be
    treated as markdown headings."""
    body = "# Real heading\n```python\n# this is a comment\n# also a comment\n```\n## Sub"
    out = _normalize_headings(body, min_level=3)
    # Real headings shifted; in-fence "#" lines untouched
    assert "### Real heading" in out
    assert "#### Sub" in out
    assert "# this is a comment" in out
    assert "# also a comment" in out


def test_normalize_headings_clamps_at_six_hashes():
    """Markdown only supports up to ``######`` — never produce ``#######``."""
    body = "##### Five\n###### Six"
    out = _normalize_headings(body, min_level=6)  # would shift +1, but clamp at 6
    assert "###### Five" in out
    assert "###### Six" in out


def test_normalize_headings_handles_body_with_no_headings():
    body = "Just a paragraph.\nNo headings here."
    assert _normalize_headings(body, min_level=3) == body


def test_render_normalizes_heading_levels_in_section_bodies():
    """Agents that emit ``# Their Title`` or ``## Their Title`` inside a
    section body must be shifted so the section's ``##`` wrapper stays
    the document's H2."""
    state = {
        "market_report": "# Market Heading\nSome body text\n## Sub-section",
    }
    md = render_markdown_report(
        ticker="X", trade_date="2026-05-03", state=state, decision="HOLD",
    )
    assert "## Market Analyst" in md
    # Agent's "# Market Heading" got shifted to "###"
    assert "### Market Heading" in md
    # And its "## Sub-section" got shifted to "####"
    assert "#### Sub-section" in md
    # The original "# Market Heading" line is gone (was shifted in place)
    assert "\n# Market Heading\n" not in md


def test_partial_state_skips_missing_sections():
    state = {
        "market_report": "Only the chart was analysed.",
        "final_trade_decision": "FINAL DECISION: HOLD.",
    }
    md = render_markdown_report(
        ticker="XYZ", trade_date="2026-05-03",
        state=state, decision="HOLD",
    )
    assert "## Market Analyst" in md
    assert "## Portfolio Manager Verdict" in md
    # Sections that don't exist in state should not appear
    assert "## Bull Researcher" not in md
    assert "## Macro Backdrop" not in md
    assert "## Options Analyst" not in md


def test_handles_trader_investment_plan_alias():
    """Raw final_state uses trader_investment_plan; _log_state writes
    trader_investment_decision. Either should populate the Trader section."""
    state_raw = {"trader_investment_plan": "Trader (raw): BUY"}
    md_raw = render_markdown_report(
        ticker="X", trade_date="2026-05-03", state=state_raw, decision="BUY",
    )
    assert "Trader Proposal" in md_raw
    assert "Trader (raw): BUY" in md_raw

    state_logged = {"trader_investment_decision": "Trader (logged): SELL"}
    md_logged = render_markdown_report(
        ticker="X", trade_date="2026-05-03", state=state_logged, decision="SELL",
    )
    assert "Trader (logged): SELL" in md_logged


def test_handles_completely_empty_state():
    md = render_markdown_report(
        ticker="ZZZ", trade_date="2026-05-03", state={}, decision="HOLD",
    )
    assert "# ZZZ — Trading Agent Report" in md
    assert "Bottom Line Up Front" in md
    assert "`HOLD`" in md
    # Footer always present
    assert "Generated by TradingAgents screener pipeline" in md
