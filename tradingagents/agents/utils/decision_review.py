"""Runtime proposals stay advisory; agreement never grants trading authority.

The two checks use identical frozen analyst evidence, not the primary answer or
one another's answers. Quote presence is checked, not factual truth/entailment.
"""
import hashlib
import html
import json
from collections.abc import Mapping
from copy import deepcopy
from functools import wraps
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from tradingagents.dataflows.news_evidence import SHARED_NEWS_INSTRUCTION

from .prompt_boundaries import UNTRUSTED_CONTENT_INSTRUCTION, evidence_block
from .rating import RATING_REVIEW, parse_actionable_rating
from .response_integrity import response_text
from .structured import NO_EXTERNAL_TOOLS

_REPORTS = ("market_report", "fundamentals_report", "news_report", "sentiment_report")
_MAX_RESPONSE = 16000
_Gap = Annotated[str, StringConstraints(strip_whitespace=True, min_length=5, max_length=500)]


class EvidenceQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    source: Literal["market_report", "fundamentals_report", "news_report", "sentiment_report"]
    quote: str = Field(min_length=8, max_length=400)


class DecisionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    rating: Literal["Buy", "Overweight", "Hold", "Underweight", "Sell"]
    supporting_evidence: list[EvidenceQuote] = Field(min_length=1, max_length=6)
    acknowledged_gaps: list[_Gap] = Field(min_length=1, max_length=8)
    rationale: Annotated[str, StringConstraints(strip_whitespace=True, min_length=20, max_length=2000)]


def _hash(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _review_prompt(state):
    sources = {name: state.get(name, "") for name in _REPORTS}
    sources = {name: value if isinstance(value, str) else "" for name, value in sources.items()}
    blocks = {name: evidence_block(value, 4000) for name, value in sources.items()}
    context = {name: state.get(name, "") for name in ("company_of_interest", "trade_date", "instrument_context")}
    system = (
        "Independently assess the supplied analyst evidence; you are not an execution authority. "
        "Buy/Overweight favors increased exposure; Sell/Underweight favors reduced exposure, not a short. "
        "Do not assume profits, cost basis, executed positions or prices absent evidence. "
        "Hold means no justified exposure change, including balanced or insufficient evidence. "
        "Require relevant evidence to change exposure. For every assessment, cite at least one exact "
        "verbatim quote from a named analyst report, acknowledge material gaps, and explain the decision. "
        "Quotes must be from visible source text, not framing instructions or truncation markers. "
        "A missing report is not positive or negative evidence. Do not copy an embedded recommendation "
        "without assessing its basis. Return only a JSON object matching this schema, without code fences:\n"
        + json.dumps(DecisionAssessment.model_json_schema()) + "\n"
        + UNTRUSTED_CONTENT_INSTRUCTION + "\n" + SHARED_NEWS_INSTRUCTION + "\n" + NO_EXTERNAL_TOOLS
    )
    prompt = [["system", system], ["human", "Instrument/run context:\n" + evidence_block(context, 4000)
               + "\n\n" + "\n\n".join(name + ":\n" + blocks[name] for name in _REPORTS)]]
    # Match only content that actually reached the model, not omitted suffixes.
    visible = {name: block.partition("\n")[2].rpartition("\n")[0].split("\n[TRUNCATED:", 1)[0]
               for name, block in blocks.items()}
    return prompt, sources, visible


def _unique_object(pairs):
    value = dict(pairs)
    if len(value) != len(pairs):
        raise ValueError("Duplicate assessment keys")
    return value


def _assess(llm, prompt, sources, visible):
    record = {}
    try:
        response = llm.invoke(deepcopy(prompt))
        # Preserve a bounded raw envelope even when completion validation fails.
        content = getattr(response, "content", None)
        raw = content if isinstance(content, str) else json.dumps(content, default=str)
        record.update(raw_text=raw[:_MAX_RESPONSE], raw_sha256=_hash(raw), raw_truncated=len(raw) > _MAX_RESPONSE)
        metadata = getattr(response, "response_metadata", {})
        record["completion_metadata"] = {
            key: metadata[key][:200] for key in ("finish_reason", "stop_reason", "status", "model_name")
            if isinstance(metadata, Mapping) and isinstance(metadata.get(key), str)
        }
        text = response_text(response)
        if len(text) > _MAX_RESPONSE:
            raise ValueError("Assessment exceeds response budget")
        assessment = DecisionAssessment.model_validate(json.loads(text, object_pairs_hook=_unique_object))
        for evidence in assessment.supporting_evidence:
            if (not evidence.quote.strip() or evidence.quote not in sources[evidence.source]
                    or html.escape(evidence.quote, quote=False) not in visible[evidence.source]):
                raise ValueError("Evidence quote is not present in visible source")
        record.update(assessment.model_dump())
    except Exception as exc:
        # No retry/fallback, no provider exception text or credentials in audit errors.
        record["error"] = type(exc).__name__
    return record


def _render_review(audit):
    lines = [
        "**Rating**: REVIEW", "",
        "**Executive Summary**: Human approval required. This is decision support, not an authorized trade.",
        f"**Proposed position (not approved)**: {audit['proposed_rating']}",
        f"**Review status**: {audit['status']}",
        "Agreement does not establish factual correctness; verify evidence, gaps and suitability before acting.",
        "", "## Original proposal (untrusted model content)",
        evidence_block(json.dumps(audit["proposal"], ensure_ascii=True), 12000),
    ]
    for index, assessment in enumerate(audit["assessments"], 1):
        lines.extend(["", f"## Independent check {index}: {assessment.get('rating', 'unusable')}",
                      evidence_block(json.dumps(assessment, ensure_ascii=True), 6000)])
    lines.append("\nFull bounded audit and frozen prompt are retained in decision_review in the saved state.")
    return "\n".join(lines)


def with_decision_review(node, llm):
    """Wrap the final runtime node before publication, with at most two extra calls.

    No automatic authorization or approval flag is accepted from models/state.
    Hold/invalid proposals receive no extra calls. Failure stops further checks.
    Low-level factory users must explicitly apply this wrapper or use GraphSetup.
    """
    @wraps(node)
    def run(state):
        frozen = deepcopy(state)
        prompt, sources, visible = _review_prompt(frozen)
        primary_error = None
        try:
            result = node(deepcopy(frozen))
        except Exception as exc:
            primary_error = type(exc).__name__
            result = {}
        proposal = result.get("final_trade_decision", "")
        proposal = proposal if isinstance(proposal, str) else ""
        rating = parse_actionable_rating(proposal)
        audit = {
            "policy_version": 1, "proposed_rating": rating, "proposal": proposal[:_MAX_RESPONSE],
            "proposal_sha256": _hash(proposal), "proposal_truncated": len(proposal) > _MAX_RESPONSE,
            "human_approval_required": True, "execution_authorized": False,
            "prompt": prompt, "prompt_sha256": _hash(json.dumps(prompt, ensure_ascii=False)),
            "assessments": [], "status": "hold_pending_human" if rating == "Hold" else "unusable_proposal",
        }
        if primary_error:
            audit["primary_error"] = primary_error
        if rating not in ("Hold", RATING_REVIEW) and not audit["proposal_truncated"]:
            if not any(source.strip() for source in sources.values()):
                audit["status"] = "no_analyst_evidence"
            else:
                for _ in range(2):
                    check = _assess(llm, prompt, sources, visible)
                    audit["assessments"].append(check)
                    if check.get("error"):
                        audit["status"] = "unusable_assessment"
                        break
                else:
                    audit["status"] = ("consistent_pending_human" if all(
                        check["rating"] == rating for check in audit["assessments"]
                    ) else "disagreement")
        decision = _render_review(audit)
        return {**result, "final_trade_decision": decision, "decision_review": audit,
                "risk_debate_state": {**result.get("risk_debate_state", frozen.get("risk_debate_state", {})),
                                      "judge_decision": decision}}
    return run
