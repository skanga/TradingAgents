"""Evidence-first debate guidance and balanced judge input budgets."""
from .prompt_boundaries import MAX_EVIDENCE_CHARS, evidence_block

DEBATE_EVIDENCE_INSTRUCTION = (
    "\nEvidence takes precedence over advocacy: your assigned perspective is a hypothesis, "
    "not a required conclusion. Acknowledge strong contrary evidence and concede unsupported claims. "
    "State what would change your assessment. Do not manufacture disagreements or force a midpoint. "
    "When no opponent argument is supplied, make an independent opening from the analyst evidence; "
    "do not invent a rebuttal. Later contributions address only supplied completed-round arguments.\n"
)


def balanced_debate_evidence(debate: dict, roles: tuple[str, ...]) -> str:
    """Fixed role order, equal body budgets; chronology does not set prominence.

    Legacy/manual callers with only a combined history retain a bounded fallback.
    Role histories are rendered separately so one verbose role cannot crowd out
    another through prefix truncation of the combined transcript.
    """
    if not any(debate.get(f"{role}_history") for role in roles):
        return evidence_block(debate.get("history", ""))
    budget = MAX_EVIDENCE_CHARS // len(roles)
    return "\n\n".join(
        f"{role.title()} contributions:\n" + evidence_block(debate.get(f"{role}_history", ""), budget)
        for role in roles
    )
