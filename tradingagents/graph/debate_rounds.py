"""Sequential execution with round-isolated evidence and equal speaking budgets.

Snapshots are persisted in graph state so a mid-round resume sees exactly the
same completed-round evidence. No extra model calls or parallel state reducers.
"""
from functools import wraps


def round_isolated_debater(node, key: str, speaker: str, participants: tuple[str, ...]):
    @wraps(node)
    def run(state):
        actual = state[key]
        count = actual["count"]
        if count % len(participants) == 0:
            snapshot = {k: v for k, v in actual.items() if k not in {"round_snapshot", "round_speakers"}}
            seen = []
        else:
            if "round_snapshot" not in actual or "round_speakers" not in actual:
                raise ValueError("Incomplete debate-round snapshot; start a fresh run")
            snapshot = actual["round_snapshot"]
            seen = list(actual["round_speakers"])
        if speaker not in participants or speaker in seen or len(seen) != count % len(participants):
            raise ValueError("Each debate participant must speak exactly once per round")

        visible = {**snapshot, "count": count}
        if key == "investment_debate_state":
            opponent = next(role for role in participants if role != speaker)
            visible["current_response"] = snapshot.get(
                f"current_{opponent}_response", snapshot.get(f"{opponent}_history", "")
            )
        output = node({**state, key: visible})
        contribution = output[key]
        response_key = "current_response" if key == "investment_debate_state" else f"current_{speaker}_response"
        argument = contribution[response_key]
        # Merge onto the actual persisted state, NOT the frozen prompt view:
        # otherwise later speakers would overwrite earlier same-round work.
        merged = {
            **actual,
            "history": actual.get("history", "") + "\n" + argument,
            f"{speaker}_history": actual.get(f"{speaker}_history", "") + "\n" + argument,
            f"current_{speaker}_response": argument,
            "count": count + 1,
            "round_snapshot": snapshot,
            "round_speakers": [*seen, speaker],
        }
        if key == "investment_debate_state":
            merged.update(current_response=argument, last_debater=speaker)
        else:
            merged["latest_speaker"] = contribution["latest_speaker"]
        return {**output, key: merged}

    return run
