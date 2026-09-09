# Human review and decision consistency

## Runtime contract

Standard `TradingAgentsGraph` / `GraphSetup` runs (including CLI and GUI) now finish with **REVIEW**, not an authorized trading signal. This applies even when the proposed position is Hold or all checks agree.

For an exposure-changing Portfolio Manager proposal (Buy, Overweight, Underweight or Sell), the final node makes at most two additional logical model invocations. Both receive identical frozen, bounded analyst reports and instrument/run context. They do **not** receive the primary Portfolio Manager answer, the trader/research proposals, risk advocacy, or one another's answers. The two checker prompts are identical to each other, not to the primary Portfolio Manager prompt.

Each usable check must provide:
- A five-tier proposed rating and a substantive rationale.
- At least one verbatim quote, with a named analyst-report source. The quote must occur in both the original report and its visible bounded prompt body.
- Explicit acknowledged gaps.

All three proposed ratings must match for `consistent_pending_human`. There is no majority vote and no forced Hold. Other audit statuses are `disagreement`, `unusable_assessment`, `no_analyst_evidence`, `unusable_proposal`, and `hold_pending_human`. Hold/invalid proposals, oversized primary proposals and absent analyst evidence incur no extra checks. A failed/unusable check stops further checks without a gate-level retry or fallback.

**Consistency is not factual verification.** A copied quote can be false, irrelevant or insufficient; agreeing models can repeat the same error. Underweight/Sell denotes reduced exposure, not a short or verified realized profit. Missing completion metadata remains inconclusive.

## Human approval is outside the application

`decision_review.human_approval_required` is always true and `execution_authorized` is always false. No model output or supplied state flag grants approval. There is deliberately **no automatic approval endpoint, checkbox, or mechanism to promote a candidate rating into a trade**.

Before acting outside the application, a human must inspect the full proposal, disagreements, evidence and gaps; verify original sources/as-of coverage, holdings/cost basis, suitability and execution constraints; and record their own decision. A useful independent review record identifies the run, prompt hash, reviewer, time, disposition, evidence checked, unresolved gaps, and any separately approved action. Do not manufacture this record with a model or paste it into evidence as an instruction.

Batch results retain `rating=REVIEW`, `trader_action=REVIEW`, and `review_required=true`. Analysis `status=success` means processing completed, not that a trade was approved. Allocation rejects REVIEW, flagged, missing/unknown-rated or failed results rather than treating them as Hold or emitting quantities. New unapproved runtime decisions do not become learnable trading memory, including simulation runs. Existing memory is not migrated or retroactively approved.

Low-level `create_portfolio_manager()` and frozen evaluation APIs remain advisory analysis primitives. Custom graphs must apply `with_decision_review()` or use `GraphSetup`; custom code that constructs its own raw ratings/allocation inputs remains caller-controlled. These boundaries do not restrict arbitrary external code or brokerage activity.

## Audits, bounds and cost

The graph state/checkpoint contains `decision_review`; checkpoint identity adds `decision_review_policy=1`. Canonical state logs and GUI archives retain it. Markdown report exports lead with a review warning and also write `5_portfolio/decision_review.json`.

The audit records the primary proposal, bounded raw checker content, parsed checks, selected completion metadata, sanitized exception classes, status, and frozen prompt/hash. Provider headers and exception messages are not copied into the new audit. Treat all prompts, reports and generated text as sensitive, untrusted material when storing or sharing them.

Bounds:
- Four analyst-report blocks and one instrument-context block: 4,000 escaped body characters each, plus fixed framing/schema instructions. Prefix omissions are marked; decisive later evidence may be omitted.
- Primary proposal and raw checker-content records: 16,000 characters each, with truncation flags and full-content hashes. Oversized answer text is unusable, not silently accepted after truncation.
- At most six quotes of 400 characters, eight gaps of 500 characters, and a 2,000-character rationale per usable check.

The prompt hash covers UTF-8 JSON of the stored role/content pairs, not the provider's wire serialization. Content hashes are mismatch checks, not signatures or proof of model identity. An audit can contain a prepared prompt with zero invocations; the assessment list records actual attempts.

The two-invocation limit is **additional to existing primary-generation/fallback calls** and is not a token, monetary, SDK retry or gateway-attempt ceiling. Restarting an interrupted final node can repeat calls because checks are not separately checkpointed. SDK/proxy retries, upstream routing, response caches and sampling independence remain outside this gate's guarantee. Standard production configuration has no application response cache enabled. Two prompt-isolated checks do not imply statistically independent samples.

## Remaining work and recommended sequence

The original 14-task implementation and the runtime review gate are implemented; effectiveness is not established. Remaining work is:

1. **Next: assemble and independently human-review genuine frozen cases.** Cover bullish, bearish and ambiguous/insufficient evidence before spending on further model evaluation.
2. **Then: authorize and measure the actual review gate.** Assess disagreement, grounding, unsupported claims, latency and cost using its own frozen prompts and captured responses. Previous measurements covered the underlying Portfolio Manager, not this gate. No live budget remains; commit/push approval does not authorize model or market-data requests.
3. **Optional operational extension: authenticated in-app approval.** Approval currently happens outside the application. An in-app workflow would need reviewer identity, an auditable disposition tied to an immutable run/proposal, authorization checks, and explicit rules for any downstream allocation or learning. It is not implemented or implicitly enabled by model agreement.

Bias reduction, universal injection resistance and trading accuracy remain unproven. Historical statements without reliable publication/as-published vintages remain explicitly unavailable; acquiring a trustworthy filing-vintage provider is a separate extension, not an unfinished implementation requirement of the original plan.

## Representative genuine-evidence validation — still pending

No genuine snapshots or independent adjudication were supplied for this follow-up. No live calls were made, and previous evaluation authorizations remain exhausted.

Before commissioning another measurement:
1. Select genuine frozen bullish, bearish and ambiguous/insufficient cases across relevant instruments and dates. Preserve original reports, source URLs/IDs, publication/retrieval timestamps, coverage gaps and available raw tool traces. Exclude credentials. Do not relabel synthetic fixtures as real evidence.
2. Have a human independently assess supporting and contradicting facts, missing evidence and defensible exposure decisions before inspecting model agreement. Record uncertainty; do not tune the model to a preferred rating.
3. Freeze candidate proposals and inputs, specify provider/model/settings, disclose caching/retries, and obtain a **new explicit call budget** before generating checks. A fresh full graph costs more than the two checker calls; budget the actual intended workflow.
4. Report disagreement/unusable/manual-review rates alongside independently assessed grounding, unsupported claims, latency and cost. Include repeated unchanged inputs. Agreement alone is not quality, causal bias reduction or trading accuracy.

`tradingagents.evaluation` still measures the original first-attempt Portfolio Manager prompt, **not this runtime review gate**. Earlier live synthetic-case recordings remain separate and reproducible; they do not validate the new gate. Gate-specific empirical validation must use its actual frozen inputs and captured responses, not silently reuse the old prompt experiment as evidence of effectiveness.

Offline regression entry point: `.venv/Scripts/python.exe -m pytest tests/test_decision_review.py -q`. Tests use scripted responses, mocks and LangChain's local fake model; they verify software contracts, not live model compliance or an authenticated human-approval workflow.
