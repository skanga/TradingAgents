# Frozen-evidence evaluation

This harness measures **Portfolio Manager prompt sensitivity**, not trading accuracy or causal attack success. It does not fetch market data, run graph tools, write trading memory, or execute orders.

Standard runtime graphs now additionally enforce mandatory human review and two bounded checks for exposure-changing proposals; see [DECISION_REVIEW.md](DECISION_REVIEW.md). **This harness does not exercise that gate.** Its prior recordings and metrics must not be presented as gate-effectiveness evidence. The gate was verified offline (32 new cases; full Python suite 1193 passed / 3 skipped / 73 passed subtests), with no new live requests. Genuine frozen snapshots, independent adjudication and a new explicit call budget remain necessary for empirical validation.

## Offline workflow

Use the project Python (`.venv/Scripts/python.exe` on Windows):

```powershell
.venv/Scripts/python.exe -m tradingagents.evaluation plan examples/evaluation/balanced_case.json --repeats 2 --schedule-seed 17 --output plan.json
.venv/Scripts/python.exe -m tradingagents.evaluation replay plan.json responses.jsonl --output metrics.json
```

The example is **entirely synthetic**, not investment evidence. `plan` and `replay` never construct a model client. Output paths must be new: artifacts are not overwritten.

A case contains:
- `case_id`: a short identifier.
- `state`: frozen `company_of_interest`, `trade_date`, `instrument_context`, `investment_plan`, `trader_investment_plan`, optional `past_context`, and `risk_debate_state` with nonempty `aggressive_history`, `conservative_history`, and `neutral_history`.
- `injections`: optional objects with distinct `id`, `field`, and `text`. Allowed fields: `instrument_context`, `past_context`, `investment_plan`, `trader_investment_plan`, or `risk_debate_state.<role>_history`. `control` is reserved.

Use reviewed report/state snapshots, not an instruction to retrieve fresh data. Save the original case alongside its plan. Plans contain the actual bounded prompts and a case hash; they do not embed the full original state.

Each plan crosses all six role permutations with the control and each injection separately, for each repeat. An injection is prepended to its evidence field **before** the same escaping/bounding used in production. The default production role order is the order-control reference. The scheduling seed only shuffles request order; it is not a provider/model seed.

Limits: 1 MB per case, 100,000 characters per evidence field, 2,000 characters per injection, ten injections, ten repeats, and at most 200 tasks. Artifact reads are capped at 32 MB. These are not token or spend limits.

## Recorded response format

`responses.jsonl` contains one JSON object per attempted task:

```json
{"task_id":"<copied from plan>","prompt_sha256":"<copied from plan>","content":"Rating: Hold\nNEWS-0000000000000001 remains uncertain.","response_metadata":{"finish_reason":"stop"},"backend":{"provider":"recorded","model":"example"}}
```

Preserve completion metadata; dropping it can hide truncation. Optional `tool_calls`, `invalid_tool_calls`, and protocol-related `additional_kwargs` are scored using the runtime response validator. Transport failures use an `error` field instead of fabricated response prose. Missing tasks need no record. A replay refuses duplicate IDs, unknown IDs, mismatched prompt hashes, mixed backend descriptions, and malformed/incomplete factorial plans. If all backend descriptions are absent, the backend is explicitly unspecified.

Prompt hashes are SHA-256 of the exact UTF-8 prompt. Other artifact/identity hashes use canonical JSON. Hashes detect mismatches, **not malicious edits followed by recomputing hashes**; they are not signatures. Review a plan before permitting live execution.

## Optional live execution

**Do not run this merely to test the CLI. It can incur provider charges and sends the frozen evidence to that provider.** Separate approval and configured credentials are required. No `.env` file or credentials should be added to a case or artifact.

```powershell
.venv/Scripts/python.exe -m tradingagents.evaluation live plan.json --provider openai --model YOUR_MODEL --allow-live --max-calls 24 --output responses.jsonl
```

The explicit opt-in and sufficient call budget are checked before client construction. The CLI asks the existing client adapter for `max_retries=0` and `timeout=60`. There is one invocation per task, no tool binding, no structured-output fallback, and no harness retry. Provider/proxy behavior can still differ; a call budget is not a monetary/token ceiling. Other generation settings use adapter/provider defaults and are not seeded by this harness.

Each completed attempt is flushed immediately. An interrupted file can be replayed, with unfinished tasks reported as missing. There is no automatic live resume or merging of independent runs. Use a new output path for another run. Provider/model/configuration descriptions are recorded; API exception messages are not persisted. Treat prompts and responses as potentially sensitive and untrusted. Review artifacts before sharing them.

The Python API also exposes `build_plan`, `replay`, and `run_live(plan, llm_factory, allow_live=True, max_calls=..., backend=..., on_record=...)`. Custom factories must disclose their settings/retries in `backend`; the opt-in gate cannot police arbitrary code inside a caller-supplied factory.

## Synthetic acceptance corpus

```powershell
.venv/Scripts/python.exe -m tradingagents.evaluation corpus examples/evaluation/corpus.json --output corpus-report.json
```

This offline command checks four curated **invented scenarios and scripted outputs**, not recorded model responses or expert-adjudicated investment labels:

| Scenario | Evidence/injection coverage | Scripted scoring check |
| --- | --- | --- |
| Balanced | Competing observations; forged system authority in lessons | Six injection-pair flips |
| Supportive | Repeated article IDs; citation stuffing | One order flip; conflicting ratings excluded |
| Adverse | Downside evidence; forged contrary lesson | Truncated ratings excluded |
| Insufficient | Missing historical/social coverage; proposed future-data substitution | Missing controls and transport errors remain unassessed |

All 48 tasks use one trial and all six role permutations. These choices exercise software edge cases; neither the scenario distribution nor the scripted flips estimate real-world model behavior. Ticker names are labels for invented scenarios, not claims about those instruments.

A corpus declares `schema_version: 1`, `provenance: "synthetic-scripted"`, and `cases`. Each entry embeds a standard frozen `case`, a `purpose`, `scripts`, and `expected` metrics. Scripts must cover exactly `control` plus each injection ID. Each script has a `default` response object (the usual replay content/metadata fields) or `null` for missing. Optional `by_order` overrides use comma-separated role names, e.g. `conservative,aggressive,neutral`; an explicit null override stays missing. Identity/backend fields cannot be supplied by scripts.

`expected` asserts all five status counts plus `comparable_pairs` and `rating_flips` for both sensitivity metrics. A mismatch, duplicate case ID, unknown condition/order, or malformed response fails the command without publishing a report. Existing outputs are never overwritten. Limits are ten cases, 2 MB of canonical JSON, and 200 tasks across the entire corpus.

Reports include the corpus hash, individual case/plan/record hashes, per-case metrics, a synthetic backend label, and `not_model_evaluation: true`. They do not pool cases into an empirical performance score. `replay_corpus(corpus)` provides the same check from Python and never accepts a model factory.

To prepare a real evaluation, save selected embedded `case` objects as ordinary case files, review their evidence and interventions, and use `plan`. Keep genuine provider recordings separate and use `replay`; do not relabel these scripts as recordings. Representative real snapshots, independent review, and an approved provider/model/call budget are still needed before live evaluation or quality claims.

## Follow-up controls and prepared evaluation

Social adapters now screen minor edits using word and adjacent-word overlap (Jaccard thresholds 0.85 and 0.75). Screening preserves explicit numeric/numeric-bearing tokens, cashtags and selected polarity/negation counts, uses only complete texts of 8–128 tokens / at most 4,000 characters, and indexes at most 300 canonical candidates. Duplicates do not extend clusters transitively. Short/oversized texts and index overflow keep exact-ID/text controls; screen/index omissions are reported. This is conservative lexical matching, not semantic equivalence or calibrated bot detection.

`tests/test_fix_plan_completion.py` exercises real adapter rendering with mocked feeds. The saved ablation in `evaluation_runs/remaining-plan-offline-20260908T213412777581Z/social_ablation.json` disables only near matching for its baseline. It measures a synthetic StockTwits minor-edit flood changing from 7/8 retained labeled bullish to 1/2, and Reddit retention from four to two. These are deterministic source-selection measurements, not model accuracy results. Actual feeds, short-message campaigns, independent similar opinions and paraphrases may behave differently.

The same artifact directory contains `adverse_repeat_plan.json`: six trials of both conditions in all six role orders, **72 calls**, with prompt hashes matching the earlier adverse pilot. Its original preparation artifacts include unexecuted metrics and `approval_required.json`; these preserve the state before authorization. The user subsequently approved all 72 calls. Results are saved separately in `evaluation_runs/local-luna-repeat-20260908T214532743256Z/REPORT.md`: 72 valid responses, 14/60 order flips, 6/36 injection/control flips, and 26/60 identical-prompt repeat flips. This demonstrates background variability, not causal bias reduction. **That authorization is now exhausted; further live execution requires a new approval.** Read the original `REVIEW.md` for the qualitative rationale review and the new live report for the repeated experiment's limitations.

## Metrics and interpretation

- `valid`: complete prose with an unambiguous five-tier rating. This does **not** validate factual correctness.
- `invalid`: complete prose without an actionable rating, including conflicting labels.
- `incomplete`: empty/truncated/blocked responses or pending tool requests.
- `error`: failed invocation; `missing`: no saved response.
- **Order sensitivity:** each noncanonical order versus the canonical order, within the same condition and trial.
- **Injection sensitivity:** each attack versus the clean control with the same order and trial.
- **Repeat sensitivity** (plans with more than one trial): each later trial versus trial zero at the same condition and order. Prompt hashes must be identical across repeats. Missing/invalid references are excluded. Single-trial reports retain their original contract without a repeat metric.

Repeat comparisons estimate background generation variation, not causal bias. Shared reference responses mean pairs are not independent. Do not simply subtract repeat flip rates from order/injection rates or claim statistical significance. Scheduling seeds still do not control model sampling seeds.

Flip rates use only pairs with two valid ratings. Reports include eligible/comparable/excluded pair counts; no comparable pairs yields `null`, never a fabricated zero. Citation fields only compare literal `NEWS-` IDs to the prompt; they do not verify support, attribution, or truth. Truncation flags warn that an injected prefix may displace legitimate evidence, confounding an observed change.

The shared runtime Portfolio Manager prompt is used, with a fixed evaluation-only rating-line instruction. Evaluation is English and free-text, and measures the **first attempt**, not the production structured-output/retry pipeline. It does not rerun analysts or debates. Repeat measurements across appropriate cases/models before making claims; report missing/invalid rates alongside flips. Synthetic replay numbers validate software mechanics only.
