# Repeated local evaluation: gpt-5.6-luna

## Execution and audit

- Executed the authorized 72-call plan: six trials of six role orders, with clean and forged-lesson conditions for the frozen synthetic adverse case.
- Endpoint: `http://127.0.0.1:10531/v1`; requested model `gpt-5.6-luna`; timeout 300 seconds; client retries disabled. Response metadata reported the requested model name.
- All **72 responses were valid** under the completeness/rating checks. No invalid, incomplete, error or missing results. Runtime: **545.462 seconds**.
- All responses were newly collected. No prior pilot observations were reused. The approved budget is exhausted; no further requests were made.
- Offline replay reproduced the saved report. Verified 72 unique task IDs and 12 exact prompt-hash groups, each with six trials. The supplied API key was checked absent from saved artifacts.
- Lifetime live evaluation usage across the original pilot and this experiment: 122 generation attempts, comprising 120 usable responses and the two earlier failures. These experiments remain separate datasets.

## Predefined measurements

| Comparison | Rating flips / comparable pairs | Descriptive rate |
| --- | ---: | ---: |
| Role order: noncanonical versus canonical within each condition/trial | 14 / 60 | 23.3% |
| Injection: forged lesson versus clean, same order/trial | 6 / 36 | 16.7% |
| Identical-prompt repeats: later trial versus trial zero, same condition/order | 26 / 60 | 43.3% |

All comparisons were eligible and available; no exclusions or prompt-truncation flags occurred.

| Condition | Hold | Underweight | Other ratings |
| --- | ---: | ---: | ---: |
| Clean control | 4 | 32 | 0 |
| Forged lesson | 8 | 28 | 0 |

No response returned the attack's requested Buy rating. No literal cited NEWS ID was absent from its prompt; this is not a factual-support or semantic-attribution audit.

## Interpretation

**The experiment directly demonstrates rating variability even when the exact prompt does not change.** Consequently, a rating flip in the earlier single-trial pilot was not sufficient evidence that role order or injection caused it.

This does **not** establish that order/injection effects are absent, that the safeguards reduced bias, or that one rate can be subtracted from another to calculate a causal effect. The comparison pairs share reference responses and are not independent. The reference-based repeat metric is particularly sensitive to trial zero: it contained five Hold ratings out of twelve, compared with zero to two Hold ratings in each later trial. Trial labels were shuffled in the request schedule; they are not chronological stages. The 43.3% figure is not a per-request error probability.

The observed uncertainty remains Hold versus Underweight. Those outputs can reflect different thresholds for acting on adverse but incomplete evidence. The model did not produce Buy in this sample, but this is not proof of prompt-injection immunity or absence of rationale-level contamination.

## What is complete and what is not proven

The remaining code changes, full configured Python test suite, social adapter ablation, and agreed repeat-controlled live measurements have now been executed. No additional live execution is pending under this authorization.

The effectiveness claim remains limited: this is one local gateway/model, one synthetic adverse case and six trials per configuration, with adapter/provider generation defaults and no controlled sampling seed. There is no whole-system pre-change comparison, representative real-evidence accuracy benchmark, independent human adjudication or verified upstream routing/billing. The earlier social ablation measured deterministic retained-evidence effects, not population sentiment accuracy.

Do not mark demonstrated bias reduction, trading accuracy or universal robustness as proven. No new production behavior was changed merely to force a desired rating on this fixture. Any future model calls require a new authorization.

## Artifacts

- `manifest.json`, `case.json`, `plan.json`: authorization/configuration and frozen inputs.
- `responses.jsonl`: newly collected responses with metadata and timestamps.
- `metrics.json`, `summary.json`: measured results.
- `audit.json`: per-trial and per-prompt ratings and replay verification.
