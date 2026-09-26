# Experiment log

Every run receives an immutable ID. Entries include the Git commit, data fingerprints, configuration,
command, environment, seed, runtime, peak memory, blocking diagnostics, validation results, output
hashes, observations, and decision. Failed experiments are retained because they prevent repeated
mistakes.

## E000 - Repository foundation

- Date: 2026-09-25
- Data: unavailable
- Purpose: establish contracts, validation policy, transparent baseline components, and tests.
- Result: no model was trained and no metric was calculated.
- Decision: wait for the official dataset and student resource bundle; audit before selecting any
  dataset-dependent parameters.

## E001 - Dataset-independent implementation verification

- Date: 2026-09-26
- Data: official data unavailable; synthetic records only
- Purpose: verify executable contracts and the complete control flow.
- Checks: source compilation, lint, metric cases, multiview candidates, both model families, nested
  folds, fold-model persistence, averaged inference, and two-file submission serialization.
- Result: structural checks only; no competition metric, runtime, or model-quality result recorded.
- Decision: run the data audit next. Do not select a candidate budget, model, or threshold yet.

## Entry template

```text
Experiment ID:
Date/time and timezone:
Git commit:
Data audit/report hash:
Configuration and hash:
Command:
Environment/lock hash:
Seed(s):
Hardware:
Runtime/peak memory:
Blocking recall and candidate statistics:
Outer-fold macro F0.5 (mean, spread, confidence interval):
Slice results:
Output artifact hashes:
Observations:
Decision and rationale:
```
