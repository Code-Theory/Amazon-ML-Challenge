# Ordered experiment plan

No experiment below has been run on the official competition data. Result fields must remain empty
until the data audit is accepted and the named configuration is executed unchanged.

Every comparison uses nested entity-group cross-validation, the organizer's macro entity-level
F0.5, blocking diagnostics, identical raw-data hashes, and saved OOF artifacts. Change one major
factor at a time.

## Experiment 0 - reference components

- Purpose: document the inherited starting point.
- Components: combined name/address character TF-IDF top-k candidates, explicit pair features,
  balanced logistic regression, entity-level metric utilities.
- Limitation: the inherited repository exposed components but did not have an end-to-end nested-CV,
  final-training, inference, or submission-generation workflow.
- Competition result: not available.

## Experiment 1 - corrected baseline

- Config: `configs/baseline.toml`
- Change: missing-safe features, entity-balanced sample weights, deterministic group splits, nested
  threshold selection, OOF artifacts, CV ensemble inference, strict serialization.
- Decision gate: all contract tests pass; blocking recall and runtime are viable; no leakage audit
  failures.
- Competition result: not run.

## Experiment 2 - improved candidate recall and features

- Config: `configs/multiview_logistic.toml`
- Change: union combined/name/address character TF-IDF retrieval while retaining the linear matcher.
- Primary question: does matched-entity complete recall improve enough to justify extra candidates,
  memory, and runtime?
- Competition result: not run.

## Experiment 3 - stronger nonlinear matcher

- Config: `configs/multiview_histgb.toml`
- Change: histogram gradient boosting on the same multiview candidates and features.
- Primary question: are nonlinear interactions useful without degrading singleton precision or
  calibration?
- Competition result: not run.

## Experiment 4 - constrained hyperparameter tuning

- Prerequisite: freeze the better candidate strategy from Experiments 1-3.
- Search only a small, declared space for candidate budget, regularization/tree complexity, and
  learning rate inside inner folds.
- Reject searches that use outer-fold or leaderboard outcomes for selection.
- Competition result: not run.

## Experiment 5 - justified alternative model

- Candidate: ExtraTrees or a license-compliant CatBoost model only if audit scale and residual error
  analysis justify it.
- Keep candidates, features, folds, and threshold protocol fixed for attribution.
- Do not add a dependency before recording license, version, memory cost, and deterministic settings.
- Competition result: not run.

## Experiment 6 - ensemble or blending

- Prerequisite: two models must show complementary, stable outer-fold errors.
- Learn blend weights and the final threshold only from inner OOF predictions.
- Compare against the stronger single model; do not ensemble merely because multiple models exist.
- Competition result: not run.

## Experiment 7 - metric-specific decision policy

- Start with the global threshold already implemented.
- Consider per-source or per-missingness thresholds only when outer-fold slice evidence is stable and
  the rule can handle unseen France records without hard-coded country categories.
- Never force a match; retain empty predictions for singletons.
- Competition result: not run.

## Required fields for every completed run

- experiment ID and timestamp;
- Git commit or source-diff hash;
- data file SHA-256 values;
- exact config and config hash;
- environment/lock hash and hardware;
- fold metrics, mean/std, and overall OOF macro F0.5;
- blocking recall, matched-entity complete recall, reduction ratio, and candidate-count quantiles;
- singleton and missing-field slices;
- runtime and peak memory;
- artifact/output hashes;
- decision and rationale.

