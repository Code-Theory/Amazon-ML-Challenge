# Amazon Business Entity Resolution Challenge

This repository implements an offline, reproducible solution for the challenge defined by
`docs/specification/amazon_ml_challenge_problem_statement.pdf`.

For every deduplicated Source 1 business, the system must return zero or more matching Source 2
and Source 3 record IDs. The official metric is macro entity-level F0.5: each Source 1 entity is
scored separately and then averaged, with an exact-empty prediction earning 1.0 for a true
singleton. The output is therefore not ordinary pairwise classification: candidate recall, false
merges, the decision threshold, and empty predictions all matter.

## Honest status

The official dataset is not present. No dataset profile, trained competition model, CV score,
runtime estimate, leaderboard score, or improvement claim exists. Synthetic records are used only
to test contracts, shapes, leakage barriers, model initialization, persistence, inference, and
submission serialization.

## Implemented pipeline

```text
official TSVs
  -> strict schema/ID audit and SHA-256 fingerprints
  -> entity-group fold manifest
  -> label-free character TF-IDF candidate generation
  -> pairwise name/address/country/source features
  -> logistic-regression or histogram-gradient-boosting matcher
  -> inner-OOF F0.5 threshold selection
  -> outer nested-CV evaluation OR fold-model ensemble training
  -> averaged test scores
  -> matching_results.tsv + candidate_pairs.tsv
  -> repository preflight + organizer validator
```

Important behavior:

- Source 1 groups and all labeled S2/S3 matches stay in one fold; candidate-pair rows are never
  randomly split.
- Candidate indices are rebuilt inside each train/validation partition.
- Outer-fold thresholds come only from that fold's inner OOF predictions.
- Single-view combined text is the corrected baseline. Optional multiview blocking unions
  combined, name-only, and address-only retrieval.
- Missing-vs-missing text is not treated as similarity evidence.
- Country is an open string label. It is not restricted to US/India, so France and other unseen
  labels pass through safely.
- The final training command creates a fold-model ensemble and tunes one deployment threshold from
  its OOF scores. The predict command averages fold-model probabilities on one fixed test candidate
  set.
- No match is ever forced. Final matches are always a subset of submitted candidates.

## Repository layout

```text
configs/
  fast_dev.toml                 # quick structural smoke test; not for model selection
  baseline.toml                 # corrected single-view logistic baseline
  multiview_logistic.toml       # candidate-generation/feature ablation
  multiview_histgb.toml         # nonlinear matcher candidate
data/
  train/ and test/              # official TSVs; ignored by Git
docs/
  DATA_CONTRACT.md
  DECISIONS.md
  EXPERIMENT_LOG.md
  EXPERIMENT_PLAN.md
  RISK_REGISTER.md
  VALIDATION_PROTOCOL.md
src/mlchallenge/
  audit.py                      # hashes and descriptive data audit
  candidates.py                 # blocking and blocking diagnostics
  cli.py                        # doctor/audit/CV/train/predict/preflight commands
  config.py                     # typed TOML configuration and seeds
  contracts.py                  # strict loaders and data gate
  features.py                   # pairwise features
  folds.py                      # deterministic entity-group splits
  metrics.py                    # exact macro entity-level F0.5
  modeling.py                   # supported pair classifiers
  pipeline.py                   # nested CV, ensemble training, inference
  submission.py                 # exact TSV writing and validation
tests/                          # synthetic correctness tests only
artifacts/                      # generated models/manifests; ignored
reports/                        # generated audits/CV reports; ignored
output/                         # required competition TSVs; ignored
```

## Dataset setup

Copy the official files without renaming them:

```text
data/
|-- train/
|   |-- train_source1.tsv
|   |-- train_source2.tsv
|   |-- train_source3.tsv
|   `-- train_ground_truth.tsv
`-- test/
    |-- test_source1.tsv
    |-- test_source2.tsv
    `-- test_source3.tsv
```

The PDF names the organizer root `dataset/`; this repository centralizes it as `data/`. You may
instead pass `--data-root` to every command. Files remain tab-separated. Do not convert them to CSV.

When the full organizer bundle is available, also keep `utils/validate_submission.py` and
`Documentation_template.md`. The organizer validator is the final format authority.

## Windows setup and commands

The archive currently contains an outer folder and the actual project folder with the same name.
Run these commands from the inner project root:

```powershell
Set-Location 'D:\Amazon ML challenge\MLChall-main\MLChall-main'
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
```

If `uv` is installed, the locked alternative is:

```powershell
uv sync --extra dev
```

Validate imports, config, and data paths:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge doctor --config configs\baseline.toml
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
```

Audit the official data before modeling:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge audit --config configs\baseline.toml
```

The audit reports progress for every major stage. For a quick exploratory pass, add
`--skip-hashes` to avoid rereading all TSV files for SHA-256 fingerprints. Keep hashing enabled
for recorded experiments so their exact input files remain traceable.

Run leakage-resistant nested cross-validation:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge cross-validate --config configs\baseline.toml
```

For a faster end-to-end structural check before committing to the full nested CV run:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge cross-validate --config configs\fast_dev.toml
```

`fast_dev.toml` deliberately uses fewer folds, candidates, vocabulary features, and threshold
points. It is for debugging only; do not compare its score with the full experiment configs.

Linux/WSL equivalent:

```bash
cd '/mnt/d/Amazon ML challenge/MLChall-main/MLChall-main'
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m mlchallenge cross-validate --config configs/fast_dev.toml
```

For better WSL disk performance, copy the repository to `~/MLChall-main` before running full CV.

Fit the fold ensemble and frozen deployment threshold:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge train --config configs\baseline.toml
```

Run test inference and generate both required files:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge predict --config configs\baseline.toml
```

Run local submission checks:

```powershell
.\.venv\Scripts\python.exe -m mlchallenge preflight --config configs\baseline.toml
```

Then run the organizer validator when it is supplied:

```powershell
.\.venv\Scripts\python.exe utils\validate_submission.py `
  --matching output\matching_results.tsv `
  --candidate output\candidate_pairs.tsv `
  --test-dir data\test
```

## Outputs

`cross-validate` writes a JSON report, pair-level OOF scores, and entity-level OOF predictions. It
logs the model, fold, train/validation Source 1 counts, pair counts, feature count, selected inner
threshold, fold metric, and duration. The report contains fold mean/std and the overall OOF macro
F0.5.

`train` writes a joblib model bundle, readable JSON metadata, training OOF pair scores, and the
threshold search table. Its OOF selection score is explicitly not a nested-CV estimate.

`predict` writes:

- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`
- `reports/<experiment>/inference.json`

Before writing, and again after reading the serialized bytes, the code verifies exact columns,
Source 1 coverage and row order, allowed S2/S3 IDs, duplicate-free lists, finite scores, and the
final-match-subset-of-candidates rule.

## Experiment discipline

Start with `configs/baseline.toml`. Compare multiview retrieval before changing the matcher; a
strong classifier cannot recover a blocked-out true pair. Use `configs/multiview_histgb.toml` only
after the multiview candidate ceiling and runtime are acceptable. Do not tune from public
leaderboard feedback. The ordered experiment plan and unfilled result fields are in
`docs/EXPERIMENT_PLAN.md` and `docs/EXPERIMENT_LOG.md`.

Only supplied challenge data may provide identity evidence. Business registries, geocoders, search
engines, entity-resolution APIs, and internet augmentation are prohibited. Any future pretrained
model must be documented and satisfy the PDF's MIT/Apache-2.0 and at-most-8B-parameter rule.
