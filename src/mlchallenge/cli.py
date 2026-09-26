"""Command-line entry points for auditable project operations."""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from mlchallenge.audit import audit_data, sha256_file, write_audit
from mlchallenge.config import ExperimentConfig, load_experiment_config
from mlchallenge.contracts import (
    ContractError,
    DatasetNotFoundError,
    load_test_data,
    load_training_data,
    missing_data_files,
)
from mlchallenge.folds import build_fold_manifest
from mlchallenge.pipeline import (
    cross_validate,
    load_model_bundle,
    predict_with_bundle,
    save_model_bundle,
    train_cv_ensemble,
)
from mlchallenge.submission import validate_submission_frames

DEFAULT_CONFIG = "configs/baseline.toml"


def _config(args: argparse.Namespace) -> ExperimentConfig:
    return load_experiment_config(args.config)


def _data_root(args: argparse.Namespace, config: ExperimentConfig) -> str:
    return args.data_root or config.paths.data_root


def _write_json(payload: object, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _runtime_versions() -> dict[str, str]:
    try:
        rapidfuzz_version = version("rapidfuzz")
    except PackageNotFoundError:
        rapidfuzz_version = "not installed; deterministic stdlib fallback active"
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "rapidfuzz": rapidfuzz_version,
    }


def _fingerprints(data_root: str | Path, split: str) -> dict[str, dict[str, str | int]]:
    root = Path(data_root)
    return {
        str(path.relative_to(root)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(root.joinpath(split).glob("*.tsv"))
    }


def _doctor(args: argparse.Namespace) -> int:
    config = _config(args)
    data_root = Path(_data_root(args, config))
    missing = missing_data_files(data_root)
    report = {
        **_runtime_versions(),
        "config": str(Path(args.config).resolve()),
        "data_root": str(data_root.resolve()),
        "missing_data_files": [str(path) for path in missing],
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if missing:
        print("Dataset not found. Place the official TSV files at the paths listed above.")
        return 2
    print("PASS: environment imports and all required dataset paths are available")
    return 0


def _audit(args: argparse.Namespace) -> int:
    config = _config(args)
    report = audit_data(_data_root(args, config))
    output = args.output or str(Path(config.paths.report_dir) / "data_audit.json")
    write_audit(report, output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _make_folds(args: argparse.Namespace) -> int:
    config = _config(args)
    data = load_training_data(_data_root(args, config))
    n_splits = args.n_splits or config.validation.n_outer_splits
    seed = args.seed if args.seed is not None else config.validation.seed
    manifest = build_fold_manifest(
        data.source1,
        data.source2,
        data.source3,
        data.ground_truth,
        n_splits=n_splits,
        seed=seed,
    )
    output = Path(args.output or Path(config.paths.artifact_dir) / "fold_manifest.tsv")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output, sep="\t", index=False, lineterminator="\n")
    print(f"Wrote {len(manifest)} records to {output}")
    return 0


def _cross_validate(args: argparse.Namespace) -> int:
    config = _config(args)
    data_root = _data_root(args, config)
    data = load_training_data(data_root)
    report, pairs, entities = cross_validate(data, config)
    report["data_files"] = _fingerprints(data_root, "train")
    report["environment"] = _runtime_versions()
    report["config_sha256"] = sha256_file(Path(args.config))
    report_dir = Path(config.paths.report_dir)
    report_path = Path(args.output or report_dir / "cross_validation.json")
    pair_path = Path(args.oof_pairs or report_dir / "oof_pair_scores.tsv")
    entity_path = Path(args.oof_entities or report_dir / "oof_entity_predictions.tsv")
    _write_json(report, report_path)
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    entity_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(pair_path, sep="\t", index=False, lineterminator="\n")
    entities.to_csv(entity_path, sep="\t", index=False, lineterminator="\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Wrote {report_path}, {pair_path}, and {entity_path}")
    return 0


def _train(args: argparse.Namespace) -> int:
    config = _config(args)
    data_root = _data_root(args, config)
    data = load_training_data(data_root)
    bundle, oof_scores, threshold_table = train_cv_ensemble(data, config)
    bundle.training_summary["data_files"] = _fingerprints(data_root, "train")
    bundle.training_summary["environment"] = _runtime_versions()
    bundle.training_summary["config_sha256"] = sha256_file(Path(args.config))
    artifact_dir = Path(config.paths.artifact_dir)
    model_path = Path(args.model_output or artifact_dir / "model_bundle.joblib")
    metadata_path = Path(args.metadata_output or artifact_dir / "model_bundle.json")
    save_model_bundle(bundle, model_path, metadata_path=metadata_path)
    oof_path = Path(args.oof_output or artifact_dir / "training_oof_pair_scores.tsv")
    threshold_path = Path(args.threshold_output or artifact_dir / "threshold_search.tsv")
    oof_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_path.parent.mkdir(parents=True, exist_ok=True)
    oof_scores.to_csv(oof_path, sep="\t", index=False, lineterminator="\n")
    threshold_table.to_csv(threshold_path, sep="\t", index=False, lineterminator="\n")
    print(json.dumps(bundle.training_summary, indent=2, sort_keys=True))
    print(f"Wrote model bundle to {model_path}")
    return 0


def _predict(args: argparse.Namespace) -> int:
    config = _config(args)
    data_root = _data_root(args, config)
    artifact_dir = Path(config.paths.artifact_dir)
    model_path = Path(args.model or artifact_dir / "model_bundle.joblib")
    output_dir = Path(args.output_dir or config.paths.output_dir)
    bundle = load_model_bundle(model_path)
    test = load_test_data(data_root)
    report = predict_with_bundle(bundle, test, output_dir)
    report["data_files"] = _fingerprints(data_root, "test")
    report["environment"] = _runtime_versions()
    report_path = Path(args.report or Path(config.paths.report_dir) / "inference.json")
    _write_json(report, report_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    config = _config(args)
    test = load_test_data(_data_root(args, config))
    matching_path = args.matching or str(Path(config.paths.output_dir) / "matching_results.tsv")
    candidate_path = args.candidates or str(Path(config.paths.output_dir) / "candidate_pairs.tsv")
    matching = pd.read_csv(
        matching_path, sep="\t", dtype="string", keep_default_na=False, na_filter=False
    )
    candidates = pd.read_csv(
        candidate_path, sep="\t", dtype="string", keep_default_na=False, na_filter=False
    )
    issues = validate_submission_frames(matching, candidates, test)
    if issues:
        for number, issue in enumerate(issues, start=1):
            print(f"{number}. {issue}")
        return 1
    print("PASS: repository preflight checks succeeded")
    return 0


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default=None)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mlchallenge",
        description="Auditable tools for the Amazon Business Entity Resolution Challenge",
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check imports, config, and dataset paths")
    _common(doctor)
    doctor.set_defaults(func=_doctor)

    audit_parser = subparsers.add_parser("audit", help="validate and fingerprint challenge data")
    _common(audit_parser)
    audit_parser.add_argument("--output", default=None)
    audit_parser.set_defaults(func=_audit)

    folds_parser = subparsers.add_parser(
        "make-folds", help="build a deterministic entity-group fold manifest"
    )
    _common(folds_parser)
    folds_parser.add_argument("--output", default=None)
    folds_parser.add_argument("--n-splits", type=int, default=None)
    folds_parser.add_argument("--seed", type=int, default=None)
    folds_parser.set_defaults(func=_make_folds)

    cv_parser = subparsers.add_parser(
        "cross-validate", help="run nested group-disjoint cross-validation"
    )
    _common(cv_parser)
    cv_parser.add_argument("--output", default=None)
    cv_parser.add_argument("--oof-pairs", default=None)
    cv_parser.add_argument("--oof-entities", default=None)
    cv_parser.set_defaults(func=_cross_validate)

    train_parser = subparsers.add_parser(
        "train", help="fit a CV ensemble and tune its deployment threshold from OOF scores"
    )
    _common(train_parser)
    train_parser.add_argument("--model-output", default=None)
    train_parser.add_argument("--metadata-output", default=None)
    train_parser.add_argument("--oof-output", default=None)
    train_parser.add_argument("--threshold-output", default=None)
    train_parser.set_defaults(func=_train)

    predict_parser = subparsers.add_parser(
        "predict", help="average ensemble scores and generate both submission TSVs"
    )
    _common(predict_parser)
    predict_parser.add_argument("--model", default=None)
    predict_parser.add_argument("--output-dir", default=None)
    predict_parser.add_argument("--report", default=None)
    predict_parser.set_defaults(func=_predict)

    preflight_parser = subparsers.add_parser(
        "preflight", help="validate the two submission TSV files locally"
    )
    _common(preflight_parser)
    preflight_parser.add_argument("--matching", default=None)
    preflight_parser.add_argument("--candidates", default=None)
    preflight_parser.set_defaults(func=_preflight)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    try:
        return int(args.func(args))
    except (DatasetNotFoundError, ContractError, FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
