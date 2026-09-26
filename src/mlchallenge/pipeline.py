"""End-to-end, leakage-resistant training, validation, and inference workflows."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from mlchallenge.candidates import (
    CandidateConfig,
    CandidateTextCache,
    blocking_diagnostics,
    generate_candidates,
    prepare_candidate_text_cache,
)
from mlchallenge.config import ExperimentConfig, set_global_seed
from mlchallenge.contracts import TrainingData, truth_mapping
from mlchallenge.features import (
    FEATURE_COLUMNS,
    PairFeatureCache,
    build_pair_features,
    prepare_pair_feature_cache,
)
from mlchallenge.folds import build_fold_manifest, select_fold_partition
from mlchallenge.metrics import (
    entity_fbeta,
    macro_entity_fbeta,
    predictions_at_threshold,
    tune_threshold_from_oof,
)
from mlchallenge.modeling import PairMatcher, build_matcher, label_candidate_pairs
from mlchallenge.progress import ProgressBar
from mlchallenge.submission import write_submission_files

LOGGER = logging.getLogger("mlchallenge")


@dataclass
class ModelBundle:
    """Serializable CV ensemble plus the frozen inference policy."""

    bundle_version: int
    created_at_utc: str
    config: ExperimentConfig
    threshold: float
    matchers: list[PairMatcher]
    feature_columns: tuple[str, ...]
    training_summary: dict[str, Any]


def _subset_truth(ground_truth: pd.DataFrame, source1: pd.DataFrame) -> pd.DataFrame:
    wanted = set(source1["entity_id"].astype(str))
    result = ground_truth.loc[ground_truth["source1_entity_id"].astype(str).isin(wanted)].copy()
    return result.reset_index(drop=True)


def _partition_training(
    data: TrainingData,
    manifest: pd.DataFrame,
    *,
    held_out_fold: int,
    validation: bool,
) -> TrainingData:
    source1 = select_fold_partition(
        data.source1,
        manifest,
        source="source1",
        held_out_fold=held_out_fold,
        validation=validation,
    )
    source2 = select_fold_partition(
        data.source2,
        manifest,
        source="source2",
        held_out_fold=held_out_fold,
        validation=validation,
    )
    source3 = select_fold_partition(
        data.source3,
        manifest,
        source="source3",
        held_out_fold=held_out_fold,
        validation=validation,
    )
    return TrainingData(source1, source2, source3, _subset_truth(data.ground_truth, source1))


def _candidate_features(
    data: TrainingData,
    candidate_config: CandidateConfig,
    feature_cache: PairFeatureCache | None = None,
    text_cache: CandidateTextCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    started = time.perf_counter()
    LOGGER.info(
        "generating candidates method=%s source1=%d source2=%d source3=%d",
        candidate_config.method,
        len(data.source1),
        len(data.source2),
        len(data.source3),
    )
    candidates = generate_candidates(
        data.source1,
        data.source2,
        data.source3,
        candidate_config,
        text_cache=text_cache,
    )
    candidate_seconds = time.perf_counter() - started
    LOGGER.info("building pair features for %d candidate pairs", len(candidates))
    features = build_pair_features(
        candidates,
        data.source1,
        data.source2,
        data.source3,
        n_jobs=candidate_config.n_jobs,
        cache=feature_cache,
    )
    LOGGER.info(
        "prepared candidate_pairs=%d candidate_seconds=%.2f feature_seconds=%.2f",
        len(candidates),
        candidate_seconds,
        time.perf_counter() - started - candidate_seconds,
    )
    return candidates, features


def _fit(
    data: TrainingData,
    config: ExperimentConfig,
    feature_cache: PairFeatureCache | None = None,
    text_cache: CandidateTextCache | None = None,
) -> tuple[PairMatcher, int, int]:
    candidates, features = _candidate_features(data, config.candidates, feature_cache, text_cache)
    labels = label_candidate_pairs(features, data.ground_truth)
    matcher = build_matcher(config.model).fit(features, labels)
    return matcher, len(candidates), len(features)


def _score(
    matcher: PairMatcher,
    data: TrainingData,
    config: ExperimentConfig,
    feature_cache: PairFeatureCache | None = None,
    text_cache: CandidateTextCache | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates, features = _candidate_features(data, config.candidates, feature_cache, text_cache)
    if features.empty:
        scores = pd.DataFrame(columns=("source1_entity_id", "candidate_entity_id", "score"))
    else:
        scores = matcher.predict_scores(features)
    return candidates, scores


def _inner_oof_threshold(
    data: TrainingData,
    config: ExperimentConfig,
    *,
    n_splits: int,
    seed: int,
    context: str,
    feature_cache: PairFeatureCache,
    text_cache: CandidateTextCache,
) -> tuple[float, float, pd.DataFrame, pd.DataFrame]:
    """Tune a threshold using only OOF predictions within the supplied training partition."""

    if len(data.source1) < n_splits:
        raise ValueError(
            f"{context}: n_inner_splits={n_splits} exceeds Source 1 rows={len(data.source1)}"
        )
    manifest = build_fold_manifest(
        data.source1,
        data.source2,
        data.source3,
        data.ground_truth,
        n_splits=n_splits,
        seed=seed,
    )
    oof_parts: list[pd.DataFrame] = []
    progress = ProgressBar(
        f"{context} inner CV",
        total=n_splits,
        enabled=LOGGER.isEnabledFor(logging.INFO),
    )
    progress.start()
    for fold in range(n_splits):
        started = time.perf_counter()
        train = _partition_training(data, manifest, held_out_fold=fold, validation=False)
        valid = _partition_training(data, manifest, held_out_fold=fold, validation=True)
        matcher, train_candidates, _ = _fit(train, config, feature_cache, text_cache)
        _, scores = _score(matcher, valid, config, feature_cache, text_cache)
        scores.insert(0, "fold", fold)
        oof_parts.append(scores)
        LOGGER.info(
            "%s inner_fold=%d model=%s train_s1=%d valid_s1=%d train_pairs=%d "
            "valid_pairs=%d features=%d duration_seconds=%.2f",
            context,
            fold,
            config.model.family,
            len(train.source1),
            len(valid.source1),
            train_candidates,
            len(scores),
            len(FEATURE_COLUMNS),
            time.perf_counter() - started,
        )
        progress.update()
    oof = pd.concat(oof_parts, ignore_index=True)
    truth = truth_mapping(data.ground_truth)
    threshold, score, table = tune_threshold_from_oof(
        oof,
        truth,
        config.threshold.grid(),
    )
    return threshold, score, table, oof


def cross_validate(
    data: TrainingData,
    config: ExperimentConfig,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run nested entity-disjoint CV and return report plus OOF pair/entity predictions."""

    set_global_seed(config.validation.seed)
    outer_manifest = build_fold_manifest(
        data.source1,
        data.source2,
        data.source3,
        data.ground_truth,
        n_splits=config.validation.n_outer_splits,
        seed=config.validation.seed,
    )
    cache_started = time.perf_counter()
    LOGGER.info("preparing reusable normalized feature and retrieval caches")
    feature_cache = prepare_pair_feature_cache(data.source1, data.source2, data.source3)
    text_cache = prepare_candidate_text_cache(
        data.source1, data.source2, data.source3, config.candidates
    )
    LOGGER.info(
        "prepared reusable caches duration_seconds=%.2f", time.perf_counter() - cache_started
    )
    fold_reports: list[dict[str, Any]] = []
    pair_parts: list[pd.DataFrame] = []
    entity_rows: list[dict[str, Any]] = []

    progress = ProgressBar(
        "Outer cross-validation",
        total=config.validation.n_outer_splits,
        enabled=LOGGER.isEnabledFor(logging.INFO),
    )
    progress.start()
    for outer_fold in range(config.validation.n_outer_splits):
        started = time.perf_counter()
        train = _partition_training(
            data, outer_manifest, held_out_fold=outer_fold, validation=False
        )
        valid = _partition_training(data, outer_manifest, held_out_fold=outer_fold, validation=True)
        threshold, inner_score, _, _ = _inner_oof_threshold(
            train,
            config,
            n_splits=config.validation.n_inner_splits,
            seed=config.validation.seed + 10_000 + outer_fold,
            context=f"outer_fold={outer_fold}",
            feature_cache=feature_cache,
            text_cache=text_cache,
        )
        matcher, train_candidate_count, _ = _fit(train, config, feature_cache, text_cache)
        valid_candidates, scores = _score(matcher, valid, config, feature_cache, text_cache)
        valid_truth = truth_mapping(valid.ground_truth)
        predictions = predictions_at_threshold(scores, valid_truth, threshold)
        fold_score = macro_entity_fbeta(predictions, valid_truth, beta=0.5)
        diagnostics = blocking_diagnostics(
            valid_candidates,
            valid.ground_truth,
            possible_target_count=len(valid.source2) + len(valid.source3),
        )
        duration = time.perf_counter() - started
        fold_reports.append(
            {
                "fold": outer_fold,
                "model": config.model.family,
                "train_source1_rows": len(train.source1),
                "validation_source1_rows": len(valid.source1),
                "train_candidate_pairs": train_candidate_count,
                "validation_candidate_pairs": len(valid_candidates),
                "feature_count": len(FEATURE_COLUMNS),
                "threshold_from_inner_oof": threshold,
                "inner_oof_selection_score": inner_score,
                "validation_macro_f0_5": fold_score,
                "duration_seconds": duration,
                "blocking": diagnostics,
            }
        )
        scored = scores.copy()
        scored.insert(0, "outer_fold", outer_fold)
        scored["threshold"] = threshold
        scored["selected"] = scored["score"].ge(threshold)
        pair_parts.append(scored)
        for source1_id, expected in valid_truth.items():
            predicted = predictions[source1_id]
            entity_rows.append(
                {
                    "outer_fold": outer_fold,
                    "source1_entity_id": source1_id,
                    "expected_entity_ids": ",".join(sorted(expected)),
                    "predicted_entity_ids": ",".join(sorted(predicted)),
                    "entity_f0_5": entity_fbeta(predicted, expected, beta=0.5),
                }
            )
        LOGGER.info(
            "outer_fold=%d model=%s train_s1=%d valid_s1=%d features=%d "
            "threshold=%.4f fold_macro_f0_5=%.6f duration_seconds=%.2f",
            outer_fold,
            config.model.family,
            len(train.source1),
            len(valid.source1),
            len(FEATURE_COLUMNS),
            threshold,
            fold_score,
            duration,
        )
        progress.update()

    fold_scores = np.asarray(
        [item["validation_macro_f0_5"] for item in fold_reports], dtype=np.float64
    )
    entities = pd.DataFrame(entity_rows)
    overall = float(entities["entity_f0_5"].mean())
    report = {
        "evaluation": "nested entity-disjoint cross-validation",
        "metric": "macro_f0.5",
        "model": config.model.family,
        "config": config.as_dict(),
        "feature_count": len(FEATURE_COLUMNS),
        "outer_folds": fold_reports,
        "cv_mean": float(fold_scores.mean()),
        "cv_std": float(fold_scores.std(ddof=1)) if len(fold_scores) > 1 else 0.0,
        "overall_oof_macro_f0_5": overall,
        "note": (
            "Each outer-fold threshold was chosen only from that fold's inner OOF predictions. "
            "No competition performance exists until this is run on the official data."
        ),
    }
    pairs = pd.concat(pair_parts, ignore_index=True)
    return report, pairs, entities


def train_cv_ensemble(
    data: TrainingData,
    config: ExperimentConfig,
) -> tuple[ModelBundle, pd.DataFrame, pd.DataFrame]:
    """Fit fold models for averaged test inference and tune one deployment threshold from OOF."""

    set_global_seed(config.validation.seed)
    manifest = build_fold_manifest(
        data.source1,
        data.source2,
        data.source3,
        data.ground_truth,
        n_splits=config.validation.n_outer_splits,
        seed=config.validation.seed,
    )
    cache_started = time.perf_counter()
    LOGGER.info("preparing reusable normalized feature and retrieval caches")
    feature_cache = prepare_pair_feature_cache(data.source1, data.source2, data.source3)
    text_cache = prepare_candidate_text_cache(
        data.source1, data.source2, data.source3, config.candidates
    )
    LOGGER.info(
        "prepared reusable caches duration_seconds=%.2f", time.perf_counter() - cache_started
    )
    matchers: list[PairMatcher] = []
    oof_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    progress = ProgressBar(
        "Training ensemble",
        total=config.validation.n_outer_splits,
        enabled=LOGGER.isEnabledFor(logging.INFO),
    )
    progress.start()
    for fold in range(config.validation.n_outer_splits):
        started = time.perf_counter()
        train = _partition_training(data, manifest, held_out_fold=fold, validation=False)
        valid = _partition_training(data, manifest, held_out_fold=fold, validation=True)
        matcher, train_pairs, _ = _fit(train, config, feature_cache, text_cache)
        _, scores = _score(matcher, valid, config, feature_cache, text_cache)
        scores.insert(0, "fold", fold)
        oof_parts.append(scores)
        matchers.append(matcher)
        fold_rows.append(
            {
                "fold": fold,
                "train_source1_rows": len(train.source1),
                "validation_source1_rows": len(valid.source1),
                "train_candidate_pairs": train_pairs,
                "validation_candidate_pairs": len(scores),
                "duration_seconds": time.perf_counter() - started,
            }
        )
        LOGGER.info(
            "final_ensemble_fold=%d model=%s train_s1=%d valid_s1=%d train_pairs=%d "
            "valid_pairs=%d features=%d duration_seconds=%.2f",
            fold,
            config.model.family,
            len(train.source1),
            len(valid.source1),
            train_pairs,
            len(scores),
            len(FEATURE_COLUMNS),
            fold_rows[-1]["duration_seconds"],
        )
        progress.update()
    oof = pd.concat(oof_parts, ignore_index=True)
    threshold, selection_score, threshold_table = tune_threshold_from_oof(
        oof,
        truth_mapping(data.ground_truth),
        config.threshold.grid(),
    )
    summary = {
        "training_mode": "cv_ensemble",
        "ensemble_size": len(matchers),
        "deployment_threshold": threshold,
        "oof_threshold_selection_score": selection_score,
        "folds": fold_rows,
        "warning": (
            "The OOF threshold-selection score is not a nested-CV performance estimate. "
            "Use the cross-validate command for model comparison."
        ),
    }
    bundle = ModelBundle(
        bundle_version=1,
        created_at_utc=datetime.now(UTC).isoformat(),
        config=config,
        threshold=threshold,
        matchers=matchers,
        feature_columns=FEATURE_COLUMNS,
        training_summary=summary,
    )
    return bundle, oof, threshold_table


def save_model_bundle(
    bundle: ModelBundle,
    path: str | Path,
    *,
    metadata_path: str | Path | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output)
    metadata = Path(metadata_path) if metadata_path else output.with_suffix(".json")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bundle_version": bundle.bundle_version,
        "created_at_utc": bundle.created_at_utc,
        "threshold": bundle.threshold,
        "feature_columns": list(bundle.feature_columns),
        "config": bundle.config.as_dict(),
        "training_summary": bundle.training_summary,
    }
    metadata.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_model_bundle(path: str | Path) -> ModelBundle:
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, ModelBundle) or bundle.bundle_version != 1:
        raise ValueError("unsupported or invalid model bundle")
    if tuple(bundle.feature_columns) != FEATURE_COLUMNS:
        raise ValueError("model feature schema does not match this code version")
    if not bundle.matchers:
        raise ValueError("model bundle contains no fitted estimators")
    return bundle


def predict_with_bundle(
    bundle: ModelBundle,
    test_data: Any,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Generate final candidates, average fold-model scores, and write both required TSVs."""

    started = time.perf_counter()
    progress = ProgressBar(
        "Inference",
        total=5,
        enabled=LOGGER.isEnabledFor(logging.INFO),
    )
    progress.start()
    text_cache = prepare_candidate_text_cache(
        test_data.source1,
        test_data.source2,
        test_data.source3,
        bundle.config.candidates,
    )
    progress.update()
    candidates = generate_candidates(
        test_data.source1,
        test_data.source2,
        test_data.source3,
        bundle.config.candidates,
        text_cache=text_cache,
    )
    progress.update()
    feature_cache = prepare_pair_feature_cache(
        test_data.source1,
        test_data.source2,
        test_data.source3,
    )
    features = build_pair_features(
        candidates,
        test_data.source1,
        test_data.source2,
        test_data.source3,
        n_jobs=bundle.config.candidates.n_jobs,
        cache=feature_cache,
    )
    progress.update()
    if features.empty:
        averaged_scores = pd.DataFrame(
            columns=("source1_entity_id", "candidate_entity_id", "score")
        )
    else:
        probabilities = np.vstack(
            [matcher.predict_scores(features)["score"].to_numpy() for matcher in bundle.matchers]
        )
        averaged_scores = pd.DataFrame(
            {
                "source1_entity_id": features["source1_entity_id"].astype(str),
                "candidate_entity_id": features["candidate_entity_id"].astype(str),
                "score": probabilities.mean(axis=0),
            }
        )
    source1_ids = test_data.source1["entity_id"].astype(str).tolist()
    predictions = predictions_at_threshold(averaged_scores, source1_ids, bundle.threshold)
    progress.update()
    candidate_mapping = {source1_id: set() for source1_id in source1_ids}
    for source1_id, group in candidates.groupby("source1_entity_id", sort=False):
        candidate_mapping[str(source1_id)] = set(group["candidate_entity_id"].astype(str))
    matching_path, candidate_path = write_submission_files(
        test_data,
        predictions,
        candidate_mapping,
        output_dir,
    )
    progress.update()
    return {
        "model": bundle.config.model.family,
        "ensemble_size": len(bundle.matchers),
        "threshold": bundle.threshold,
        "test_source1_rows": len(test_data.source1),
        "candidate_pairs": len(candidates),
        "predicted_pairs": int(sum(len(value) for value in predictions.values())),
        "matching_results": str(matching_path),
        "candidate_pairs_file": str(candidate_path),
        "duration_seconds": time.perf_counter() - started,
    }
