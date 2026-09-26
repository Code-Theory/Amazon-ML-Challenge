"""Organizer-aligned entity-level evaluation functions."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd


def _validated_pair_scores(scored_pairs: pd.DataFrame) -> pd.Series:
    required = {"source1_entity_id", "candidate_entity_id", "score"}
    missing = required - set(scored_pairs.columns)
    if missing:
        raise ValueError(f"scored pairs missing columns: {sorted(missing)}")
    scores = pd.to_numeric(scored_pairs["score"], errors="coerce")
    if not np.isfinite(scores.to_numpy()).all() or not scores.between(0.0, 1.0).all():
        raise ValueError("pair scores must be finite probabilities in [0, 1]")
    return scores


def entity_fbeta(
    predicted: Iterable[str],
    expected: Iterable[str],
    *,
    beta: float = 0.5,
) -> float:
    """Score one Source 1 entity, including organizer-defined singleton behavior."""

    if beta <= 0:
        raise ValueError("beta must be positive")
    predicted_set = frozenset(predicted)
    expected_set = frozenset(expected)
    if not predicted_set and not expected_set:
        return 1.0
    if not predicted_set or not expected_set:
        return 0.0
    true_positive = len(predicted_set & expected_set)
    if true_positive == 0:
        return 0.0
    precision = true_positive / len(predicted_set)
    recall = true_positive / len(expected_set)
    beta_sq = beta**2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)


def macro_entity_fbeta(
    predictions: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    *,
    beta: float = 0.5,
) -> float:
    """Macro-average F-beta over exactly the Source 1 IDs in ground truth."""

    unknown = set(predictions) - set(truth)
    if unknown:
        raise ValueError(f"predictions contain unknown Source 1 IDs: {sorted(unknown)[:10]}")
    if not truth:
        raise ValueError("truth mapping is empty")
    scores = [
        entity_fbeta(predictions.get(key, ()), value, beta=beta) for key, value in truth.items()
    ]
    return float(np.mean(scores))


def predictions_at_threshold(
    scored_pairs: pd.DataFrame,
    source1_ids: Iterable[str],
    threshold: float,
) -> dict[str, frozenset[str]]:
    scores = _validated_pair_scores(scored_pairs)
    predictions = {str(entity_id): set() for entity_id in source1_ids}
    selected = scored_pairs.loc[scores >= threshold]
    for row in selected.itertuples(index=False):
        source1_id = str(row.source1_entity_id)
        if source1_id not in predictions:
            raise ValueError(f"scored pairs contain unknown Source 1 ID: {source1_id}")
        predictions[source1_id].add(str(row.candidate_entity_id))
    return {key: frozenset(value) for key, value in predictions.items()}


def tune_threshold_from_oof(
    scored_pairs: pd.DataFrame,
    truth: Mapping[str, Iterable[str]],
    thresholds: Iterable[float],
) -> tuple[float, float, pd.DataFrame]:
    """Choose a threshold from inner out-of-fold scores only.

    The caller is responsible for proving that `scored_pairs` are genuinely out-of-fold. Ties are
    resolved toward the higher threshold, reflecting the challenge's precision-heavy objective.
    """

    threshold_values = np.asarray(sorted({float(item) for item in thresholds}), dtype=np.float64)
    if threshold_values.size == 0:
        raise ValueError("at least one threshold is required")
    if not np.isfinite(threshold_values).all():
        raise ValueError("thresholds must be finite")

    scores = _validated_pair_scores(scored_pairs)
    truth_sets = {str(key): frozenset(str(item) for item in value) for key, value in truth.items()}
    if not truth_sets:
        raise ValueError("truth mapping is empty")

    pairs = pd.DataFrame(
        {
            "source1_entity_id": scored_pairs["source1_entity_id"].astype(str),
            "candidate_entity_id": scored_pairs["candidate_entity_id"].astype(str),
            "score": scores.to_numpy(dtype=np.float64),
        }
    )
    unknown = set(pairs["source1_entity_id"]) - set(truth_sets)
    if unknown:
        raise ValueError(f"scored pairs contain unknown Source 1 IDs: {sorted(unknown)[:10]}")

    # A repeated pair behaves like a set member in predictions_at_threshold: it is selected when
    # any occurrence reaches the threshold, which is equivalent to retaining its maximum score.
    pairs = pairs.groupby(["source1_entity_id", "candidate_entity_id"], as_index=False, sort=False)[
        "score"
    ].max()
    groups = {
        str(source1_id): group
        for source1_id, group in pairs.groupby("source1_entity_id", sort=False)
    }

    beta_sq = 0.25
    score_sums = np.zeros(len(threshold_values), dtype=np.float64)
    for source1_id, expected in truth_sets.items():
        group = groups.get(source1_id)
        if group is None:
            predicted_counts = np.zeros(len(threshold_values), dtype=np.int64)
            true_positive_counts = predicted_counts
        else:
            pair_scores = group["score"].to_numpy(dtype=np.float64)
            sorted_scores = np.sort(pair_scores)
            predicted_counts = len(sorted_scores) - np.searchsorted(
                sorted_scores, threshold_values, side="left"
            )
            expected_mask = group["candidate_entity_id"].isin(expected).to_numpy()
            sorted_true_scores = np.sort(pair_scores[expected_mask])
            true_positive_counts = len(sorted_true_scores) - np.searchsorted(
                sorted_true_scores, threshold_values, side="left"
            )

        if not expected:
            score_sums += predicted_counts == 0
            continue
        denominator = beta_sq * len(expected) + predicted_counts
        score_sums += np.divide(
            (1.0 + beta_sq) * true_positive_counts,
            denominator,
            out=np.zeros(len(threshold_values), dtype=np.float64),
            where=denominator > 0,
        )

    table = pd.DataFrame(
        {
            "threshold": threshold_values,
            "macro_f0_5": score_sums / len(truth_sets),
        }
    )
    best = table.sort_values(
        ["macro_f0_5", "threshold"], ascending=[False, False], kind="mergesort"
    ).iloc[0]
    return float(best["threshold"]), float(best["macro_f0_5"]), table
