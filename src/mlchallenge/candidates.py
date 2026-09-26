"""Transparent candidate generation and blocking diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from mlchallenge.contracts import truth_mapping
from mlchallenge.normalization import normalize_text

BLOCKING_SCORE_COLUMNS = (
    "combined_blocking_score",
    "name_blocking_score",
    "address_blocking_score",
)

CANDIDATE_COLUMNS = (
    "source1_entity_id",
    "candidate_entity_id",
    "candidate_source",
    "blocking_score",
    *BLOCKING_SCORE_COLUMNS,
)


@dataclass(frozen=True)
class CandidateConfig:
    method: str = "combined_char_tfidf"
    top_k_per_source: int = 50
    max_candidates_per_source: int | None = 100
    analyzer: str = "char_wb"
    ngram_min: int = 2
    ngram_max: int = 5
    min_df: int = 1
    max_features: int | None = 500_000
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.method not in {
            "combined_char_tfidf",
            "exact_char_tfidf",  # Backward-compatible name from the reference config.
            "multiview_char_tfidf",
        }:
            raise ValueError(f"unknown candidate-generation method: {self.method}")
        if self.top_k_per_source < 1:
            raise ValueError("top_k_per_source must be positive")
        if self.max_candidates_per_source is not None and self.max_candidates_per_source < 1:
            raise ValueError("max_candidates_per_source must be positive or None")
        if self.ngram_min < 1 or self.ngram_max < self.ngram_min:
            raise ValueError("invalid n-gram range")
        if self.n_jobs == 0:
            raise ValueError("n_jobs cannot be zero")

    @property
    def views(self) -> tuple[str, ...]:
        if self.method == "multiview_char_tfidf":
            return ("combined", "name", "address")
        return ("combined",)


@dataclass(frozen=True)
class CandidateTextCache:
    """Row-local normalized retrieval text reusable across CV partitions."""

    views: tuple[str, ...]
    source1: pd.DataFrame
    source2: pd.DataFrame
    source3: pd.DataFrame


def _text_views(frame: pd.DataFrame, views: tuple[str, ...]) -> dict[str, pd.Series]:
    """Build requested retrieval views while normalizing each field only once."""

    requested = set(views)
    names = frame["business_name"].map(normalize_text) if requested & {"combined", "name"} else None
    addresses = (
        frame["business_address"].map(normalize_text)
        if requested & {"combined", "address"}
        else None
    )
    result: dict[str, pd.Series] = {}
    if "combined" in requested:
        assert names is not None and addresses is not None
        values = [
            "" if not name and not address else f"name {name} address {address}".strip()
            for name, address in zip(names, addresses, strict=True)
        ]
        result["combined"] = pd.Series(values, index=frame.index, dtype="string")
    if "name" in requested:
        assert names is not None
        result["name"] = names.astype("string")
    if "address" in requested:
        assert addresses is not None
        result["address"] = addresses.astype("string")
    return result


def _indexed_text_views(frame: pd.DataFrame, views: tuple[str, ...]) -> pd.DataFrame:
    text = _text_views(frame, views)
    return pd.DataFrame(
        {view: text[view].to_numpy() for view in views},
        index=pd.Index(frame["entity_id"].astype(str), name="entity_id"),
    )


def prepare_candidate_text_cache(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: CandidateConfig,
) -> CandidateTextCache:
    """Normalize retrieval text once; these row-local transforms cannot leak fold labels."""

    return CandidateTextCache(
        views=config.views,
        source1=_indexed_text_views(source1, config.views),
        source2=_indexed_text_views(source2, config.views),
        source3=_indexed_text_views(source3, config.views),
    )


def _select_cached_views(
    cache: pd.DataFrame,
    frame: pd.DataFrame,
    views: tuple[str, ...],
) -> dict[str, pd.Series]:
    entity_ids = frame["entity_id"].astype(str)
    positions = cache.index.get_indexer(entity_ids)
    if (positions < 0).any():
        missing = entity_ids.iloc[np.flatnonzero(positions < 0)].tolist()
        raise ValueError(f"candidate text cache is missing entity IDs: {missing[:10]}")
    selected = cache.iloc[positions]
    return {
        view: pd.Series(selected[view].to_numpy(), index=frame.index, dtype="string")
        for view in views
    }


def _candidates_for_view(
    source1: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    query_text: pd.Series,
    target_text: pd.Series,
    target_source: str,
    view: str,
    config: CandidateConfig,
) -> pd.DataFrame:
    if source1.empty or targets.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    target_text = target_text.fillna("")
    query_text = query_text.fillna("")
    target_mask = target_text.ne("").to_numpy()
    query_mask = query_text.ne("").to_numpy()
    if not target_mask.any() or not query_mask.any():
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)

    filtered_targets = targets.loc[target_mask].reset_index(drop=True)
    filtered_target_text = target_text.loc[target_mask].reset_index(drop=True)
    vectorizer = TfidfVectorizer(
        analyzer=config.analyzer,
        ngram_range=(config.ngram_min, config.ngram_max),
        min_df=config.min_df,
        max_features=config.max_features,
        dtype=np.float32,
        sublinear_tf=True,
    )
    try:
        target_matrix = vectorizer.fit_transform(filtered_target_text)
    except ValueError as exc:
        if "empty vocabulary" in str(exc).lower():
            return pd.DataFrame(columns=CANDIDATE_COLUMNS)
        raise
    query_positions = np.flatnonzero(query_mask)
    query_matrix = vectorizer.transform(query_text.iloc[query_positions])
    nonzero_query_mask = np.asarray(query_matrix.getnnz(axis=1)).ravel() > 0
    if not nonzero_query_mask.any():
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    query_positions = query_positions[nonzero_query_mask]
    query_matrix = query_matrix[nonzero_query_mask]
    neighbor_count = min(config.top_k_per_source, len(filtered_targets))
    search = NearestNeighbors(
        n_neighbors=neighbor_count,
        algorithm="brute",
        metric="cosine",
        n_jobs=config.n_jobs,
    ).fit(target_matrix)
    distances, indices = search.kneighbors(query_matrix, return_distance=True)

    source1_ids = source1["entity_id"].astype(str).to_numpy()[query_positions]
    target_ids = filtered_targets["entity_id"].astype(str).to_numpy()
    scores = np.clip(1.0 - distances, 0.0, 1.0).astype(np.float32, copy=False)
    view_column = f"{view}_blocking_score"
    result = pd.DataFrame(
        {
            "source1_entity_id": np.repeat(source1_ids, neighbor_count),
            "candidate_entity_id": target_ids[indices.reshape(-1)],
            "candidate_source": target_source,
            view_column: scores.reshape(-1),
        }
    )
    for column in BLOCKING_SCORE_COLUMNS:
        if column not in result:
            result[column] = 0.0
    result["blocking_score"] = result.loc[:, BLOCKING_SCORE_COLUMNS].max(axis=1)
    return result.loc[:, CANDIDATE_COLUMNS]


def generate_candidates(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    config: CandidateConfig,
    *,
    text_cache: CandidateTextCache | None = None,
) -> pd.DataFrame:
    """Retrieve top-k candidates independently from Sources 2 and 3.

    Country is deliberately not a hard filter. It becomes a downstream comparison feature, which
    avoids assuming that labels are complete/consistent and supports unseen country strings.
    """

    if text_cache is not None and text_cache.views != config.views:
        raise ValueError(
            f"candidate text cache contains views {text_cache.views}; expected {config.views}"
        )
    query_views = (
        _select_cached_views(text_cache.source1, source1, config.views)
        if text_cache is not None
        else _text_views(source1, config.views)
    )
    parts: list[pd.DataFrame] = []
    for target_source, targets in (("source2", source2), ("source3", source3)):
        target_views = (
            _select_cached_views(getattr(text_cache, target_source), targets, config.views)
            if text_cache is not None
            else _text_views(targets, config.views)
        )
        for view in config.views:
            parts.append(
                _candidates_for_view(
                    source1,
                    targets,
                    query_text=query_views[view],
                    target_text=target_views[view],
                    target_source=target_source,
                    view=view,
                    config=config,
                )
            )
    result = pd.concat(parts, ignore_index=True)
    if result.empty:
        return pd.DataFrame(columns=CANDIDATE_COLUMNS)
    result = result.groupby(
        ["source1_entity_id", "candidate_entity_id", "candidate_source"],
        as_index=False,
        sort=False,
    ).agg({column: "max" for column in BLOCKING_SCORE_COLUMNS})
    result["blocking_score"] = result.loc[:, BLOCKING_SCORE_COLUMNS].max(axis=1)
    result = result.sort_values(
        ["source1_entity_id", "candidate_source", "blocking_score", "candidate_entity_id"],
        ascending=[True, True, False, True],
        kind="mergesort",
    )
    if config.max_candidates_per_source is not None:
        keep = (
            result.groupby(["source1_entity_id", "candidate_source"], sort=False).cumcount()
            < config.max_candidates_per_source
        )
        result = result.loc[keep]
    return result.loc[:, CANDIDATE_COLUMNS].reset_index(drop=True)


def blocking_diagnostics(
    candidates: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    possible_target_count: int,
) -> dict[str, float | int]:
    truth = truth_mapping(ground_truth)
    candidate_map = {
        source1_id: frozenset(group["candidate_entity_id"].astype(str))
        for source1_id, group in candidates.groupby("source1_entity_id", sort=False)
    }
    positive_total = sum(len(matches) for matches in truth.values())
    positive_found = sum(
        len(matches & candidate_map.get(source1_id, frozenset()))
        for source1_id, matches in truth.items()
    )
    complete_entities = sum(
        matches <= candidate_map.get(source1_id, frozenset())
        for source1_id, matches in truth.items()
    )
    matched_truth = {key: value for key, value in truth.items() if value}
    complete_matched_entities = sum(
        matches <= candidate_map.get(source1_id, frozenset())
        for source1_id, matches in matched_truth.items()
    )
    counts = candidates.groupby("source1_entity_id").size().reindex(truth, fill_value=0)
    all_pairs = len(truth) * possible_target_count
    return {
        "source1_entities": len(truth),
        "candidate_pairs": len(candidates),
        "positive_pairs": positive_total,
        "positive_pairs_retrieved": positive_found,
        "positive_pair_recall": positive_found / positive_total if positive_total else 1.0,
        "entities_with_all_matches_retrieved": complete_entities,
        "complete_entity_recall": complete_entities / len(truth) if truth else 0.0,
        "matched_entities_with_all_matches_retrieved": complete_matched_entities,
        "matched_entity_complete_recall": (
            complete_matched_entities / len(matched_truth) if matched_truth else 1.0
        ),
        "candidates_per_source1_mean": float(counts.mean()) if len(counts) else 0.0,
        "candidates_per_source1_median": float(counts.median()) if len(counts) else 0.0,
        "candidates_per_source1_p95": float(counts.quantile(0.95)) if len(counts) else 0.0,
        "candidates_per_source1_max": int(counts.max()) if len(counts) else 0,
        "reduction_ratio": 1.0 - (len(candidates) / all_pairs) if all_pairs else 1.0,
    }
