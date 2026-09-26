from __future__ import annotations

import numpy as np
import pandas as pd

from mlchallenge.candidates import (
    BLOCKING_SCORE_COLUMNS,
    CANDIDATE_COLUMNS,
    CandidateConfig,
    blocking_diagnostics,
    generate_candidates,
)
from mlchallenge.features import FEATURE_COLUMNS, build_pair_features
from mlchallenge.modeling import (
    BaselineMatcher,
    ModelConfig,
    build_matcher,
    label_candidate_pairs,
)


def test_candidates_do_not_hard_filter_unseen_country(synthetic_frames) -> None:
    frames = synthetic_frames
    candidates = generate_candidates(
        frames["source1"],
        frames["source2"],
        frames["source3"],
        CandidateConfig(top_k_per_source=2, max_features=None),
    )
    france_candidates = candidates.loc[candidates["source1_entity_id"] == "S1-003"]
    assert set(france_candidates["candidate_source"]) == {"source2", "source3"}
    assert "S2-104" in set(france_candidates["candidate_entity_id"])
    assert "S3-204" in set(france_candidates["candidate_entity_id"])


def test_features_labels_model_and_blocking_diagnostics(synthetic_frames) -> None:
    frames = synthetic_frames
    candidates = generate_candidates(
        frames["source1"],
        frames["source2"],
        frames["source3"],
        CandidateConfig(top_k_per_source=3, max_features=None),
    )
    features = build_pair_features(
        candidates, frames["source1"], frames["source2"], frames["source3"]
    )
    assert tuple(features.columns[2:]) == FEATURE_COLUMNS
    assert np.isfinite(features.loc[:, FEATURE_COLUMNS].to_numpy()).all()
    labels = label_candidate_pairs(features, frames["truth"])
    assert set(labels) == {0, 1}
    model = BaselineMatcher().fit(features, labels)
    scores = model.predict_scores(features)
    assert scores["score"].between(0, 1).all()
    nonlinear = build_matcher(
        ModelConfig(
            family="hist_gradient_boosting",
            max_iter=5,
            min_samples_leaf=1,
            seed=7,
        )
    ).fit(features, labels)
    assert nonlinear.predict_scores(features)["score"].between(0, 1).all()
    diagnostics = blocking_diagnostics(
        candidates,
        frames["truth"],
        possible_target_count=len(frames["source2"]) + len(frames["source3"]),
    )
    assert diagnostics["positive_pair_recall"] == 1.0
    assert 0.0 <= diagnostics["reduction_ratio"] <= 1.0
    assert diagnostics["matched_entity_complete_recall"] == 1.0


def test_multiview_candidates_and_missing_values_are_safe(synthetic_frames) -> None:
    frames = synthetic_frames
    source1 = frames["source1"].copy()
    source2 = frames["source2"].copy()
    source1.loc[0, ["business_name", "business_address"]] = ""
    source2.loc[0, ["business_name", "business_address"]] = ""
    candidates = generate_candidates(
        source1,
        source2,
        frames["source3"],
        CandidateConfig(
            method="multiview_char_tfidf",
            top_k_per_source=2,
            max_candidates_per_source=4,
            max_features=None,
        ),
    )
    assert set(BLOCKING_SCORE_COLUMNS) <= set(candidates.columns)
    assert candidates.groupby(["source1_entity_id", "candidate_source"]).size().max() <= 4
    manual_pair = pd.DataFrame(
        [("S1-001", "S2-101", "source2", 0.0, 0.0, 0.0, 0.0)],
        columns=CANDIDATE_COLUMNS,
    )
    missing_pair = build_pair_features(manual_pair, source1, source2, frames["source3"]).iloc[0]
    assert missing_pair["name_ratio"] == 0.0
    assert missing_pair["name_token_jaccard"] == 0.0
    assert missing_pair["address_token_set_ratio"] == 0.0
    assert missing_pair["address_token_jaccard"] == 0.0
    assert missing_pair["name_length_ratio"] == 0.0
    assert missing_pair["address_length_ratio"] == 0.0
