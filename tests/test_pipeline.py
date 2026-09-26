from __future__ import annotations

import pandas as pd

from mlchallenge.candidates import CandidateConfig
from mlchallenge.config import (
    ExperimentConfig,
    PathConfig,
    ThresholdConfig,
    ValidationConfig,
)
from mlchallenge.contracts import ChallengeTestData, TrainingData
from mlchallenge.modeling import ModelConfig
from mlchallenge.pipeline import (
    cross_validate,
    load_model_bundle,
    predict_with_bundle,
    save_model_bundle,
    train_cv_ensemble,
)


def _pipeline_data() -> TrainingData:
    source1_rows = []
    source2_rows = []
    source3_rows = []
    truth_rows = []
    for index in range(8):
        source1_id = f"S1-{index:03d}"
        source1_rows.append((source1_id, f"Business {index}", f"{10 + index} Main Street", "US"))
        if index < 6:
            source2_id = f"S2-{index:03d}"
            source3_id = f"S3-{index:03d}"
            source2_rows.append(
                (source2_id, f"Business Number {index}", f"{10 + index} Main St", "US")
            )
            source3_rows.append(
                (source3_id, f"Business {index} LLC", f"Main Street {10 + index}", "US")
            )
            truth_rows.append((source1_id, f"{source2_id},{source3_id}"))
        else:
            truth_rows.append((source1_id, ""))
    source2_rows.extend(
        [
            ("S2-900", "Unrelated Alpha", "900 Remote Road", "US"),
            ("S2-901", "Unrelated Beta", "901 Remote Road", "US"),
        ]
    )
    source3_rows.extend(
        [
            ("S3-900", "Different Gamma", "902 Remote Road", "US"),
            ("S3-901", "Different Delta", "903 Remote Road", "US"),
        ]
    )
    columns = ("entity_id", "business_name", "business_address", "country")
    return TrainingData(
        pd.DataFrame(source1_rows, columns=columns, dtype="string"),
        pd.DataFrame(source2_rows, columns=columns, dtype="string"),
        pd.DataFrame(source3_rows, columns=columns, dtype="string"),
        pd.DataFrame(
            truth_rows,
            columns=("source1_entity_id", "matched_entity_ids"),
            dtype="string",
        ),
    )


def _pipeline_config() -> ExperimentConfig:
    return ExperimentConfig(
        paths=PathConfig(),
        candidates=CandidateConfig(
            method="multiview_char_tfidf",
            top_k_per_source=10,
            max_candidates_per_source=20,
            max_features=None,
        ),
        validation=ValidationConfig(n_outer_splits=2, n_inner_splits=2, seed=7),
        model=ModelConfig(max_iter=300, seed=7),
        threshold=ThresholdConfig(minimum=0.2, maximum=0.8, steps=4),
    )


def test_nested_cv_training_bundle_and_submission(tmp_path) -> None:
    data = _pipeline_data()
    config = _pipeline_config()
    report, pair_oof, entity_oof = cross_validate(data, config)
    assert len(report["outer_folds"]) == 2
    assert len(entity_oof) == len(data.source1)
    assert pair_oof["score"].between(0, 1).all()

    bundle, training_oof, threshold_table = train_cv_ensemble(data, config)
    assert len(bundle.matchers) == 2
    assert training_oof["score"].between(0, 1).all()
    assert len(threshold_table) == 4
    model_path = tmp_path / "model.joblib"
    save_model_bundle(bundle, model_path)
    loaded = load_model_bundle(model_path)

    test_data = ChallengeTestData(data.source1, data.source2, data.source3)
    inference = predict_with_bundle(loaded, test_data, tmp_path / "output")
    assert inference["test_source1_rows"] == len(data.source1)
    assert (tmp_path / "output" / "matching_results.tsv").is_file()
    assert (tmp_path / "output" / "candidate_pairs.tsv").is_file()
