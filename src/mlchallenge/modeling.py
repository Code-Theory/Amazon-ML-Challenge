"""Conservative baseline model and candidate labeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlchallenge.contracts import truth_mapping
from mlchallenge.features import FEATURE_COLUMNS


def label_candidate_pairs(
    feature_frame: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> np.ndarray:
    truth = truth_mapping(ground_truth)
    labels = [
        int(str(row.candidate_entity_id) in truth[str(row.source1_entity_id)])
        for row in feature_frame.itertuples(index=False)
    ]
    return np.asarray(labels, dtype=np.int8)


@dataclass(frozen=True)
class ModelConfig:
    """Small, reviewable model search surface for pair classification."""

    family: str = "logistic_regression"
    regularization_c: float = 1.0
    max_iter: int = 2_000
    class_weight: str | None = "balanced"
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 20
    l2_regularization: float = 1.0
    seed: int = 20260925
    entity_balanced_weights: bool = True

    def __post_init__(self) -> None:
        if self.family not in {"logistic_regression", "hist_gradient_boosting"}:
            raise ValueError(f"unknown model family: {self.family}")
        if self.regularization_c <= 0:
            raise ValueError("regularization_c must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")


def entity_balanced_sample_weights(feature_frame: pd.DataFrame) -> np.ndarray:
    """Give each Source 1 entity equal total training weight for the macro metric."""

    counts = feature_frame.groupby("source1_entity_id")["source1_entity_id"].transform("size")
    weights = 1.0 / counts.to_numpy(dtype=np.float64)
    return weights / weights.mean()


@dataclass
class PairMatcher:
    """Deterministic pair classifier with feature-order and weight safeguards."""

    config: ModelConfig

    def __post_init__(self) -> None:
        if self.config.family == "logistic_regression":
            classifier = LogisticRegression(
                C=self.config.regularization_c,
                class_weight=self.config.class_weight,
                max_iter=self.config.max_iter,
                random_state=self.config.seed,
            )
            self.pipeline = Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("classifier", classifier),
                ]
            )
        else:
            self.pipeline = Pipeline(
                [
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            learning_rate=self.config.learning_rate,
                            max_iter=self.config.max_iter,
                            max_leaf_nodes=self.config.max_leaf_nodes,
                            min_samples_leaf=self.config.min_samples_leaf,
                            l2_regularization=self.config.l2_regularization,
                            class_weight=self.config.class_weight,
                            early_stopping=False,
                            random_state=self.config.seed,
                        ),
                    ),
                ]
            )

    def fit(self, feature_frame: pd.DataFrame, labels: np.ndarray) -> PairMatcher:
        unique = np.unique(labels)
        if not np.array_equal(unique, np.array([0, 1], dtype=unique.dtype)):
            raise ValueError(f"training candidates must contain both classes; observed {unique}")
        if not np.isfinite(feature_frame.loc[:, FEATURE_COLUMNS].to_numpy()).all():
            raise ValueError("training features contain NaN or infinite values")
        fit_params = {}
        if self.config.entity_balanced_weights:
            fit_params["classifier__sample_weight"] = entity_balanced_sample_weights(feature_frame)
        self.pipeline.fit(feature_frame.loc[:, FEATURE_COLUMNS], labels, **fit_params)
        return self

    def predict_scores(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        if not np.isfinite(feature_frame.loc[:, FEATURE_COLUMNS].to_numpy()).all():
            raise ValueError("inference features contain NaN or infinite values")
        probabilities = self.pipeline.predict_proba(feature_frame.loc[:, FEATURE_COLUMNS])[:, 1]
        return pd.DataFrame(
            {
                "source1_entity_id": feature_frame["source1_entity_id"].astype(str),
                "candidate_entity_id": feature_frame["candidate_entity_id"].astype(str),
                "score": probabilities,
            }
        )


@dataclass
class BaselineMatcher:
    """Backward-compatible constructor for the original logistic-regression baseline."""

    regularization_c: float = 1.0
    max_iter: int = 2_000
    seed: int = 20260925

    def __post_init__(self) -> None:
        self.matcher = PairMatcher(
            ModelConfig(
                family="logistic_regression",
                regularization_c=self.regularization_c,
                max_iter=self.max_iter,
                seed=self.seed,
            )
        )
        self.pipeline = self.matcher.pipeline

    def fit(self, feature_frame: pd.DataFrame, labels: np.ndarray) -> BaselineMatcher:
        self.matcher.fit(feature_frame, labels)
        return self

    def predict_scores(self, feature_frame: pd.DataFrame) -> pd.DataFrame:
        return self.matcher.predict_scores(feature_frame)


def build_matcher(config: ModelConfig) -> PairMatcher:
    return PairMatcher(config)
