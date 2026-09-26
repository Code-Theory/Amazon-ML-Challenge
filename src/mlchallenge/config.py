"""Typed experiment configuration loaded from versioned TOML files."""

from __future__ import annotations

import random
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mlchallenge.candidates import CandidateConfig
from mlchallenge.modeling import ModelConfig


@dataclass(frozen=True)
class PathConfig:
    data_root: str = "data"
    report_dir: str = "reports"
    artifact_dir: str = "artifacts"
    output_dir: str = "output"


@dataclass(frozen=True)
class ValidationConfig:
    n_outer_splits: int = 5
    n_inner_splits: int = 4
    seed: int = 20260925
    metric: str = "macro_f0.5"

    def __post_init__(self) -> None:
        if self.n_outer_splits < 2 or self.n_inner_splits < 2:
            raise ValueError("outer and inner split counts must both be at least 2")
        if self.metric != "macro_f0.5":
            raise ValueError("the challenge metric must be macro_f0.5")


@dataclass(frozen=True)
class ThresholdConfig:
    strategy: str = "inner_oof_grid"
    minimum: float = 0.05
    maximum: float = 0.95
    steps: int = 181

    def __post_init__(self) -> None:
        if self.strategy != "inner_oof_grid":
            raise ValueError("only leakage-safe inner_oof_grid threshold selection is supported")
        if not 0.0 <= self.minimum <= self.maximum <= 1.0:
            raise ValueError("threshold range must lie within [0, 1]")
        if self.steps < 2:
            raise ValueError("threshold grid requires at least two steps")

    def grid(self) -> np.ndarray:
        return np.linspace(self.minimum, self.maximum, self.steps, dtype=np.float64)


@dataclass(frozen=True)
class ExperimentConfig:
    paths: PathConfig
    candidates: CandidateConfig
    validation: ValidationConfig
    model: ModelConfig
    threshold: ThresholdConfig

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _section(document: dict[str, Any], name: str) -> dict[str, Any]:
    value = document.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"configuration section [{name}] must be a table")
    return dict(value)


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {config_path}")
    with config_path.open("rb") as stream:
        document = tomllib.load(stream)
    allowed_sections = {
        "paths",
        "candidate_generation",
        "validation",
        "model",
        "threshold_selection",
    }
    unknown_sections = set(document) - allowed_sections
    if unknown_sections:
        raise ValueError(f"unknown configuration sections: {sorted(unknown_sections)}")

    try:
        paths = PathConfig(**_section(document, "paths"))
        validation = ValidationConfig(**_section(document, "validation"))
        candidate_payload = _section(document, "candidate_generation")
        candidates = CandidateConfig(**candidate_payload)
        model_payload = _section(document, "model")
        model_payload.setdefault("seed", validation.seed)
        if model_payload.get("class_weight") == "none":
            model_payload["class_weight"] = None
        model = ModelConfig(**model_payload)
        threshold = ThresholdConfig(**_section(document, "threshold_selection"))
    except TypeError as exc:
        raise ValueError(f"invalid field in experiment configuration {config_path}: {exc}") from exc
    return ExperimentConfig(paths, candidates, validation, model, threshold)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
