"""Dataset fingerprinting and descriptive audit with no model fitting."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd

from mlchallenge.contracts import (
    TrainingData,
    load_test_data,
    load_training_data,
    parse_id_list,
)
from mlchallenge.normalization import normalize_text

ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class _NormalizedRecords:
    name: pd.Series
    address: pd.Series
    country: pd.Series


@dataclass(frozen=True)
class _NormalizedKeys:
    name_keys: frozenset[str]
    address_keys: frozenset[str]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_records(frame: pd.DataFrame) -> _NormalizedRecords:
    name = frame["business_name"].map(normalize_text)
    address = frame["business_address"].map(normalize_text)
    country = frame["country"].map(normalize_text)
    return _NormalizedRecords(
        name=name,
        address=address,
        country=country,
    )


def _normalized_keys(normalized: _NormalizedRecords) -> _NormalizedKeys:
    return _NormalizedKeys(
        name_keys=frozenset(normalized.name.loc[normalized.name.ne("")]),
        address_keys=frozenset(normalized.address.loc[normalized.address.ne("")]),
    )


def _record_summary(frame: pd.DataFrame, normalized: _NormalizedRecords) -> dict[str, Any]:
    nonempty_names = normalized.name.loc[normalized.name.ne("")]
    nonempty_addresses = normalized.address.loc[normalized.address.ne("")]
    return {
        "rows": int(len(frame)),
        "unique_entity_ids": int(frame["entity_id"].nunique()),
        "empty_business_name": int(normalized.name.eq("").sum()),
        "empty_business_address": int(normalized.address.eq("").sum()),
        "empty_country": int(normalized.country.eq("").sum()),
        "countries": {
            str(key): int(value)
            for key, value in frame["country"].astype(str).value_counts(dropna=False).items()
        },
        "normalized_exact_duplicate_names": int(nonempty_names.duplicated(keep=False).sum()),
        "normalized_exact_duplicate_addresses": int(
            nonempty_addresses.duplicated(keep=False).sum()
        ),
        "name_length": _numeric_summary(normalized.name.str.len()),
        "address_length": _numeric_summary(normalized.address.str.len()),
    }


def _numeric_summary(series: pd.Series) -> dict[str, float]:
    if series.empty:
        return {key: 0.0 for key in ("min", "median", "p95", "max")}
    return {
        "min": float(series.min()),
        "median": float(series.median()),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
    }


def _truth_summary(training: TrainingData) -> dict[str, Any]:
    parsed_matches = [parse_id_list(value) for value in training.ground_truth["matched_entity_ids"]]
    counts = pd.Series((len(matches) for matches in parsed_matches), dtype="int64")
    target_owners: dict[str, int] = {}
    source2_pairs = 0
    source3_pairs = 0
    for matches in parsed_matches:
        for target_id in matches:
            target_owners[target_id] = target_owners.get(target_id, 0) + 1
            source2_pairs += target_id.startswith("S2-")
            source3_pairs += target_id.startswith("S3-")
    return {
        "source1_entities": int(len(counts)),
        "singleton_entities": int(counts.eq(0).sum()),
        "matched_entities": int(counts.gt(0).sum()),
        "positive_pairs": int(counts.sum()),
        "positive_pairs_source2": int(source2_pairs),
        "positive_pairs_source3": int(source3_pairs),
        "matches_per_source1": _numeric_summary(counts),
        "targets_with_multiple_source1_owners": int(
            sum(owner_count > 1 for owner_count in target_owners.values())
        ),
    }


def _cross_source_summary(
    source1: _NormalizedKeys,
    source2: _NormalizedKeys,
    source3: _NormalizedKeys,
) -> dict[str, Any]:
    frames = {"source1": source1, "source2": source2, "source3": source3}
    result: dict[str, Any] = {}
    for left_name, right_name in (
        ("source1", "source2"),
        ("source1", "source3"),
        ("source2", "source3"),
    ):
        left = frames[left_name]
        right = frames[right_name]
        result[f"{left_name}_vs_{right_name}"] = {
            "shared_unique_normalized_names": len(left.name_keys & right.name_keys),
            "shared_unique_normalized_addresses": len(left.address_keys & right.address_keys),
        }
    return result


def _progress(callback: ProgressCallback | None, message: str, started: float) -> None:
    if callback is not None:
        callback(f"Audit: {message} ({perf_counter() - started:.1f}s)")


def audit_data(
    data_root: str | Path,
    *,
    include_hashes: bool = True,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    root = Path(data_root)
    started = perf_counter()

    if progress is not None:
        progress("Audit: loading and validating training data...")
    training = load_training_data(root)
    _progress(progress, "training data loaded", started)

    if progress is not None:
        progress("Audit: loading and validating test data...")
    test = load_test_data(root)
    _progress(progress, "test data loaded", started)

    frames = {
        "train": {
            "source1": training.source1,
            "source2": training.source2,
            "source3": training.source3,
        },
        "test": {
            "source1": test.source1,
            "source2": test.source2,
            "source3": test.source3,
        },
    }
    summaries: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "test": {}}
    overlap_keys: dict[str, dict[str, _NormalizedKeys]] = {"train": {}, "test": {}}
    for split, split_frames in frames.items():
        for source, frame in split_frames.items():
            if progress is not None:
                progress(f"Audit: normalizing {split}/{source} ({len(frame):,} rows)...")
            normalized = _normalize_records(frame)
            summaries[split][source] = _record_summary(frame, normalized)
            overlap_keys[split][source] = _normalized_keys(normalized)
            del normalized
    _progress(progress, "text normalization complete", started)

    files = sorted((*root.joinpath("train").glob("*.tsv"), *root.joinpath("test").glob("*.tsv")))
    file_report: dict[str, dict[str, str | int | None]] = {}
    for path in files:
        relative = str(path.relative_to(root))
        digest = None
        if include_hashes:
            if progress is not None:
                size_mib = path.stat().st_size / 1_048_576
                progress(f"Audit: hashing {relative} ({size_mib:.1f} MiB)...")
            digest = sha256_file(path)
        file_report[relative] = {"bytes": path.stat().st_size, "sha256": digest}
    _progress(progress, "file fingerprinting complete", started)

    train_keys = overlap_keys["train"]
    test_keys = overlap_keys["test"]
    report = {
        "audit_version": 1,
        "hashing_enabled": include_hashes,
        "files": file_report,
        "train": {
            "source1": summaries["train"]["source1"],
            "source2": summaries["train"]["source2"],
            "source3": summaries["train"]["source3"],
            "ground_truth": _truth_summary(training),
            "cross_source_exact_overlap": _cross_source_summary(
                train_keys["source1"],
                train_keys["source2"],
                train_keys["source3"],
            ),
            "possible_cross_source_pairs": int(
                len(training.source1) * (len(training.source2) + len(training.source3))
            ),
        },
        "test": {
            "source1": summaries["test"]["source1"],
            "source2": summaries["test"]["source2"],
            "source3": summaries["test"]["source3"],
            "cross_source_exact_overlap": _cross_source_summary(
                test_keys["source1"],
                test_keys["source2"],
                test_keys["source3"],
            ),
            "possible_cross_source_pairs": int(
                len(test.source1) * (len(test.source2) + len(test.source3))
            ),
        },
    }
    _progress(progress, "report complete", started)
    return report


def write_audit(report: dict[str, Any], output: str | Path) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
