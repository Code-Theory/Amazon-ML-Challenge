"""Auditable pairwise features for a non-neural baseline."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher

import numpy as np
import pandas as pd

try:
    from rapidfuzz import process as rapidfuzz_process
    from rapidfuzz.fuzz import ratio, token_set_ratio
except ImportError:  # pragma: no cover - exercised only in minimal diagnostic environments
    rapidfuzz_process = None

    def ratio(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left, right, autojunk=False).ratio()

    def token_set_ratio(left: str, right: str) -> float:
        left_tokens = set(left.split())
        right_tokens = set(right.split())
        intersection = sorted(left_tokens & right_tokens)
        left_only = sorted(left_tokens - right_tokens)
        right_only = sorted(right_tokens - left_tokens)
        common = " ".join(intersection)
        left_combined = " ".join((*intersection, *left_only))
        right_combined = " ".join((*intersection, *right_only))
        return max(
            ratio(common, left_combined),
            ratio(common, right_combined),
            ratio(left_combined, right_combined),
        )


from mlchallenge.normalization import digit_set, normalize_text

FEATURE_COLUMNS = (
    "blocking_score",
    "combined_blocking_score",
    "name_blocking_score",
    "address_blocking_score",
    "name_ratio",
    "name_token_set_ratio",
    "name_token_jaccard",
    "name_token_containment",
    "name_exact",
    "name_missing_either",
    "address_ratio",
    "address_token_set_ratio",
    "address_token_jaccard",
    "address_token_containment",
    "address_exact",
    "address_missing_either",
    "address_digit_jaccard",
    "address_digits_exact",
    "address_digits_conflict",
    "country_exact",
    "country_mismatch",
    "country_missing_either",
    "candidate_is_source3",
    "name_length_ratio",
    "address_length_ratio",
    "name_and_address_exact",
)


@dataclass(frozen=True)
class PairFeatureCache:
    """Label-free entity representations reusable across CV partitions."""

    source1: pd.DataFrame
    targets: pd.DataFrame


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    # Missing evidence is not positive evidence. Returning 1.0 for two empty values creates a
    # particularly damaging false-match signal on this precision-heavy task.
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _containment(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def _tokens_from_normalized(value: str) -> frozenset[str]:
    return frozenset(value.split()) if value else frozenset()


def _prepare_records(frame: pd.DataFrame, *, id_column: str, prefix: str) -> pd.DataFrame:
    """Precompute reusable entity representations before expanding candidate pairs."""

    names = frame["business_name"].map(normalize_text)
    addresses = frame["business_address"].map(normalize_text)
    countries = frame["country"].map(normalize_text)
    return pd.DataFrame(
        {
            id_column: frame["entity_id"].astype(str),
            f"{prefix}_name": names,
            f"{prefix}_address": addresses,
            f"{prefix}_country": countries,
            f"{prefix}_name_compact": names.str.replace(" ", "", regex=False),
            f"{prefix}_address_compact": addresses.str.replace(" ", "", regex=False),
            f"{prefix}_name_tokens": names.map(_tokens_from_normalized),
            f"{prefix}_address_tokens": addresses.map(_tokens_from_normalized),
            f"{prefix}_address_digits": frame["business_address"].map(digit_set),
        }
    )


def prepare_pair_feature_cache(
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> PairFeatureCache:
    """Normalize each entity once for reuse across fold-specific candidate pairs.

    Every transformation is deterministic and row-local: no corpus or target statistics are
    learned, so reusing these representations across validation partitions cannot leak labels.
    """

    targets = pd.concat([source2, source3], ignore_index=True)
    return PairFeatureCache(
        source1=_prepare_records(source1, id_column="source1_entity_id", prefix="s1"),
        targets=_prepare_records(
            targets,
            id_column="candidate_entity_id",
            prefix="candidate",
        ),
    )


def _fuzzy_scores(
    left: pd.Series,
    right: pd.Series,
    *,
    scorer: object,
    n_jobs: int,
) -> np.ndarray:
    left_values = left.astype(str).to_numpy()
    right_values = right.astype(str).to_numpy()
    valid = (left_values != "") & (right_values != "")
    scores = np.zeros(len(left_values), dtype=np.float64)
    if not valid.any():
        return scores
    if rapidfuzz_process is not None:
        scores[valid] = (
            rapidfuzz_process.cpdist(
                left_values[valid].tolist(),
                right_values[valid].tolist(),
                scorer=scorer,
                dtype=np.float32,
                workers=n_jobs,
            )
            / 100.0
        )
    else:  # pragma: no cover - only used without the declared RapidFuzz dependency
        scores[valid] = np.fromiter(
            (
                scorer(left_value, right_value) / 100.0
                for left_value, right_value in zip(
                    left_values[valid], right_values[valid], strict=True
                )
            ),
            dtype=np.float64,
            count=int(valid.sum()),
        )
    return scores


def _set_scores(left: pd.Series, right: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    count = len(left)
    jaccard = np.fromiter(
        (
            _jaccard(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        ),
        dtype=np.float64,
        count=count,
    )
    containment = np.fromiter(
        (
            _containment(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        ),
        dtype=np.float64,
        count=count,
    )
    return jaccard, containment


def _length_ratios(left: pd.Series, right: pd.Series) -> np.ndarray:
    left_lengths = left.str.len().to_numpy(dtype=np.float64)
    right_lengths = right.str.len().to_numpy(dtype=np.float64)
    maximum = np.maximum(left_lengths, right_lengths)
    return np.divide(
        np.minimum(left_lengths, right_lengths),
        maximum,
        out=np.zeros(len(left), dtype=np.float64),
        where=maximum > 0,
    )


def build_pair_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
    *,
    n_jobs: int = 1,
    cache: PairFeatureCache | None = None,
) -> pd.DataFrame:
    """Join candidate records and compute numeric, inspectable similarities."""

    output_columns = ("source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS)
    if candidates.empty:
        return pd.DataFrame(columns=output_columns)

    prepared = cache or prepare_pair_feature_cache(source1, source2, source3)
    left = prepared.source1
    right = prepared.targets
    merged = candidates.merge(left, on="source1_entity_id", how="left", validate="many_to_one")
    merged = merged.merge(right, on="candidate_entity_id", how="left", validate="many_to_one")
    if merged[["s1_name", "candidate_name"]].isna().any().any():
        raise ValueError("candidate pairs reference IDs absent from source records")

    left_name = merged["s1_name"]
    right_name = merged["candidate_name"]
    left_address = merged["s1_address"]
    right_address = merged["candidate_address"]
    left_country = merged["s1_country"]
    right_country = merged["candidate_country"]

    name_jaccard, name_containment = _set_scores(
        merged["s1_name_tokens"], merged["candidate_name_tokens"]
    )
    address_jaccard, address_containment = _set_scores(
        merged["s1_address_tokens"], merged["candidate_address_tokens"]
    )
    left_digits = merged["s1_address_digits"]
    right_digits = merged["candidate_address_digits"]
    pair_count = len(merged)
    name_exact = left_name.ne("") & merged["s1_name_compact"].eq(merged["candidate_name_compact"])
    address_exact = left_address.ne("") & merged["s1_address_compact"].eq(
        merged["candidate_address_compact"]
    )
    country_present = left_country.ne("") & right_country.ne("")

    blocking_score = merged["blocking_score"].to_numpy(dtype=np.float64)
    output = pd.DataFrame(
        {
            "source1_entity_id": merged["source1_entity_id"].astype(str),
            "candidate_entity_id": merged["candidate_entity_id"].astype(str),
            "blocking_score": blocking_score,
            "combined_blocking_score": merged.get(
                "combined_blocking_score", merged["blocking_score"]
            ).to_numpy(dtype=np.float64),
            "name_blocking_score": merged.get(
                "name_blocking_score", pd.Series(0.0, index=merged.index)
            ).to_numpy(dtype=np.float64),
            "address_blocking_score": merged.get(
                "address_blocking_score", pd.Series(0.0, index=merged.index)
            ).to_numpy(dtype=np.float64),
            "name_ratio": _fuzzy_scores(left_name, right_name, scorer=ratio, n_jobs=n_jobs),
            "name_token_set_ratio": _fuzzy_scores(
                left_name, right_name, scorer=token_set_ratio, n_jobs=n_jobs
            ),
            "name_token_jaccard": name_jaccard,
            "name_token_containment": name_containment,
            "name_exact": name_exact.to_numpy(dtype=np.float64),
            "name_missing_either": (~(left_name.ne("") & right_name.ne(""))).to_numpy(
                dtype=np.float64
            ),
            "address_ratio": _fuzzy_scores(
                left_address, right_address, scorer=ratio, n_jobs=n_jobs
            ),
            "address_token_set_ratio": _fuzzy_scores(
                left_address, right_address, scorer=token_set_ratio, n_jobs=n_jobs
            ),
            "address_token_jaccard": address_jaccard,
            "address_token_containment": address_containment,
            "address_exact": address_exact.to_numpy(dtype=np.float64),
            "address_missing_either": (~(left_address.ne("") & right_address.ne(""))).to_numpy(
                dtype=np.float64
            ),
            "address_digit_jaccard": np.fromiter(
                (
                    _jaccard(left_value, right_value)
                    for left_value, right_value in zip(left_digits, right_digits, strict=True)
                ),
                dtype=np.float64,
                count=pair_count,
            ),
            "address_digits_exact": np.fromiter(
                (
                    float(bool(left_value) and left_value == right_value)
                    for left_value, right_value in zip(left_digits, right_digits, strict=True)
                ),
                dtype=np.float64,
                count=pair_count,
            ),
            "address_digits_conflict": np.fromiter(
                (
                    float(
                        bool(left_value)
                        and bool(right_value)
                        and left_value.isdisjoint(right_value)
                    )
                    for left_value, right_value in zip(left_digits, right_digits, strict=True)
                ),
                dtype=np.float64,
                count=pair_count,
            ),
            "country_exact": (country_present & left_country.eq(right_country)).to_numpy(
                dtype=np.float64
            ),
            "country_mismatch": (country_present & left_country.ne(right_country)).to_numpy(
                dtype=np.float64
            ),
            "country_missing_either": (~country_present).to_numpy(dtype=np.float64),
            "candidate_is_source3": merged["candidate_entity_id"]
            .astype(str)
            .str.startswith("S3-")
            .to_numpy(dtype=np.float64),
            "name_length_ratio": _length_ratios(left_name, right_name),
            "address_length_ratio": _length_ratios(left_address, right_address),
            "name_and_address_exact": (name_exact & address_exact).to_numpy(dtype=np.float64),
        }
    )
    return output.loc[:, output_columns]
