"""Auditable pairwise features for a non-neural baseline."""

from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

try:
    from rapidfuzz.fuzz import ratio, token_set_ratio
except ImportError:  # pragma: no cover - exercised only in minimal diagnostic environments

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


from mlchallenge.normalization import compact_text, digit_set, normalize_text, token_set

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


def _jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    # Missing evidence is not positive evidence. Returning 1.0 for two empty values creates a
    # particularly damaging false-match signal on this precision-heavy task.
    if not left or not right:
        return 0.0
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _length_ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    maximum = max(len(left), len(right))
    return min(len(left), len(right)) / maximum if maximum else 0.0


def _fuzzy_ratio(left: str, right: str) -> float:
    return ratio(left, right) / 100.0 if left and right else 0.0


def _token_set_ratio(left: str, right: str) -> float:
    return token_set_ratio(left, right) / 100.0 if left and right else 0.0


def _containment(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))


def build_pair_features(
    candidates: pd.DataFrame,
    source1: pd.DataFrame,
    source2: pd.DataFrame,
    source3: pd.DataFrame,
) -> pd.DataFrame:
    """Join candidate records and compute numeric, inspectable similarities."""

    target = pd.concat([source2, source3], ignore_index=True)
    left = source1.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "s1_name",
            "business_address": "s1_address",
            "country": "s1_country",
        }
    )
    right = target.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "candidate_name",
            "business_address": "candidate_address",
            "country": "candidate_country",
        }
    )
    merged = candidates.merge(left, on="source1_entity_id", how="left", validate="many_to_one")
    merged = merged.merge(right, on="candidate_entity_id", how="left", validate="many_to_one")
    if merged[["s1_name", "candidate_name"]].isna().any().any():
        raise ValueError("candidate pairs reference IDs absent from source records")

    rows: list[dict[str, object]] = []
    for row in merged.itertuples(index=False):
        left_name = normalize_text(row.s1_name)
        right_name = normalize_text(row.candidate_name)
        left_address = normalize_text(row.s1_address)
        right_address = normalize_text(row.candidate_address)
        left_country = normalize_text(row.s1_country)
        right_country = normalize_text(row.candidate_country)
        left_name_tokens = token_set(left_name)
        right_name_tokens = token_set(right_name)
        left_address_tokens = token_set(left_address)
        right_address_tokens = token_set(right_address)
        left_digits = digit_set(row.s1_address)
        right_digits = digit_set(row.candidate_address)
        name_exact = bool(left_name) and compact_text(left_name) == compact_text(right_name)
        address_exact = bool(left_address) and compact_text(left_address) == compact_text(
            right_address
        )
        rows.append(
            {
                "source1_entity_id": str(row.source1_entity_id),
                "candidate_entity_id": str(row.candidate_entity_id),
                "blocking_score": float(row.blocking_score),
                "combined_blocking_score": float(
                    getattr(row, "combined_blocking_score", row.blocking_score)
                ),
                "name_blocking_score": float(getattr(row, "name_blocking_score", 0.0)),
                "address_blocking_score": float(getattr(row, "address_blocking_score", 0.0)),
                "name_ratio": _fuzzy_ratio(left_name, right_name),
                "name_token_set_ratio": _token_set_ratio(left_name, right_name),
                "name_token_jaccard": _jaccard(left_name_tokens, right_name_tokens),
                "name_token_containment": _containment(left_name_tokens, right_name_tokens),
                "name_exact": float(name_exact),
                "name_missing_either": float(not left_name or not right_name),
                "address_ratio": _fuzzy_ratio(left_address, right_address),
                "address_token_set_ratio": _token_set_ratio(left_address, right_address),
                "address_token_jaccard": _jaccard(left_address_tokens, right_address_tokens),
                "address_token_containment": _containment(
                    left_address_tokens, right_address_tokens
                ),
                "address_exact": float(address_exact),
                "address_missing_either": float(not left_address or not right_address),
                "address_digit_jaccard": _jaccard(left_digits, right_digits),
                "address_digits_exact": float(bool(left_digits) and left_digits == right_digits),
                "address_digits_conflict": float(
                    bool(left_digits)
                    and bool(right_digits)
                    and left_digits.isdisjoint(right_digits)
                ),
                "country_exact": float(bool(left_country) and left_country == right_country),
                "country_mismatch": float(
                    bool(left_country) and bool(right_country) and left_country != right_country
                ),
                "country_missing_either": float(not left_country or not right_country),
                "candidate_is_source3": float(str(row.candidate_entity_id).startswith("S3-")),
                "name_length_ratio": _length_ratio(left_name, right_name),
                "address_length_ratio": _length_ratio(left_address, right_address),
                "name_and_address_exact": float(name_exact and address_exact),
            }
        )
    return pd.DataFrame(
        rows,
        columns=("source1_entity_id", "candidate_entity_id", *FEATURE_COLUMNS),
    )
