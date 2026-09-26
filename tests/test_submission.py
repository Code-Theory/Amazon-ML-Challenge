from __future__ import annotations

import pytest

from mlchallenge.contracts import ChallengeTestData
from mlchallenge.submission import validate_submission_frames, write_submission_files


def test_submission_round_trip_and_subset_rule(tmp_path, synthetic_frames) -> None:
    frames = synthetic_frames
    test_data = ChallengeTestData(frames["source1"], frames["source2"], frames["source3"])
    predictions = {
        "S1-001": {"S2-101"},
        "S1-002": {"S3-202"},
        "S1-003": set(),
        "S1-004": set(),
    }
    candidates = {
        "S1-001": {"S2-101", "S3-201"},
        "S1-002": {"S2-102", "S3-202"},
        "S1-003": {"S2-104"},
        "S1-004": set(),
    }
    matching_path, candidate_path = write_submission_files(
        test_data, predictions, candidates, tmp_path
    )
    assert matching_path.is_file()
    assert candidate_path.is_file()

    matching = __import__("pandas").read_csv(
        matching_path, sep="\t", dtype="string", keep_default_na=False
    )
    candidate = __import__("pandas").read_csv(
        candidate_path, sep="\t", dtype="string", keep_default_na=False
    )
    assert validate_submission_frames(matching, candidate, test_data) == []

    matching.loc[0, "matched_entity_ids"] = "S2-103"
    issues = validate_submission_frames(matching, candidate, test_data)
    assert any("absent from candidates" in issue for issue in issues)


def test_submission_rejects_unknown_mapping_keys_and_wrong_row_order(
    tmp_path, synthetic_frames
) -> None:
    frames = synthetic_frames
    test_data = ChallengeTestData(frames["source1"], frames["source2"], frames["source3"])
    with pytest.raises(ValueError, match="unknown Source 1"):
        write_submission_files(
            test_data,
            {"S1-unknown": {"S2-101"}},
            {},
            tmp_path,
        )

    matching = __import__("pandas").DataFrame(
        {
            "source1_entity_id": list(reversed(frames["source1"]["entity_id"].tolist())),
            "matched_entity_ids": [""] * len(frames["source1"]),
        }
    )
    candidates = matching.rename(columns={"matched_entity_ids": "candidate_entity_ids"})
    issues = validate_submission_frames(matching, candidates, test_data)
    assert any("row order" in issue for issue in issues)
