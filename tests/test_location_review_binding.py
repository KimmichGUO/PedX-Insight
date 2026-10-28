"""City reviews must remain bound to the analyzed bytes and reviewed interval."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.pipeline import MANIFEST_NAME, atomic_json
from modules.quality import video_metadata


CITY = "geonames:2820256"
SOURCE = "a" * 64


def review_record():
    return {"schema_version": 1, "review_id": "review-ulm-001", "reviewer": "human:reviewer",
            "reviewed_at": "2026-10-27T12:00:00Z", "method": "manual_video_review",
            "city_id": CITY, "source_video_sha256": SOURCE,
            "clip_start_seconds": 0, "clip_end_seconds": 300,
            "evidence": [{"type": "frame_landmark", "description": "Reviewed landmark at 30 seconds"}]}


def metadata(tmp_path, *, review=None, encoded=None, configuration=None, status="complete", raw_updates=None):
    folder = tmp_path / "Ulm1_video"
    config = {"source_sha256": SOURCE, "clip_start_seconds": 0, "clip_end_seconds": 300}
    if configuration:
        config.update(configuration)
    atomic_json(folder / MANIFEST_NAME, {"status": status, "run_id": "analysis-one",
                "finished_at": "2026-10-28T10:00:00Z", "configuration": config})
    evidence = encoded if encoded is not None else json.dumps(review if review is not None else review_record(), indent=2)
    raw = {"city_id": CITY, "searched_city_id": CITY, "verified_city_id": CITY,
           "eligibility_status": "verified", "location_verification_source": evidence}
    raw.update(raw_updates or {})
    original = copy.deepcopy(raw)
    result = video_metadata(folder, "Ulm", "video", raw)
    assert raw == original, "Review history must not be rewritten while checking eligibility"
    assert result["location_verification_source"] == evidence
    return result


@pytest.mark.parametrize("bounds", [(0, 300), (30, 240)])
def test_matching_source_review_covers_current_interval(tmp_path, bounds):
    result = metadata(tmp_path, configuration=dict(zip(("clip_start_seconds", "clip_end_seconds"), bounds)))
    assert result["eligible_for_comparison"] is True
    assert result["eligibility_status"] == "verified"


def test_changed_source_invalidates_review_without_destroying_original_bytes(tmp_path):
    encoded = " \n" + json.dumps(review_record(), indent=3) + "\n "
    result = metadata(tmp_path, encoded=encoded, configuration={"source_sha256": "b" * 64})
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_source_mismatch"
    assert result["source_video_sha256"] == "b" * 64


@pytest.mark.parametrize("bounds", [(0, 1800), (0, 301), (0, 300)])
def test_review_does_not_authorize_an_unreviewed_part_of_a_window(tmp_path, bounds):
    review = review_record()
    if bounds == (0, 300):
        review["clip_start_seconds"] = 30
    result = metadata(tmp_path, review=review,
                      configuration=dict(zip(("clip_start_seconds", "clip_end_seconds"), bounds)))
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_interval_mismatch"


def test_reviewed_city_must_match_current_identity(tmp_path):
    review = review_record()
    review["city_id"] = "geonames:2950159"
    result = metadata(tmp_path, review=review)
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_city_mismatch"


@pytest.mark.parametrize("field", list(review_record()))
def test_missing_required_review_fields_fail_closed(tmp_path, field):
    review = review_record()
    del review[field]
    result = metadata(tmp_path, review=review)
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_invalid"


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("reviewer", "  "), ("reviewed_at", "2026-02-30T12:00:00Z"),
    ("reviewed_at", "2026-10-27T12:00:00"), ("source_video_sha256", "not-a-source-digest"),
    ("clip_end_seconds", float("inf")), ("clip_start_seconds", True),
    ("evidence", []), ("evidence", [""]),
])
def test_malformed_review_fields_fail_closed(tmp_path, field, value):
    review = review_record()
    review[field] = value
    result = metadata(tmp_path, review=review)
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_invalid"


@pytest.mark.parametrize("encoded", ['{invalid', '\ufeff{invalid', '[]', '{"schema_version":1}', 'null', '"manual:review"'])
def test_invalid_json_review_never_becomes_legacy_manual_evidence(tmp_path, encoded):
    result = metadata(tmp_path, encoded=encoded)
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_invalid"


@pytest.mark.parametrize("configuration,expected", [
    ({"source_sha256": None}, "review_source_unverified"),
    ({"clip_start_seconds": None}, "review_interval_unknown"),
    ({"clip_end_seconds": None}, "review_interval_unknown"),
    ({"clip_end_seconds": float("nan")}, "review_interval_unknown"),
])
def test_unknown_actual_source_or_bounds_cannot_inherit_review(tmp_path, configuration, expected):
    result = metadata(tmp_path, configuration=configuration)
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == expected


def test_incomplete_analysis_cannot_use_a_bound_review(tmp_path):
    result = metadata(tmp_path, status="failed")
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "review_analysis_incomplete"


def test_same_source_and_window_remain_reviewed_after_a_code_or_run_change(tmp_path):
    metadata(tmp_path)
    folder = tmp_path / "Ulm1_video"
    manifest = json.loads((folder / MANIFEST_NAME).read_text(encoding="utf-8"))
    manifest.update(run_id="analysis-new-code", git_sha="c" * 40)
    manifest["configuration"]["source_code_sha256"] = "d" * 64
    atomic_json(folder / MANIFEST_NAME, manifest)
    result = video_metadata(folder, "Ulm", "video", {"city_id": CITY, "verified_city_id": CITY,
        "eligibility_status": "verified", "location_verification_source": json.dumps(review_record())})
    assert result["eligible_for_comparison"] is True
    assert result["analysis_run_id"] == "analysis-new-code"


def test_valid_review_does_not_promote_a_row_that_is_still_unverified(tmp_path):
    result = metadata(tmp_path, raw_updates={"eligibility_status": "unverified"})
    assert result["eligible_for_comparison"] is False
    assert result["eligibility_status"] == "unverified"


def test_legacy_plain_text_manual_evidence_remains_compatible(tmp_path):
    result = metadata(tmp_path, encoded="human:synthetic-review",
                      configuration={"source_sha256": None, "clip_end_seconds": None})
    assert result["eligible_for_comparison"] is True
    assert result["eligibility_status"] == "verified"
