"""Synthetic regression tests for quality gates, empty contracts, and observation gaps."""
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.summary import video_info, pede_info
from modules.waiting_time_pede import waiting_time_pede as waiting


@pytest.mark.parametrize("flags", [[False, False], [None, None]])
def test_rollup_never_falls_back_to_unreliable_or_unknown_rows(tmp_path, flags):
    pd.DataFrame({"median_speed_mps": [40, 60], "reliable": flags}).to_csv(tmp_path / "[V8]vehicle_speed.csv", index=False)
    pd.DataFrame({"looked_left": [True, True], "looked_right": [True, True],
                  "cadence_hz": [2, 2.2], "reliable": flags}).to_csv(tmp_path / "[P12]pose_behavior.csv", index=False)
    result = video_info._insight_rollup(str(tmp_path))
    assert not {"vehicle_median_speed_mps", "vehicle_p85_speed_mps", "look_before_cross_frac",
                "looked_both_ways_frac", "median_cadence_hz", "cadence_n"}.intersection(result)


def test_rollup_uses_only_explicitly_reliable_rows(tmp_path):
    pd.DataFrame({"median_speed_mps": [3, 60], "reliable": [True, False]}).to_csv(tmp_path / "[V8]vehicle_speed.csv", index=False)
    pd.DataFrame({"looked_left": [True, False], "looked_right": [False, True],
                  "cadence_hz": [2, 2.4], "reliable": [True, False]}).to_csv(tmp_path / "[P12]pose_behavior.csv", index=False)
    result = video_info._insight_rollup(str(tmp_path))
    assert result["vehicle_median_speed_mps"] == 3
    assert result["looked_both_ways_frac"] == 0
    assert result["median_cadence_hz"] == 2 and result["cadence_n"] == 1


def test_missing_reliability_field_cannot_publish_legacy_estimates(tmp_path):
    pd.DataFrame({"median_speed_mps": [60]}).to_csv(tmp_path / "[V8]vehicle_speed.csv", index=False)
    pd.DataFrame({"looked_left": [True], "looked_right": [True], "cadence_hz": [2]}).to_csv(tmp_path / "[P12]pose_behavior.csv", index=False)
    assert video_info._insight_rollup(str(tmp_path)) == {}


@pytest.mark.parametrize("personal,environment", [(None, None), ("empty", "header"), ("header", "empty")])
def test_empty_pedestrian_summary_keeps_schema(tmp_path, monkeypatch, personal, environment):
    monkeypatch.setattr(pede_info, "RESULTS_ROOT", str(tmp_path / "results"))
    paths = [tmp_path / "personal.csv", tmp_path / "environment.csv"]
    for path, kind in zip(paths, (personal, environment)):
        if kind:
            path.write_text("track_id,extra_metric\n" if kind == "header" else "\n", encoding="utf-8")
    output = tmp_path / "nested" / "summary.csv"
    pede_info.summary_all_info("City1_video.mp4", str(paths[0]), str(paths[1]), str(output))
    result = pd.read_csv(output)
    assert result.empty and set(pede_info.OUTPUT_COLUMNS).issubset(result.columns)
    if "header" in (personal, environment):
        assert "extra_metric" in result.columns


def fixture(tmp_path, monkeypatch, timestamps, **options):
    monkeypatch.setattr(waiting, "RESULTS_ROOT", str(tmp_path / "results"))
    source = tmp_path / "tracks.csv"
    pd.DataFrame([{"track_id": 1, "frame_id": round(t * 30) + 1, "timestamp": t,
                   "x1": 100, "x2": 160, "y1": 100, "y2": 270} for t in timestamps]).to_csv(source, index=False)
    output = tmp_path / "waiting.csv"
    waiting.run_waiting_time_analysis("City1_video.mp4", csv_path=str(source), output_csv=str(output),
                                       mapping_csv=str(tmp_path / "no_mapping.csv"), **options)
    return pd.read_csv(output)


def test_unobserved_gaps_are_not_waiting(tmp_path, monkeypatch):
    assert fixture(tmp_path, monkeypatch, [0, 10, 20]).empty


def test_gap_breaks_contiguous_wait_and_does_not_join_short_spans(tmp_path, monkeypatch):
    assert fixture(tmp_path, monkeypatch, [0, .5, 10, 10.5]).empty


def test_observed_wait_duration_excludes_occlusion_gap(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch, [0, 1, 2, 20, 21, 22]).iloc[0]
    assert out.waiting_time == 4 and out.observed_duration_s == 4 and out.unobserved_gap_count == 1


def test_wait_threshold_and_sparse_cadence_override(tmp_path, monkeypatch):
    out = fixture(tmp_path, monkeypatch, [0, .25, .5], min_wait_seconds=.4).iloc[0]
    assert out.waiting_time == .5
    out = fixture(tmp_path, monkeypatch, [0, 2, 4], max_gap_seconds=3).iloc[0]
    assert out.waiting_time == 4
