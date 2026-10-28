"""Dependency drift is reported explicitly, including damaged metadata and cv2 collisions."""
from importlib import metadata
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import check_environment as environment


def test_drift_is_reported_without_hiding_invalid_metadata(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=2.5.1\nPillow>=12.3.0\nstatsmodels>=0.15.0\n", encoding="utf-8")
    observed = {"numpy": "2.5.3", "Pillow": "10.4.0", "statsmodels": None,
                "opencv-python": "5.0.0.93", "opencv-contrib-python": "5.0.0.93"}
    def version(name):
        if name in observed:
            return observed[name]
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(environment.metadata, "version", version)
    report = environment.inspect_environment(requirements)
    assert len(report["errors"]) == 3
    assert any("Pillow" in error for error in report["errors"])
    assert any("statsmodels" in error for error in report["errors"])
    assert len(report["cv2_providers"]) == 2
    assert report["imported"] == []


def test_clean_profile_accepts_single_provider_and_valid_versions(tmp_path, monkeypatch):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("numpy>=2.5.1\nopencv-python>=5.0.0.93\n", encoding="utf-8")
    def version(name):
        if name == "numpy":
            return "2.5.3"
        if name == "opencv-python":
            return "5.0.0.93"
        raise metadata.PackageNotFoundError(name)
    monkeypatch.setattr(environment.metadata, "version", version)
    assert not environment.inspect_environment(requirements)["errors"]
