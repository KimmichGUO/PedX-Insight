"""Build a checksummed dataset in a fresh directory without publishing it."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import shutil
import tempfile

import pandas as pd

from modules.completeness import publication_issues
from modules.paths import RESULTS_ROOT
from modules.pipeline import atomic_json, file_hash, output_record, read_manifest
from modules.provenance import git_state, utc_now
from modules.quality import metric_validity

CORE_FILES = ("all_video_info.csv", "all_pedestrian_info.csv", "all_time_info.csv", "all_video_locations.csv")
ANALYTICS_FILES = ("accident_road_condition_stats.csv", "age_stats.csv", "carried_items_stats.csv",
    "clothing_stats.csv", "continent_stats.csv", "crossing_stats.csv", "crosswalk_coeff.csv", "gender_stats.csv",
    "phone_stats.csv", "road_corr.csv", "time_stats.csv", "vehicle_stats.csv", "weather_daytime_stats.csv")


def portable_basename(value):
    """Result folders can have been produced on a different operating system."""
    return str(value).replace("\\", "/").rsplit("/", 1)[-1]


def public_model_id(value):
    """Keep hub identifiers, but omit machine-specific paths to local weights."""
    text = str(value)
    if (Path(text).is_absolute() or PureWindowsPath(text).anchor or
            Path(text).suffix.lower() in {".pt", ".pth", ".onnx", ".bin", ".safetensors", ".pb"}):
        return portable_basename(text)
    return text


def validate_snapshot(directory, *, require_completed=True):
    frames = {name: pd.read_csv(Path(directory) / name, encoding="utf-8", dtype={"link": "str"}) for name in CORE_FILES}
    videos = frames[CORE_FILES[0]]
    if videos.empty:
        raise ValueError("No completed videos available; retaining the previous published release")
    if videos["link"].isna().any() or videos["link"].duplicated().any():
        raise ValueError("Video identities must be nonempty and unique")
    if require_completed and (videos["analysis_status"] != "complete").any():
        raise ValueError("A release cannot mix completed runs with legacy analysis")
    links = set(videos["link"])
    for name, frame in frames.items():
        if len(frame) and (frame["link"].isna().any() or not set(frame["link"]).issubset(links)):
            raise ValueError(f"{name} contains an unknown video identity")
    pedestrians = frames[CORE_FILES[1]]
    if len(pedestrians):
        if pedestrians["track_id"].isna().any() or pedestrians.duplicated(["link", "track_id"]).any():
            raise ValueError("Pedestrian identities must be unique within a video")
        reference = videos.set_index("link")["analysis_run_id"]
        if not pedestrians["analysis_run_id"].fillna("").eq(pedestrians["link"].map(reference).fillna("")).all():
            raise ValueError("Video and pedestrian snapshots belong to different runs")
    return frames


def build_release(output, *, results_root=None, mapping_path="mapping.csv", skip_stats=False,
                  include_experimental=False, allow_legacy=False):
    with tempfile.TemporaryDirectory(prefix="pedx_release_inputs_") as scratch:
        snapshot = Path(scratch) / "mapping.csv"
        shutil.copyfile(Path(mapping_path).resolve(), snapshot)
        return _build_release_snapshot(output, results_root=results_root, mapping_path=snapshot,
            skip_stats=skip_stats, include_experimental=include_experimental, allow_legacy=allow_legacy)


def _build_release_snapshot(output, *, results_root=None, mapping_path="mapping.csv", skip_stats=False,
                            include_experimental=False, allow_legacy=False):
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Release destination must be fresh: {output}")
    output.mkdir(parents=True, exist_ok=True)
    results, mapping = Path(results_root or RESULTS_ROOT).resolve(), Path(mapping_path).resolve()
    try:
        import get_all_video_info
        import get_all_pede_info
        import get_all_video_locations
        for module in (get_all_video_info, get_all_pede_info, get_all_video_locations):
            module.main(results_root=results, mapping_path=mapping, output_root=output)
        videos = pd.read_csv(output / CORE_FILES[0], dtype={"link": "str"})
        if not allow_legacy:
            videos = videos[videos["analysis_status"] == "complete"]
        videos.to_csv(output / CORE_FILES[0], index=False, encoding="utf-8")
        links = set(videos["link"])
        for name in CORE_FILES[1:]:
            frame = pd.read_csv(output / name, dtype={"link": "str"})
            frame[frame["link"].isin(links)].to_csv(output / name, index=False, encoding="utf-8")
        frames = validate_snapshot(output, require_completed=not allow_legacy)
        optional = []
        env = dict(os.environ, PEDX_SUMMARY_ROOT=str(output),
                   PEDX_INCLUDE_EXPERIMENTAL="1" if include_experimental else "0",
                   PYTHONDONTWRITEBYTECODE="1", MPLBACKEND="Agg")
        root = Path(__file__).resolve().parents[1]
        if not skip_stats:
            subprocess.run([sys.executable, "-B", "statistics_with_pdf_save.py"], check=True, env=env, cwd=root)
            result = subprocess.run([sys.executable, "-B", "statistics_inferential.py"], env=env, cwd=root)
            if result.returncode:
                for name in ("inferential_associations.csv", "inferential_variance_partition.csv", "inferential_forest.pdf"):
                    (output / name).unlink(missing_ok=True)
                    optional.append({"path": name, "status": "unavailable", "reason": "inferential fit failed for this release"})
        else:
            optional.extend({"path": name, "status": "omitted", "reason": "statistics explicitly skipped"} for name in ANALYTICS_FILES)
        # Reject snapshots changed by an analysis that started during aggregation.
        source_runs = []
        for row in frames[CORE_FILES[0]].to_dict("records"):
            folder = results / row["video_name"]
            if row["analysis_status"] == "complete":
                issues = publication_issues(folder, allow_legacy=False)
                current = read_manifest(folder) or {}
                if issues or current.get("run_id") != row["analysis_run_id"]:
                    raise ValueError(f"Analysis changed during snapshot: {row['video_name']}: {issues}")
                configuration = dict(current.get("configuration", {}))
                if configuration.get("weights_yolo"):
                    configuration["weights_yolo"] = portable_basename(configuration["weights_yolo"])
                models = {}
                assets_path = folder / "[A0]model_assets.json"
                if configuration.get("model_assets_sha256") and not assets_path.is_file():
                    raise ValueError(f"Model provenance missing during snapshot: {row['video_name']}")
                if assets_path.exists():
                    assets_bytes = assets_path.read_bytes()
                    if hashlib.sha256(assets_bytes).hexdigest() != configuration.get("model_assets_sha256"):
                        raise ValueError(f"Model provenance changed during snapshot: {row['video_name']}")
                    models = json.loads(assets_bytes)
                    for asset in models.values():
                        asset["model_id"] = public_model_id(asset.get("model_id", ""))
                        asset["files"] = [{"name": portable_basename(name), "sha256": digest}
                                          for name, digest in asset.get("files", {}).items()]
                source_runs.append({"link": row["link"], "video_name": row["video_name"],
                    "analysis_run_id": current["run_id"], "git_sha": current.get("git_sha"),
                    "started_at": current.get("started_at"), "finished_at": current.get("finished_at"),
                    "configuration": configuration, "model_assets": models})
        atomic_json(output / "analysis_runs.json", {"schema_version": 1, "runs": source_runs})
        atomic_json(output / "metrics_registry.json", {"schema_version": 1, "scope": "global_comparison",
            "benchmark": "validation/results/JAAD_REPORT.md", "default_statuses": metric_validity(output),
            "default_allowed": ["validated"], "experimental_opt_in": ["experimental", "unknown"],
            "invalid_never_compared": ["invalid", "invalid_legacy"]})
        population = {"eligible_only": True, "allowed_metric_statuses": ["validated"] +
            (["experimental", "unknown"] if include_experimental else []), "filters": {},
            "metric_specific_denominators": True, "invalid_excluded": True}
        atomic_json(output / "statistics_status.json", {"statistics_scope": "eligible", "population": population,
            "optional_artifacts": optional})
        artifacts = []
        for path in sorted(output.iterdir()):
            if path.is_file() and not path.name.startswith("."):
                record = output_record(path) if path.suffix == ".csv" else {"sha256": file_hash(path), "rows": None, "columns": []}
                artifacts.append({"path": path.name, **record})
        sha, dirty = git_state()
        identity = hashlib.sha256(json.dumps({"artifacts": artifacts, "git_sha": sha,
            "mapping_sha256": file_hash(mapping)}, sort_keys=True).encode()).hexdigest()[:24]
        manifest = {"schema_version": 1, "dataset_release_id": identity, "created_at": utc_now(),
            "producer": {"git_sha": sha, "git_dirty": dirty, "mapping_sha256": file_hash(mapping)},
            "files": artifacts, "optional_artifacts": optional, "statistics_scope": "eligible",
            "statistics_population": population, "n_videos": len(videos),
            "n_pedestrians": len(frames[CORE_FILES[1]]),
            "n_eligible_videos": int(videos["eligible_for_comparison"].astype(str).str.lower().eq("true").sum()),
            "legacy_audit_only": allow_legacy}
        atomic_json(output / "release_manifest.json", manifest)
        return manifest
    except BaseException as error:
        atomic_json(output / ".failed.json", {"created_at": utc_now(), "error": str(error)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--results-root", default=RESULTS_ROOT)
    parser.add_argument("--mapping", default="mapping.csv")
    parser.add_argument("--skip-stats", action="store_true")
    parser.add_argument("--include-experimental", action="store_true")
    parser.add_argument("--allow-legacy-audit", action="store_true", help="Build an audit artifact; production publisher rejects it")
    args = parser.parse_args()
    release = build_release(args.output, results_root=args.results_root, mapping_path=args.mapping,
        skip_stats=args.skip_stats, include_experimental=args.include_experimental, allow_legacy=args.allow_legacy_audit)
    print(f"Release {release['dataset_release_id']} ready in {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
