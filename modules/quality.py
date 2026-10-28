"""Machine-readable validity; raw history is retained separately from eligibility."""
import json
from pathlib import Path
import unicodedata

import pandas as pd

from modules.pipeline import read_manifest

METADATA_COLUMNS = ["city_id", "searched_city_id", "verified_city_id", "eligibility_status",
    "location_verification_source", "admin1_code", "discovery_run_id", "discovered_at", "sampling_strategy", "search_order",
    "recorded_at", "recording_date_source", "analyzed_at", "analyzed_at_source",
    "analysis_run_id", "analysis_status", "source_video_sha256", "clip_start_seconds", "clip_end_seconds",
    "eligible_for_comparison", "metric_validity"]


def _token(value):
    text = unicodedata.normalize("NFKD", str(_clean(value)))
    text = "".join(char for char in text if not unicodedata.combining(char)).casefold()
    return "-".join("".join(char if char.isalnum() else " " for char in text).split()) or "unknown"


def stable_city_id(city, country=None, admin=None, geonameid=None):
    geonameid = _clean(geonameid)
    if geonameid and str(geonameid).split(".")[0].isdigit() and float(geonameid) > 0:
        return "geonames:" + str(int(float(geonameid)))
    return f"place:{_token(country)}:{_token(admin)}:{_token(city)}"


def _clean(value):
    if value is None or (not isinstance(value, (dict, list)) and pd.isna(value)):
        return ""
    return value


def _status(status, reason):
    return {"status": status, "reason": reason}


def metric_validity(folder):
    invalid = {"weather": "Weather classifier failed JAAD validation.",
        "main_weather": "Weather classifier failed JAAD validation.",
        "daytime": "Day/night classifier has not established cross-city validity.",
        "gender": "Inferred gender is at the majority-class baseline on JAAD.",
        "age": "Inferred age has not established useful accuracy on JAAD.",
        "average_age": "Inferred age has not established useful accuracy on JAAD."}
    result = {key: _status("invalid", reason) for key, reason in invalid.items()}
    for key in ("crossing_speed", "crossing_time"):
        result[key] = _status("reference_only", "Imported city-level constant; not measured from this video.")
    experimental = ["crossed", "total_pedestrians", "total_crossed_pedestrians", "risky_crossing", "risky_crossing_rate",
        "risky_crossing_ratio", "run_red_light", "run_red_light_rate", "run_red_light_ratio",
        "phone_using", "phone_usage_rate", "phone_usage_ratio", "walking_speed_mps", "crossing_speed_mps",
        "measured_avg_walking_speed_mps", "measured_walking_speed", "measured_crossing_speed_mps",
        "measured_crossing_speed", "decision_delay_s", "vehicle_median_speed_mps", "vehicle_p85_speed_mps",
        "pet_severe_conflicts", "pet_moderate_conflicts", "pet_queued_interactions", "pet_min_s",
        "mean_headway_s", "platoon_frac", "vehicle_flow_per_min", "anticipatory_start_frac",
        "mean_red_exposure_s", "hesitation_rate", "aborted_start_rate", "evasive_event_count",
        "n_social_groups", "grouped_pedestrians", "look_before_cross_frac", "looked_both_ways_frac",
        "median_cadence_hz", "cadence_n", "waiting_time", "avg_road_width", "sidewalk_prob", "traffic_light_prob"]
    for key in experimental:
        result[key] = _status("experimental", "Global transfer/calibration is unverified; inspect coverage and validation evidence.")
    segmented = (Path(folder) / "[E10]scene_segmentation.csv").is_file()
    for key in ("crosswalk_prob", "crosswalk", "crosswalk_usage_rate", "crosswalk_usage_ratio", "crosswalk_use_or_not"):
        result[key] = _status("experimental" if segmented else "invalid_legacy",
            "Presence validated on JAAD; global transfer, box geometry and usage remain unverified."
            if segmented else "Legacy crosswalk detector was at chance; reanalysis required.")
    aliases = {"avg_age": "age", "avg_pedestrian_age": "age", "age_gender": "age",
        "crosswalk_usage": "crosswalk_usage_rate", "phone_usage": "phone_usage_rate",
        "look_before_cross": "look_before_cross_frac", "pet": "pet_severe_conflicts",
        "vehicle_speed": "vehicle_median_speed_mps", "groups": "n_social_groups",
        "social_groups": "n_social_groups", "tracking": "total_pedestrians",
        "pedestrian_density": "total_pedestrians", "road_width": "avg_road_width",
        "micro_events": "hesitation_rate", "clothing": "total_pedestrians",
        "belongings": "total_pedestrians", "vehicle_type": "total_pedestrians"}
    for alias, original in aliases.items():
        result[alias] = dict(result[original])
    return result


def analysis_time_record(folder):
    """Timing belongs to the current run, not just the same folder name."""
    path = Path(folder) / "[A3]analysis_time.csv"
    if not path.exists():
        return None
    try:
        timing = pd.read_csv(path)
        if timing.empty:
            return None
        row = timing.iloc[0].to_dict()
        manifest = read_manifest(folder)
        if manifest and row.get("analysis_run_id") != manifest.get("run_id"):
            return None
        return row
    except (ValueError, OSError):
        return None


def video_metadata(folder, city, link, city_info=None):
    raw = {} if city_info is None else {key: _clean(value) for key, value in dict(city_info).items()}
    provenance = raw.get("video_provenance")
    if isinstance(provenance, str) and provenance:
        try:
            raw.update(json.loads(provenance).get(link, {}))
        except (ValueError, AttributeError):
            pass
    manifest = read_manifest(folder)
    state = manifest.get("status", "unknown") if manifest else "legacy_unverified"
    analyzed_at, analyzed_source = (manifest.get("finished_at"), "run_manifest") if manifest else (None, "unknown")
    if not analyzed_at:
        path = Path(folder) / "[A0]provenance.csv"
        if path.exists():
            try:
                prov = pd.read_csv(path)
                values = prov.get("finished_utc", pd.Series(dtype="str")).dropna()
                if len(values):
                    analyzed_at, analyzed_source = str(values.max()), "mode_provenance"
            except (ValueError, OSError):
                pass
    identity = raw.get("city_id") or stable_city_id(city, raw.get("country_code") or raw.get("iso3") or raw.get("country"),
        raw.get("admin1_code") or raw.get("state"), raw.get("geonameid"))
    verified = raw.get("verified_city_id") or ""
    eligibility = raw.get("eligibility_status") or "unverified"
    evidence = raw.get("location_verification_source") or ""
    out = {key: raw.get(key) or None for key in METADATA_COLUMNS}
    out.update(city_id=identity, searched_city_id=raw.get("searched_city_id") or identity,
        verified_city_id=verified or None, eligibility_status=eligibility,
        location_verification_source=evidence or None,
        recorded_at=raw.get("recorded_at") or raw.get("recording_date") or None,
        recording_date_source=raw.get("recording_date_source") or "unknown",
        analyzed_at=analyzed_at, analyzed_at_source=analyzed_source,
        analysis_run_id=manifest.get("run_id") if manifest else None, analysis_status=state,
        eligible_for_comparison=bool(state == "complete" and eligibility == "verified" and verified == identity and evidence),
        metric_validity=json.dumps(metric_validity(folder), ensure_ascii=False, sort_keys=True))
    configuration = manifest.get("configuration", {}) if manifest else {}
    out["source_video_sha256"] = configuration.get("source_sha256")
    timing = analysis_time_record(folder) or {}
    for key in ("clip_start_seconds", "clip_end_seconds"):
        value = timing.get(key, configuration.get(key))
        if value is not None and pd.notna(value):
            out[key] = value
    return out


def apply_metric_policy(frame, *, include_experimental=False):
    """Mask ineligible observations before statistical grouping; unknown is never zero."""
    out = frame.copy()
    allowed = {"validated"} | ({"experimental", "unknown"} if include_experimental else set())
    preserve = set(METADATA_COLUMNS) | {"city", "link", "city_link", "video_name", "track_id", "state", "country", "iso3", "continent",
        "lat", "lon", "gmp", "population_city", "population_country", "traffic_mortality", "literacy_rate",
        "avg_height", "med_age", "gini", "crossing_time", "crossing_speed", "pipeline_version", "cluster",
        "risky_crossing_coverage", "measured_crossing_speed_n", "code_version", "code_version_mixed",
        "duration_seconds", "duration_source", "total_frames", "data_collected_date"}
    measured = [column for column in out.columns if column not in preserve]
    for column in measured:
        out[column] = out[column].astype(object)
    for index, row in out.iterrows():
        eligible = str(row.get("eligible_for_comparison", "")).strip().lower() in {"true", "1", "1.0"}
        encoded = row.get("metric_validity")
        try:
            policies = json.loads(encoded) if isinstance(encoded, str) else (encoded if isinstance(encoded, dict) else {})
        except ValueError:
            policies = {}
        if not isinstance(policies, dict):
            policies = {}
        for column in measured:
            rule = policies.get(column, {})
            status = rule.get("status", "unknown") if isinstance(rule, dict) else "unknown"
            if not eligible or status not in allowed:
                out.at[index, column] = pd.NA
    return out
