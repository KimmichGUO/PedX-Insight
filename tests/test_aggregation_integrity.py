"""Regression tests for the aggregation / statistics path.

Plain asserts; runs under pytest or directly:
    python tests/test_aggregation_integrity.py

Each test pins a bug that reached published data:
  * [C1] risky crossing scored a SILENT detector as maximum risk (2026-09-07 construct fix)
  * a folder from a run that died mid-pipeline entered the dataset (completeness gate)
  * nullable 0/1 flags were written as '1.0', which the Visualizer importer read as false
  * 'NewYork' never joined mapping.csv 'New York'; 0 was used as a missing-value code
  * str + NA under pandas 3 produced NA cluster ids that dropped rows from clustered stats
  * provenance must be recorded and summarised without ever raising
"""

import os
import shutil
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import pandas as pd


def _tmpdir():
    return tempfile.mkdtemp(prefix="pedx_test_")


# --------------------------------------------------------------------------- [C1] risk

def _write_risk_fixture(d, light_rows=True):
    # Three crossers: #1 crosses during a GREEN vehicle light (against traffic),
    # #2 during RED (with the pedestrian phase), #3 while the detector saw no light.
    pd.DataFrame({"track_id": [1, 2, 3], "crossed": [True, True, True],
                  "started_frame": [10, 30, 50], "ended_frame": [14, 34, 54]}
                 ).to_csv(os.path.join(d, "[C3]crossing_judge.csv"), index=False)
    if light_rows:
        frames = list(range(0, 60))
        color = ["green" if 10 <= f <= 14 else "red" if 30 <= f <= 34 else "None" for f in frames]
        pd.DataFrame({"frame_id": frames, "main_light_color": color}
                     ).to_csv(os.path.join(d, "[E2]traffic_light.csv"), index=False)
    pd.DataFrame({"frame_id": list(range(0, 60)), "crosswalk_detected": ["no"] * 60}
                 ).to_csv(os.path.join(d, "[E7]crosswalk_detection.csv"), index=False)


def _run_risk(d):
    from modules.risky_crossing.risky_crossing import detect_crossing_risk
    out = os.path.join(d, "[C1]risky_crossing.csv")
    detect_crossing_risk(
        video_path=os.path.join(d, "x.mp4"),
        traffic_light_csv=os.path.join(d, "[E2]traffic_light.csv"),
        crosswalk_csv=os.path.join(d, "[E7]crosswalk_detection.csv"),
        traffic_sign_csv=os.path.join(d, "[E3]traffic_sign.csv"),
        crossing_judge_csv=os.path.join(d, "[C3]crossing_judge.csv"),
        output_csv_path=out,
        tracked_csv=os.path.join(d, "[B1]tracked_pedestrians.csv"),
    )
    return pd.read_csv(out).set_index("track_id")


def test_risk_is_evidence_positive():
    d = _tmpdir()
    old = os.environ.get("PEDX_RESULTS_ROOT")
    try:
        os.environ["PEDX_RESULTS_ROOT"] = d  # detect_crossing_risk mkdirs its default dir
        _write_risk_fixture(d)
        r = _run_risk(d)
        assert r.loc[1, "risk"] == "risky" and r.loc[1, "risky_ratio"] == 1.0
        assert r.loc[2, "risk"] == "not risky" and r.loc[2, "risky_ratio"] == 0.0
        # No observed signal: undetermined (NA), never 'not risky' and never 'risky'.
        assert pd.isna(r.loc[3, "risk"]) and pd.isna(r.loc[3, "risky_ratio"])
        assert r.loc[3, "signal_coverage"] == 0.0
        # The old construct survives under its honest name: crosser 3 saw neither
        # a light nor a crosswalk in any frame.
        assert r.loc[3, "unsignalled_ratio"] == 1.0
    finally:
        _restore_env(old)
        shutil.rmtree(d, ignore_errors=True)


def test_missing_light_detector_is_not_risk():
    """A detector that never ran must yield NA risk, not 100% risky (the old bug)."""
    d = _tmpdir()
    old = os.environ.get("PEDX_RESULTS_ROOT")
    try:
        os.environ["PEDX_RESULTS_ROOT"] = d
        _write_risk_fixture(d, light_rows=False)
        r = _run_risk(d)
        assert r["risk"].isna().all(), r
        assert (r["env_available"] == False).all()  # noqa: E712
    finally:
        _restore_env(old)
        shutil.rmtree(d, ignore_errors=True)


def _restore_env(old):
    if old is None:
        os.environ.pop("PEDX_RESULTS_ROOT", None)
    else:
        os.environ["PEDX_RESULTS_ROOT"] = old


# ----------------------------------------------------------------- completeness gate

def test_completeness_gate():
    from pathlib import Path
    from modules.completeness import REQUIRED_OUTPUTS, missing_outputs
    d = Path(_tmpdir())
    try:
        assert missing_outputs(d) == REQUIRED_OUTPUTS
        for name in REQUIRED_OUTPUTS[:-1]:
            (d / name).write_text("x\n")
        assert missing_outputs(d) == [REQUIRED_OUTPUTS[-1]]
        (d / REQUIRED_OUTPUTS[-1]).write_text("x\n")
        assert missing_outputs(d) == []
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------- end-to-end pedestrian aggregation (flags)

def _build_results_tree(root, complete=True):
    from modules.completeness import REQUIRED_OUTPUTS
    folder = os.path.join(root, "analysis_results", "NewYork1_abc_DEF")  # id contains '_'
    os.makedirs(folder)
    if complete:
        for name in REQUIRED_OUTPUTS:
            open(os.path.join(folder, name), "w").write("x\n")
    # risky_crossing: one risky, one safe, one undetermined (NA) -> nullable column
    pd.DataFrame({"track_id": [1, 2, 3], "crossed": [True, True, True],
                  "risky_crossing": [1, 0, None], "run_red_light": [1, 0, 0],
                  "crosswalk_use_or_not": [0, 0, 1], "phone_using": [0, 1, None],
                  "gender": ["male", "female", None], "age": [30, 41, None]}
                 ).to_csv(os.path.join(folder, "[A2]pedestrian_info.csv"), index=False)
    pd.DataFrame({"id": [1, 2], "city": ["New York", "Ulm"], "state": ["NY", None],
                  "country": ["United States", "Germany"], "iso3": ["USA", None],
                  "continent": ["North America", None], "lat": [40.7, 48.4], "lon": [-74.0, 9.99],
                  "gmp": [0, None], "population_city": [8e6, None],
                  "population_country": [3.3e8, None], "traffic_mortality": [12.4, None],
                  "literacy_rate": [86.0, None], "avg_height": [170.1, None],
                  "med_age": [38.5, None], "gini": [41.3, None],
                  "videos": ["[abc_DEF]", "[zzz]"]}
                 ).to_csv(os.path.join(root, "mapping.csv"), index=False)


def test_pede_aggregation_flags_and_join():
    import get_all_pede_info
    root = _tmpdir()
    cwd = os.getcwd()
    try:
        _build_results_tree(root)
        os.chdir(root)
        get_all_pede_info.main()
        out_path = os.path.join(root, "summary_data", "all_pedestrian_info.csv")
        # The Visualizer importer matches the literal token '1'; '1.0' read as false.
        df = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        assert set(df["risky_crossing"]) == {"1", "0", ""}, df["risky_crossing"].tolist()
        assert set(df["phone_using"]) == {"0", "1", ""}
        # Crawler slug 'NewYork1' joins mapping 'New York' and adopts the canonical name;
        # the YouTube id keeps its underscore (first-underscore split).
        assert set(df["city"]) == {"New York"} and set(df["link"]) == {"abc_DEF"}
        assert set(df["iso3"]) == {"USA"}
        # gmp 0 is a missing-value code, not a value.
        assert set(df["gmp"]) == {""}
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)


def test_pede_aggregation_skips_incomplete_folder():
    import get_all_pede_info
    root = _tmpdir()
    cwd = os.getcwd()
    try:
        _build_results_tree(root, complete=False)
        os.chdir(root)
        get_all_pede_info.main()
        out_path = os.path.join(root, "summary_data", "all_pedestrian_info.csv")
        assert not os.path.exists(out_path) or pd.read_csv(out_path).empty
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)


# ------------------------------------------------------------------- mapping utils

def test_clean_mapping_backfills_only_unanimous_country_fields():
    from modules.mapping_utils import clean_mapping, find_city_in_mapping
    m = pd.DataFrame({
        "city": ["Berlin", "Hamburg", "Ulm", "London", "London"],
        "country": ["Germany", "Germany", "Germany", "United Kingdom", "Canada"],
        "iso3": ["DEU", "DEU", None, "GBR", "CAN"],
        "continent": ["Europe", "Europe", None, "Europe", "North America"],
        "literacy_rate": [99.0, 0.0, None, 99.0, 99.0],   # Hamburg's 0 = missing
        "gini": [32.4, 31.9, None, 35.1, 33.3],            # German rows disagree
        "videos": ["[a]", "[b]", "[c]", "[uk1]", "[ca1]"],
    })
    c = clean_mapping(m)
    ulm = c[c["city"] == "Ulm"].iloc[0]
    assert ulm["iso3"] == "DEU" and ulm["continent"] == "Europe"
    assert ulm["literacy_rate"] == 99.0          # unanimous once 0 is treated as missing
    assert pd.isna(ulm["gini"])                  # never invented when rows disagree
    # Hamburg's 0 became missing and was then backfilled from the unanimous German value.
    assert c[c["city"] == "Hamburg"].iloc[0]["literacy_rate"] == 99.0
    # Same-name cities are disambiguated by link.
    assert find_city_in_mapping("London", "ca1", c)["country"] == "Canada"
    assert find_city_in_mapping("london", "uk1", c)["country"] == "United Kingdom"
    assert find_city_in_mapping("Atlantis", "x", c) is None


# -------------------------------------------------------------------- stats utils

def test_rate_with_ci_support_floor_and_na():
    from modules.stats_utils import rate_with_ci
    r = rate_with_ci([1, 0, None, 1], ["a", "a", "b", "b"])
    assert r["estimate"] is None and r["n"] == 3   # under MIN_N: blank, n still reported
    vals = [1] * 30 + [0] * 30 + [None] * 40
    clusters = [f"c{i % 6}" for i in range(100)]
    r = rate_with_ci(vals, clusters)
    assert r["n"] == 60 and r["n_clusters"] == 6
    assert abs(r["estimate"] - 0.5) < 1e-12        # NA excluded from the denominator
    assert r["ci_low"] <= r["estimate"] <= r["ci_high"]


def test_cluster_bootstrap_is_wider_than_iid_for_clustered_data():
    """Perfectly clustered outcomes: an i.i.d. CI would be tiny, the cluster CI is not."""
    from modules.stats_utils import rate_with_ci
    vals, clusters = [], []
    for i in range(8):
        vals += [1 if i % 2 else 0] * 50
        clusters += [f"city{i}"] * 50
    r = rate_with_ci(vals, clusters)
    iid_half_width = 1.96 * np.sqrt(0.25 / len(vals))
    assert (r["ci_high"] - r["ci_low"]) / 2 > 3 * iid_half_width


def test_rate_with_ci_is_reproducible():
    from modules.stats_utils import rate_with_ci
    vals = list(np.random.default_rng(1).integers(0, 2, 200))
    clusters = [f"c{i % 9}" for i in range(200)]
    assert rate_with_ci(vals, clusters) == rate_with_ci(vals, clusters)


def test_bh_fdr_matches_statsmodels_and_keeps_na():
    from modules.stats_utils import bh_fdr
    q = bh_fdr([0.01, None, 0.04, 0.03])
    assert pd.isna(q.iloc[1])
    assert np.allclose(q.dropna().to_numpy(), [0.03, 0.04, 0.04])


def test_city_level_counts_each_city_once():
    from modules.stats_utils import city_level
    v = pd.DataFrame({"city": ["A", "A", "A", "B"], "x": [1.0, 2.0, 3.0, 10.0],
                      "gini": [30, 30, 30, 40]})
    c = city_level(v, ["x"], constant_cols=["gini"]).set_index("city")
    assert c.loc["A", "x"] == 2.0 and c.loc["A", "n_videos"] == 3 and len(c) == 2


def test_stats_cluster_id_survives_missing_country():
    """str + NA is NA under pandas 3; the cluster id must not drop such rows."""
    from modules.stats_utils import city_cluster_id
    df = pd.DataFrame({"city": ["Ulm", "Paris"], "country": [None, "France"]})
    out = city_cluster_id(df)
    assert out["cluster"].notna().all() and out["cluster"].nunique() == 2


# ---------------------------------------------------------------------- provenance

def test_provenance_records_and_summarises():
    d = _tmpdir()
    old = os.environ.get("PEDX_RESULTS_ROOT")
    try:
        os.makedirs(os.path.join(d, "vid_X"))
        import modules.provenance as prov
        orig_root = prov.RESULTS_ROOT
        prov.RESULTS_ROOT = d
        try:
            prov.record_mode_run("some/where/vid_X.mp4", "weather", prov.utc_now())
            prov.record_mode_run("some/where/vid_X.mp4", "light", prov.utc_now())
            prov.record_mode_run("some/where/missing.mp4", "light", prov.utc_now())  # no dir: no-op
        finally:
            prov.RESULTS_ROOT = orig_root
        df = pd.read_csv(os.path.join(d, "vid_X", "[A0]provenance.csv"))
        assert list(df["mode"]) == ["weather", "light"]
        assert list(df.columns) == prov.PROVENANCE_COLUMNS
        sha, mixed = prov.summarize(os.path.join(d, "vid_X"))
        assert mixed is False
        assert sha is None or len(sha.split("+")[0]) == 40
        assert prov.summarize(os.path.join(d, "nope")) == (None, None)
        assert not os.path.exists(os.path.join(d, "missing"))
    finally:
        _restore_env(old)
        shutil.rmtree(d, ignore_errors=True)


def test_inferential_runs_on_synthetic_data():
    """statistics_inferential must produce both tables on a small clustered dataset."""
    import statistics_inferential as si
    rng = np.random.default_rng(0)
    rows, vids = [], []
    for c in range(12):
        city_effect = rng.normal(0, 0.6)
        for v in range(2 if c < 4 else 1):
            link = f"L{c}_{v}"
            vids.append({"link": link, "pipeline_version": "dense_v2",
                         "traffic_light_prob": rng.uniform(0, 1),
                         "risky_crossing_coverage": rng.uniform(0.3, 1)})
            for i in range(40):
                p = 1 / (1 + np.exp(-(-1.5 + city_effect)))
                rows.append({"city": f"C{c}", "country": "X", "link": link, "track_id": i,
                             "run_red_light": int(rng.random() < p),
                             "risky_crossing": int(rng.random() < p) if i % 2 else None,
                             "crossing_speed_mps": 1.35 + 0.1 * city_effect + rng.normal(0, 0.3),
                             "gender": rng.choice(["male", "female"]), "age": rng.integers(18, 70),
                             "phone_using": int(rng.random() < 0.1),
                             "nearby_count_beginning": rng.integers(0, 5),
                             "daytime": "Day", "weather": rng.choice(["shine", "rain"]),
                             "avg_vehicle_total": rng.uniform(0, 10)})
    root = _tmpdir()
    cwd = os.getcwd()
    try:
        os.makedirs(os.path.join(root, "summary_data"))
        pd.DataFrame(rows).to_csv(os.path.join(root, "summary_data", "all_pedestrian_info.csv"), index=False)
        pd.DataFrame(vids).to_csv(os.path.join(root, "summary_data", "all_video_info.csv"), index=False)
        os.chdir(root)
        si.main()
        vp = pd.read_csv("summary_data/inferential_variance_partition.csv")
        assoc = pd.read_csv("summary_data/inferential_associations.csv")
        assert set(vp["outcome"].str.split().str[0]) == {"crossing_speed_mps", "run_red_light", "risky_crossing"}
        assert vp["icc_city_or_video"].between(0, 1).all()
        est = assoc.dropna(subset=["estimate"])
        assert {"run_red_light", "risky_crossing", "crossing_speed_mps"} <= set(est["outcome"])
        assert (est["ci_low"] <= est["estimate"]).all() and (est["estimate"] <= est["ci_high"]).all()
        assert est["q_value_bh"].between(0, 1).all()
    finally:
        os.chdir(cwd)
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"ok  {t.__name__}")
    print(f"{len(tests)} tests passed")
