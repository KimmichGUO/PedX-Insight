"""Descriptive statistics over summary_data/ -> per-topic CSVs + PDFs.

Design rules (2026-10, see modules/stats_utils.py for the reasoning):

  * Pedestrians are clustered in cities. Every group rate is published with its n, its
    number of cities, and a CLUSTER-bootstrap 95% CI (whole cities resampled). A rate
    resting on fewer than MIN_N observations or MIN_CLUSTERS cities is written BLANK
    (the Visualizer importer skips blank cells) with its n still shown.
  * NA is not 0. risky_crossing is NA when no signal state was observed; every rate is
    computed over its own non-missing subset, never after listwise deletion on another
    column.
  * Continent tables weight each CITY equally (mean of city rates). Pooling pedestrians
    let one busy video decide a continent: Manila alone is 358 of 1371 rows.
  * Measured and imported quantities are never mixed in one column. crossing_time and
    crossing_speed in all_video_info.csv are city constants imported from an external
    study (crossing_*.csv); the measured counterpart is measured_crossing_speed_mps.
  * City-level correlations are computed on ONE ROW PER CITY. Socio-economic covariates
    are constant within a city, so correlating them over video rows counts a city with
    three videos three times.

Column names consumed by PedX-Visualizer/scripts/aggregate-csv-data.js are preserved;
added columns (<metric>_n, _n_clusters, _ci_low, _ci_high) are ignored by it.
Inferential models (GEE / mixed models, variance partition) live in
statistics_inferential.py.
"""

import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

from modules.provenance import git_state, utc_now
from modules.stats_utils import (AGE_LABELS, MIN_CLUSTERS, MIN_N, age_band, bh_fdr,
                                 cluster_bootstrap_ci, city_level, rate_columns,
                                 rate_with_ci)

OUT = "./summary_data"
# A video's measured crossing speed is used only when it rests on at least this many
# crossers (Milan's 4.19 m/s rests on 3).
MIN_SPEED_CROSSERS = 5
# risky_crossing_ratio is evidence-positive and must be gated on coverage (see
# modules/risky_crossing/risky_crossing.py). Below this share of determinable crossings
# the ratio rests on a subsample biased toward detector-friendly scenes.
MIN_RISKY_COVERAGE = 0.25
# A city contributes to a continent rate only with this many observations of the metric.
MIN_CITY_N = 5
# City-level correlations need at least this many cities with both values.
MIN_CITIES_CORR = 8


def save_csv(df, filename):
    df.to_csv(filename, index=False, encoding="utf-8")
    print(f"CSV saved: {filename}")


def save_bar_chart_pdf(df, x_col, y_cols, filename, title="Bar Chart", ylabel="Rate"):
    """Grouped bars with cluster-bootstrap CI error bars and n under each bar.

    Blank (under-powered) cells are drawn as an empty slot labelled with their n, so a
    missing bar is visibly 'not enough data', not 'zero'.
    """
    if df is None or df.empty or x_col not in df.columns:
        print(f"PDF skipped (no data): {filename}")
        return
    y_cols = [c for c in y_cols if c in df.columns]
    if not y_cols:
        print(f"PDF skipped (no metric columns): {filename}")
        return

    x = np.arange(len(df))
    width = 0.8 / len(y_cols)
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(df) * len(y_cols) + 3), 6))
    for i, col in enumerate(y_cols):
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
        pos = x + (i - (len(y_cols) - 1) / 2) * width
        lo_col, hi_col = f"{col}_ci_low", f"{col}_ci_high"
        yerr = None
        if lo_col in df.columns and hi_col in df.columns:
            lo = pd.to_numeric(df[lo_col], errors="coerce").to_numpy(float)
            hi = pd.to_numeric(df[hi_col], errors="coerce").to_numpy(float)
            yerr = np.vstack([np.nan_to_num(np.clip(vals - lo, 0, None)),
                              np.nan_to_num(np.clip(hi - vals, 0, None))])
        ax.bar(pos, np.nan_to_num(vals), width, yerr=yerr, capsize=3, label=col)
        n_col, k_col = f"{col}_n", f"{col}_n_clusters"
        ns = df[n_col] if n_col in df.columns else [None] * len(df)
        ks = df[k_col] if k_col in df.columns else [None] * len(df)
        for p, v, n, k in zip(pos, vals, ns, ks):
            # k = number of cities behind the bar; no error bar is drawn below 5 cities,
            # so k is what tells the reader a bar rests on two or three cities.
            n_txt = f"n={int(n)}" if n is not None and pd.notna(n) else ""
            if k is not None and pd.notna(k):
                n_txt += f"\nk={int(k)}"
            txt = f"{v:.2f}\n{n_txt}" if np.isfinite(v) else f"(too few)\n{n_txt}"
            ax.annotate(txt, (p, np.nan_to_num(v)), ha="center", va="bottom", fontsize=7,
                        xytext=(0, 2), textcoords="offset points")
    ax.set_xticks(x)
    ax.set_xticklabels(df[x_col].astype(str), rotation=45, ha="right")
    ax.set_title(title + "\n(error bars: 95% CI, cities resampled)")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    with PdfPages(filename) as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print(f"PDF saved: {filename}")


def _with_cluster(df):
    """City cluster id. City names alone collide across countries (two Londons).

    fillna BEFORE concatenating: under pandas 3, str + NA is NA, and a NA cluster id
    silently drops the row from every clustered computation.
    """
    df = df.copy()
    country = df["country"].fillna("").astype(str) if "country" in df.columns else ""
    df["cluster"] = df["city"].fillna("").astype(str) + "|" + country
    return df


RATE_METRICS = ["risky_crossing", "run_red_light"]
PCT_NAMES = {"risky_crossing": "risky_crossing_rate(%)", "run_red_light": "run_red_light_rate(%)"}


def rates_for_masks(df, label_col, masks, *, pct=True):
    """One row per (label, boolean mask) with clustered rates for RATE_METRICS."""
    rows = []
    for label, mask in masks:
        sub = df[mask]
        if sub.empty:
            continue
        row = {label_col: label}
        row.update(rate_columns(sub, RATE_METRICS, "cluster", scale=100.0 if pct else 1.0,
                                names=PCT_NAMES if pct else None))
        rows.append(row)
    return pd.DataFrame(rows)


def city_weighted_row(frame, metrics, *, scale=1.0, names=None, min_city_n=MIN_CITY_N):
    """Mean of per-city rates (each city weighted equally) with a bootstrap CI over cities.

    `<m>_n` is the number of observations behind the cities used; `<m>_n_clusters` the
    number of cities that met `min_city_n`.
    """
    names = names or {}
    row = {}
    for m in metrics:
        col = names.get(m, m)
        vals = pd.to_numeric(frame[m], errors="coerce")
        g = pd.DataFrame({"v": vals, "c": frame["cluster"]}).dropna().groupby("c")["v"]
        per_city = g.mean()[g.count() >= min_city_n]
        n_obs = int(g.count()[g.count() >= min_city_n].sum())
        r = rate_with_ci(per_city.values, per_city.index, scale=scale, min_n=MIN_CLUSTERS)
        row[col] = r["estimate"]
        row[f"{col}_n"] = n_obs
        row[f"{col}_n_clusters"] = r["n_clusters"]
        row[f"{col}_ci_low"] = r["ci_low"]
        row[f"{col}_ci_high"] = r["ci_high"]
    return row


def main():
    os.makedirs(OUT, exist_ok=True)

    # ========== 1. duration of videos and analysis ==========
    df = pd.read_csv(f"{OUT}/all_time_info.csv")
    df["duration_minutes"] = df["duration_seconds"] / 60
    df["analysis_minutes"] = df["analysis_seconds"] / 60
    stats_minutes = df[["duration_minutes", "analysis_minutes"]].agg(["mean", "std"])
    print(stats_minutes)
    save_csv(stats_minutes.reset_index(), f"{OUT}/time_stats.csv")
    save_bar_chart_pdf(stats_minutes.reset_index(), "index", ["duration_minutes", "analysis_minutes"],
                       f"{OUT}/time_stats.pdf", "Video Duration vs Analysis Time", "minutes")

    ped = _with_cluster(pd.read_csv(f"{OUT}/all_pedestrian_info.csv", encoding="utf-8"))
    for col in RATE_METRICS:
        ped[col] = pd.to_numeric(ped[col], errors="coerce")

    # ========== 2. gender ==========
    rows = []
    for g, grp in ped.dropna(subset=["gender"]).groupby("gender"):
        row = {"gender": g}
        row.update(rate_columns(grp, RATE_METRICS, "cluster"))
        rows.append(row)
    gender_stats = pd.DataFrame(rows)
    print(gender_stats)
    save_csv(gender_stats, f"{OUT}/gender_stats.csv")
    save_bar_chart_pdf(gender_stats, "gender", RATE_METRICS, f"{OUT}/gender_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Gender", "rate")

    # ========== 3. age (life-stage bands; ages are model-inferred) ==========
    ped["age_band"] = age_band(ped["age"])
    rows = []
    for band in AGE_LABELS:
        grp = ped[ped["age_band"] == band]
        if grp.empty:
            continue
        row = {"age": band}
        row.update(rate_columns(grp, RATE_METRICS, "cluster"))
        rows.append(row)
    age_stats = pd.DataFrame(rows)
    print(age_stats)
    save_csv(age_stats, f"{OUT}/age_stats.csv")
    save_bar_chart_pdf(age_stats, "age", RATE_METRICS, f"{OUT}/age_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Age Band", "rate")

    # ========== 4. clothing ==========
    clothing_groups = {
        "shirt": ["short_sleeved_shirt", "long_sleeved_shirt"],
        "outerwear": ["short_sleeved_outwear", "long_sleeved_outwear"],
        "dress": ["short_sleeved_dress", "long_sleeved_dress"],
        "vest": ["vest"],
        "sling": ["sling"],
        "shorts": ["shorts"],
        "trousers": ["trousers"],
        "skirt": ["skirt"],
    }
    masks = [(name, ped[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1) > 0)
             for name, cols in clothing_groups.items()]
    results_df = rates_for_masks(ped, "clothing_type", masks)
    print(results_df)
    save_csv(results_df, f"{OUT}/clothing_stats.csv")
    save_bar_chart_pdf(results_df, "clothing_type", list(PCT_NAMES.values()), f"{OUT}/clothing_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Clothing", "%")

    # ========== 5. accident and road condition ==========
    env_cols = ["police_car", "arrow_board", "cones", "accident", "crack", "potholes"]
    masks = [(c, pd.to_numeric(ped[c], errors="coerce") == 1) for c in env_cols]
    results_df = rates_for_masks(ped, "environment_factor", masks)
    print(results_df)
    save_csv(results_df, f"{OUT}/accident_road_condition_stats.csv")
    save_bar_chart_pdf(results_df, "environment_factor", list(PCT_NAMES.values()),
                       f"{OUT}/accident_road_condition_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Environment", "%")

    # ========== 6. phone ==========
    results_df = rates_for_masks(ped, "accessory",
                                 [("phone_using", pd.to_numeric(ped["phone_using"], errors="coerce") == 1)])
    print(results_df)
    save_csv(results_df, f"{OUT}/phone_stats.csv")
    save_bar_chart_pdf(results_df, "accessory", list(PCT_NAMES.values()), f"{OUT}/phone_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Phone", "%")

    # ========== 7. carried items ==========
    carried_cols = ["backpack", "umbrella", "handbag", "suitcase"]
    masks = [(c, pd.to_numeric(ped[c], errors="coerce") == 1) for c in carried_cols]
    results_df = rates_for_masks(ped, "accessory", masks)
    print(results_df)
    save_csv(results_df, f"{OUT}/carried_items_stats.csv")
    save_bar_chart_pdf(results_df, "accessory", list(PCT_NAMES.values()), f"{OUT}/carried_items_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Carried Items", "%")

    # ========== 8. vehicle ==========
    num = lambda c: pd.to_numeric(ped[c], errors="coerce") if c in ped.columns else pd.Series(np.nan, index=ped.index)
    ped["three wheelers"] = ((num("auto rickshaw") == 1) | (num("rickshaw") == 1)
                             | (num("three wheelers -CNG-") == 1)).astype(int)
    # 'human hauler' is the real [V6]/[A2] column; legacy 'human'/'hauler' are always empty.
    vehicle_cols = ["ambulance", "army vehicle", "bicycle", "bus", "car", "garbagevan", "human hauler",
                    "minibus", "minivan", "motorbike", "pickup", "policecar", "scooter", "suv", "taxi",
                    "truck", "van", "wheelbarrow", "three wheelers"]
    masks = [(c, num(c) == 1) for c in vehicle_cols]
    results_df = rates_for_masks(ped, "vehicle_type", masks)
    print(results_df)
    save_csv(results_df, f"{OUT}/vehicle_stats.csv")
    save_bar_chart_pdf(results_df, "vehicle_type", list(PCT_NAMES.values()), f"{OUT}/vehicle_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Vehicle", "%")

    # ========== 9. traffic condition ==========
    # Point-biserial (Pearson on a 0/1 outcome), PAIRWISE over each pair's own complete
    # cases, with a city-resampled CI because the rows are clustered.
    rows = []
    for xvar in ["avg_vehicle_total", "avg_road_width"]:
        row = {"index": xvar}
        for yvar in RATE_METRICS:
            sub = ped[["cluster", xvar, yvar]].copy()
            sub[xvar] = pd.to_numeric(sub[xvar], errors="coerce")
            sub = sub.dropna()
            n, k = len(sub), sub["cluster"].nunique()
            r = None
            if n >= 30 and k >= MIN_CLUSTERS and sub[xvar].std() > 0 and sub[yvar].std() > 0:
                r = float(sub[xvar].corr(sub[yvar]))

            def _corr(s, xvar=xvar, yvar=yvar):
                if s[xvar].std() == 0 or s[yvar].std() == 0:
                    return np.nan
                return s[xvar].corr(s[yvar])
            lo, hi = cluster_bootstrap_ci(sub, "cluster", _corr) if r is not None else (None, None)
            row.update({yvar: r, f"{yvar}_n": n, f"{yvar}_n_clusters": k,
                        f"{yvar}_ci_low": lo, f"{yvar}_ci_high": hi})
        rows.append(row)
    results_df = pd.DataFrame(rows)
    print(results_df)
    save_csv(results_df, f"{OUT}/road_corr.csv")
    save_bar_chart_pdf(results_df, "index", RATE_METRICS, f"{OUT}/road_corr.pdf",
                       "Correlation with Risky Crossing & Red Light Violation", "Pearson r")

    # ========== 10. time of day and weather ==========
    # Grouping keys only are dropped; each rate uses its own valid subset.
    df_filtered = ped.dropna(subset=["weather", "daytime"])
    rows = []
    for weather, daytime in itertools.product(sorted(df_filtered["weather"].unique()),
                                              sorted(df_filtered["daytime"].unique())):
        sub = df_filtered[(df_filtered["weather"] == weather) & (df_filtered["daytime"] == daytime)]
        if sub.empty:
            continue
        row = {"weather": weather, "daytime": daytime}
        row.update(rate_columns(sub, ["run_red_light", "risky_crossing"], "cluster", scale=100.0,
                                names={"run_red_light": "run_red_light_prob",
                                       "risky_crossing": "risky_crossing_prob"}))
        rows.append(row)
    result_df = pd.DataFrame(rows)
    print(result_df)
    save_csv(result_df, f"{OUT}/weather_daytime_stats.csv")
    if not result_df.empty:
        plot_df = result_df.assign(weather_daytime=result_df["weather"] + " / " + result_df["daytime"])
        save_bar_chart_pdf(plot_df, "weather_daytime", ["run_red_light_prob", "risky_crossing_prob"],
                           f"{OUT}/weather_daytime_stats.pdf",
                           "Risky Crossing & Red Light Violation Rate by Weather & Daytime", "%")

    # ========== 11. continent (each city weighted equally) ==========
    rows = []
    for cont, grp in ped.dropna(subset=["continent"]).groupby("continent"):
        row = {"continent": cont}
        row.update(city_weighted_row(grp, RATE_METRICS, scale=100.0, names=PCT_NAMES))
        rows.append(row)
    results_df = pd.DataFrame(rows)
    print(results_df)
    save_csv(results_df, f"{OUT}/continent_stats.csv")
    save_bar_chart_pdf(results_df, "continent", list(PCT_NAMES.values()), f"{OUT}/continent_stats.pdf",
                       "Risky Crossing & Red Light Violation Rate by Continent (city-weighted)", "%")

    # ---------- video-level frame shared by sections 12-14 ----------
    vid = _with_cluster(pd.read_csv(f"{OUT}/all_video_info.csv", encoding="utf-8"))
    for col in ["measured_crossing_speed_mps", "measured_crossing_speed_n", "risky_crossing_ratio",
                "risky_crossing_coverage", "run_red_light_ratio", "crosswalk_usage_ratio",
                "crosswalk_prob", "traffic_light_prob", "crossing_time", "crossing_speed"]:
        if col in vid.columns:
            vid[col] = pd.to_numeric(vid[col], errors="coerce")
    # Measured crossing speed only where it rests on enough crossers.
    vid["measured_crossing_speed_gated"] = vid["measured_crossing_speed_mps"].where(
        vid["measured_crossing_speed_n"] >= MIN_SPEED_CROSSERS)
    # Risky ratio only where enough crossings were determinable.
    vid["risky_crossing_ratio_gated"] = vid["risky_crossing_ratio"].where(
        vid["risky_crossing_coverage"] >= MIN_RISKY_COVERAGE)

    # ========== 12. continent crosswalk coefficient ==========
    # All videos count, zeros included: dropping zero-usage / zero-presence videos selected
    # on the outcome. The coefficient (usage / presence) is published only where presence
    # is high enough to divide by -- with crosswalk_prob at most 0.18 in this dataset, a
    # ratio of two near-zero numbers is noise.
    rows = []
    for cont, grp in vid.dropna(subset=["continent"]).groupby("continent"):
        row = {"continent": cont}
        row.update(city_weighted_row(grp, ["crosswalk_usage_ratio", "crosswalk_prob"], min_city_n=1))
        prob, usage = row.get("crosswalk_prob"), row.get("crosswalk_usage_ratio")
        row["crosswalk_coeff"] = (usage / prob if prob is not None and usage is not None
                                  and prob >= 0.05 else None)
        rows.append(row)
    continent_stats = pd.DataFrame(rows)
    if not continent_stats.empty:
        continent_stats = continent_stats.sort_values("continent")
    print(continent_stats)
    save_csv(continent_stats, f"{OUT}/crosswalk_coeff.csv")
    save_bar_chart_pdf(continent_stats, "continent", ["crosswalk_usage_ratio", "crosswalk_prob"],
                       f"{OUT}/crosswalk_coeff.pdf", "Crosswalk Usage and Presence by Continent (city-weighted)",
                       "ratio")

    # ========== 13. continent speed and decision time ==========
    # avg_crossing_speed is MEASURED ([S1] crossing speed, videos with >= MIN_SPEED_CROSSERS
    # crossers, each city weighted equally). The externally imported city constant is kept
    # separately as imported_crossing_speed; avg_decision_time is the imported crossing_time
    # (PedX has no measurement of that construct). City constants are taken once per city.
    rows = []
    for cont, grp in vid.dropna(subset=["continent"]).groupby("continent"):
        row = {"continent": cont}
        row.update(city_weighted_row(grp, ["measured_crossing_speed_gated"], min_city_n=1,
                                     names={"measured_crossing_speed_gated": "avg_crossing_speed"}))
        per_city = grp.groupby("cluster")[["crossing_time", "crossing_speed"]].first()
        row["avg_decision_time"] = per_city["crossing_time"].mean() if per_city["crossing_time"].notna().any() else None
        row["avg_decision_time_n_clusters"] = int(per_city["crossing_time"].notna().sum())
        row["imported_crossing_speed"] = per_city["crossing_speed"].mean() if per_city["crossing_speed"].notna().any() else None
        row["imported_crossing_speed_n_clusters"] = int(per_city["crossing_speed"].notna().sum())
        row["crossing_speed_source"] = "measured_[S1]"
        row["decision_time_source"] = "imported_city_constant"
        rows.append(row)
    results_df = pd.DataFrame(rows)
    print(results_df)
    save_csv(results_df, f"{OUT}/crossing_stats.csv")
    save_bar_chart_pdf(results_df, "continent", ["avg_crossing_speed", "imported_crossing_speed"],
                       f"{OUT}/crossing_stats.pdf",
                       "Measured vs Imported Crossing Speed by Continent (city-weighted)", "m/s")

    # ========== 14. city-level correlations ==========
    socio = ["population_city", "population_country", "traffic_mortality", "literacy_rate",
             "avg_height", "med_age", "gini", "gmp"]
    measured = ["measured_crossing_speed_gated", "risky_crossing_ratio_gated", "run_red_light_ratio"]
    detector = ["traffic_light_prob", "crosswalk_prob"]  # included to expose detector confounds
    imported = ["crossing_time", "crossing_speed"]
    city = city_level(vid, measured + detector, city_col="cluster",
                      constant_cols=socio + imported)
    labels = {"measured_crossing_speed_gated": "measured_crossing_speed",
              "risky_crossing_ratio_gated": "risky_crossing_ratio",
              "crossing_time": "ext_crossing_time", "crossing_speed": "ext_crossing_speed"}
    city = city.rename(columns=labels)
    var_list = [labels.get(v, v) for v in socio + measured + detector + imported if labels.get(v, v) in city.columns]

    pairs = []
    for a, b in itertools.combinations(var_list, 2):
        sub = city[[a, b]].dropna()
        n = len(sub)
        rho = p = None
        if n >= MIN_CITIES_CORR and sub[a].nunique() > 1 and sub[b].nunique() > 1:
            res = stats.spearmanr(sub[a], sub[b])
            rho, p = float(res.statistic), float(res.pvalue)
        pairs.append({"var_x": a, "var_y": b, "spearman_rho": rho, "p_value": p, "n_cities": n})
    corr_df = pd.DataFrame(pairs)
    corr_df["q_value_bh"] = bh_fdr(corr_df["p_value"])
    save_csv(corr_df, f"{OUT}/city_correlations.csv")

    rho_m = pd.DataFrame(np.nan, index=var_list, columns=var_list)
    annot = pd.DataFrame("", index=var_list, columns=var_list)
    for _, r in corr_df.iterrows():
        if r["spearman_rho"] is None or pd.isna(r["spearman_rho"]):
            continue
        star = "*" if pd.notna(r["q_value_bh"]) and r["q_value_bh"] < 0.05 else ""
        for a, b in [(r["var_x"], r["var_y"]), (r["var_y"], r["var_x"])]:
            rho_m.loc[a, b] = r["spearman_rho"]
            annot.loc[a, b] = f"{r['spearman_rho']:.2f}{star}\nn={int(r['n_cities'])}"
    for v in var_list:  # .values is read-only under pandas 3 copy-on-write
        rho_m.loc[v, v] = 1.0
    fig, ax = plt.subplots(figsize=(13, 11))
    sns.heatmap(rho_m.astype(float), annot=annot, fmt="", cmap="coolwarm", vmin=-1, vmax=1,
                cbar=True, ax=ax, annot_kws={"fontsize": 6})
    ax.set_title(f"City-level Spearman correlations (one row per city; blank: < {MIN_CITIES_CORR} cities)\n"
                 "* = BH-FDR q < 0.05 across all pairs; ext_* = imported city constants, not PedX measurements")
    fig.tight_layout()
    with PdfPages(f"{OUT}/correlation_heatmap.pdf") as pdf:
        pdf.savefig(fig)
    plt.close(fig)
    print("PDF saved: correlation_heatmap.pdf")

    # ========== provenance of this aggregation run ==========
    sha, dirty = git_state()
    prov = {
        "generated_utc": utc_now(),
        "git_sha": sha,
        "git_dirty": dirty,
        "n_videos": int(len(vid)),
        "n_cities": int(vid["cluster"].nunique()),
        "n_pedestrians": int(len(ped)),
        "min_n": MIN_N, "min_clusters": MIN_CLUSTERS,
        "min_speed_crossers": MIN_SPEED_CROSSERS, "min_risky_coverage": MIN_RISKY_COVERAGE,
        "min_city_n": MIN_CITY_N, "min_cities_corr": MIN_CITIES_CORR,
        "pipeline_versions": vid["pipeline_version"].value_counts(dropna=False).to_dict()
        if "pipeline_version" in vid.columns else None,
        "code_versions": vid["code_version"].value_counts(dropna=False).to_dict()
        if "code_version" in vid.columns else None,
    }
    with open(f"{OUT}/provenance.json", "w", encoding="utf-8") as f:
        json.dump({k: (str(v) if isinstance(v, float) and np.isnan(v) else v) for k, v in prov.items()},
                  f, indent=2, default=str)
    print(f"Provenance saved: {OUT}/provenance.json")


if __name__ == "__main__":
    main()
