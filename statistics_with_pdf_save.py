import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import seaborn as sns

def save_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"CSV saved: {filename}")

def save_bar_chart_pdf(df, x_col, y_cols, filename, title="Bar Chart"):
    # Guard against empty/degenerate sections: set_index(x_col) would otherwise raise KeyError
    # and abort the whole report when a category had no matching rows.
    if df is None or df.empty or x_col not in df.columns:
        print(f"PDF skipped (no data): {filename}")
        return

    # Pass figsize to plot() directly; a bare plt.figure() before df.plot() was ignored because
    # df.plot() creates its own figure/axes.
    ax = df.set_index(x_col)[y_cols].plot(kind="bar", figsize=(10, 6))

    ax.set_title(title)
    ax.set_ylabel("Probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8, padding=2)

    with PdfPages(filename) as pdf:
        pdf.savefig(ax.figure)
    plt.close(ax.figure)
    print(f"PDF saved: {filename}")



# ========== 1. duration of videos and analysis ==========
df = pd.read_csv("./summary_data/all_time_info.csv")
stats_seconds = df[["duration_seconds", "analysis_seconds"]].agg(["mean", "std"])
df["duration_minutes"] = df["duration_seconds"] / 60
df["analysis_minutes"] = df["analysis_seconds"] / 60
stats_minutes = df[["duration_minutes", "analysis_minutes"]].agg(["mean", "std"])

print(stats_seconds)
print(stats_minutes)
save_csv(stats_minutes.reset_index(), "./summary_data/time_stats.csv")
save_bar_chart_pdf(stats_minutes.reset_index(), "index", ["duration_minutes", "analysis_minutes"], "./summary_data/time_stats.pdf", "Video Duration vs Analysis Time (minutes)")


def _rate_by_group(frame, key, metrics, min_n=10):
    """Group means with their own denominators, published alongside the means.

    groupby().mean() is skipna, so since risky_crossing became nullable (2026-09-07) each
    metric's denominator differs from the group size and from the other metrics' -- an
    invisible difference that made 'female 0.7442 / male 0.7442' look like a finding when
    it was an artifact. Emit <metric>_n per cell and blank any group under min_n rather
    than printing a confident percentage off a handful of observations.
    """
    out = []
    for gval, grp in frame.groupby(key):
        row = {key: gval}
        for m in metrics:
            vals = pd.to_numeric(grp[m], errors="coerce").dropna()
            row[m] = vals.mean() if len(vals) >= min_n else None
            row[f"{m}_n"] = len(vals)
        out.append(row)
    return pd.DataFrame(out)


# ========== 2. gender ==========
df = pd.read_csv("./summary_data/all_pedestrian_info.csv")
df_gender = df.dropna(subset=["gender"])
gender_stats = _rate_by_group(df_gender, "gender", ["risky_crossing", "run_red_light"])
print(gender_stats)
save_csv(gender_stats, "./summary_data/gender_stats.csv")
save_bar_chart_pdf(gender_stats, "gender", ["risky_crossing", "run_red_light"], "./summary_data/gender_stats.pdf", "Risky Crossing & Red Light Violation Rate by Gender")


# ========== 3. age ==========
df_age = df.dropna(subset=["age"])
age_stats = _rate_by_group(df_age, "age", ["risky_crossing", "run_red_light"])
print(age_stats)
save_csv(age_stats, "./summary_data/age_stats.csv")
save_bar_chart_pdf(age_stats, "age", ["risky_crossing", "run_red_light"], "./summary_data/age_stats.pdf", "Risky Crossing & Red Light Violation Rate by Age")


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
results = []
for clothing_type, cols in clothing_groups.items():
    subset = df[df[cols].sum(axis=1) > 0]
    if len(subset) > 0:
        results.append({
            "clothing_type": clothing_type,
            "risky_crossing_rate(%)": subset["risky_crossing"].mean(skipna=True) * 100,
            "run_red_light_rate(%)": subset["run_red_light"].mean(skipna=True) * 100,
        })
results_df = pd.DataFrame(results)
print(results_df)
save_csv(results_df, "./summary_data/clothing_stats.csv")
save_bar_chart_pdf(results_df, "clothing_type", ["risky_crossing_rate(%)", "run_red_light_rate(%)"], "./summary_data/clothing_stats.pdf", "Risky Crossing & Red Light Violation Rate by Clothing")


# ========== 5. accident and road condition ==========
env_cols = ["police_car", "arrow_board", "cones", "accident", "crack", "potholes"]
results = []
for col in env_cols:
    subset = df[df[col] == 1]
    if len(subset) > 0:
        results.append({
            "environment_factor": col,
            "risky_crossing_rate(%)": subset["risky_crossing"].mean(skipna=True) * 100,
            "run_red_light_rate(%)": subset["run_red_light"].mean(skipna=True) * 100,
        })
results_df = pd.DataFrame(results)
print(results_df)
save_csv(results_df, "./summary_data/accident_road_condition_stats.csv")
save_bar_chart_pdf(results_df, "environment_factor", ["risky_crossing_rate(%)", "run_red_light_rate(%)"], "./summary_data/accident_road_condition_stats.pdf", "Risky Crossing & Red Light Violation Rate by Environment")


# ========== 6. phone ==========
subset_phone = df[df["phone_using"] == 1]
results_phone = []
if len(subset_phone) > 0:
    results_phone.append({
        "accessory": "phone_using",
        "risky_crossing_rate(%)": subset_phone["risky_crossing"].mean(skipna=True) * 100,
        "run_red_light_rate(%)": subset_phone["run_red_light"].mean(skipna=True) * 100,
    })
results_phone_df = pd.DataFrame(results_phone)
print(results_phone_df)

save_csv(results_phone_df, "./summary_data/phone_stats.csv")
save_bar_chart_pdf(
    results_phone_df,
    "accessory",
    ["risky_crossing_rate(%)", "run_red_light_rate(%)"],
    "./summary_data/phone_stats.pdf",
    "Risky Crossing & Red Light Violation Rate by Phone"
)


# ========== 7. carried items ==========
carried_cols = ["backpack", "umbrella", "handbag", "suitcase"]
results_carried = []
for col in carried_cols:
    subset = df[df[col] == 1]
    if len(subset) > 0:
        results_carried.append({
            "accessory": col,
            "risky_crossing_rate(%)": subset["risky_crossing"].mean(skipna=True) * 100,
            "run_red_light_rate(%)": subset["run_red_light"].mean(skipna=True) * 100,
        })

results_carried_df = pd.DataFrame(results_carried)
print(results_carried_df)

save_csv(results_carried_df, "./summary_data/carried_items_stats.csv")
save_bar_chart_pdf(
    results_carried_df,
    "accessory",
    ["risky_crossing_rate(%)", "run_red_light_rate(%)"],
    "./summary_data/carried_items_stats.pdf",
    "Risky Crossing & Red Light Violation Rate by Carried Items"
)



# ========== 8. vehicle ==========
df["three wheelers"] = ((df["auto rickshaw"] == 1) | (df["rickshaw"] == 1) | (df["three wheelers -CNG-"] == 1))
# 'human hauler' is the real [V6]/[A2] column; the old list mistakenly split it into
# nonexistent 'human' and 'hauler' (kept out of the list — they are legacy always-empty
# columns retained in all_pedestrian_info.csv only for schema compatibility).
vehicle_cols = ["ambulance", "army vehicle", "bicycle", "bus", "car","garbagevan", "human hauler", "minibus", "minivan", "motorbike","pickup", "policecar", "scooter", "suv", "taxi", "truck", "van","wheelbarrow", "three wheelers"]
results = []
for col in vehicle_cols:
    subset = df[df[col] == 1]
    if len(subset) > 0:
        results.append({
            "vehicle_type": col,
            "risky_crossing_rate(%)": subset["risky_crossing"].mean(skipna=True) * 100,
            "run_red_light_rate(%)": subset["run_red_light"].mean(skipna=True) * 100,
        })
results_df = pd.DataFrame(results)
print(results_df)
save_csv(results_df, "./summary_data/vehicle_stats.csv")
save_bar_chart_pdf(results_df, "vehicle_type", ["risky_crossing_rate(%)", "run_red_light_rate(%)"], "./summary_data/vehicle_stats.pdf", "Risky Crossing & Red Light Violation Rate by Vehicle")


# ========== 9. traffic condition ==========
# PAIRWISE, not listwise. risky_crossing is NA for ~64% of rows, and a listwise dropna
# here also deleted those rows from the run_red_light correlations -- flipping signs on a
# metric that has nothing to do with the risky-crossing change.
subset = df[["avg_vehicle_total", "avg_road_width", "risky_crossing", "run_red_light"]].apply(
    pd.to_numeric, errors="coerce")
results_df = subset.corr(min_periods=30)[["risky_crossing", "run_red_light"]].loc[
    ["avg_vehicle_total", "avg_road_width"]]
print(results_df)
save_csv(results_df.reset_index(), "./summary_data/road_corr.csv")
save_bar_chart_pdf(results_df.reset_index(), "index", ["risky_crossing", "run_red_light"], "./summary_data/road_corr.pdf", "Correlation with Risky Crossing & Red Light Violation Rate")


# ========== 10. time of day and weather ==========
df = pd.read_csv("./summary_data/all_pedestrian_info.csv", encoding='utf-8')
# Drop only on the grouping keys. risky_crossing is NA for ~64% of rows since the
# 2026-09-07 evidence-positive change, and listwise-deleting on it silently re-based
# run_red_light_prob -- a metric this file's other sections compute over a different,
# larger population -- moving it from 14.5% to 41.0%. Each rate is now computed with
# skipna over its OWN valid subset, and each carries its own n so a cell resting on a
# handful of observations is visible rather than plotted as a confident percentage.
df_filtered = df.dropna(subset=["weather", "daytime"])
weathers = df_filtered["weather"].unique()
daytimes = df_filtered["daytime"].unique()
combinations = list(itertools.product(weathers, daytimes))
MIN_N = 10
results = []
for weather, daytime in combinations:
    subset = df_filtered[(df_filtered["weather"] == weather) & (df_filtered["daytime"] == daytime)]
    if len(subset) == 0: continue
    rr = pd.to_numeric(subset["run_red_light"], errors="coerce").dropna()
    rc = pd.to_numeric(subset["risky_crossing"], errors="coerce").dropna()
    results.append({
        "weather": weather,
        "daytime": daytime,
        "run_red_light_prob": rr.mean() * 100 if len(rr) >= MIN_N else None,
        "run_red_light_n": len(rr),
        "risky_crossing_prob": rc.mean() * 100 if len(rc) >= MIN_N else None,
        "risky_crossing_n": len(rc),
    })
result_df = pd.DataFrame(results)
print(result_df)
save_csv(result_df, "./summary_data/weather_daytime_stats.csv")
save_bar_chart_pdf(result_df, "daytime", ["run_red_light_prob", "risky_crossing_prob"], "./summary_data/weather_daytime_stats.pdf", "Risky Crossing & Red Light Violation Rate by Weather & Daytime")


# ========== 11. continent ==========
df = pd.read_csv("./summary_data/all_pedestrian_info.csv")
continents = df["continent"].dropna().unique()
results = []
for cont in continents:
    subset = df[df["continent"] == cont]
    if len(subset) == 0: continue
    results.append({
        "continent": cont,
        "run_red_light_rate(%)": subset["run_red_light"].mean(skipna=True) * 100,
        "risky_crossing_rate(%)": subset["risky_crossing"].mean(skipna=True) * 100,
    })
results_df = pd.DataFrame(results)
print(results_df)
save_csv(results_df, "./summary_data/continent_stats.csv")
save_bar_chart_pdf(results_df, "continent", ["risky_crossing_rate(%)", "run_red_light_rate(%)"], "./summary_data/continent_stats.pdf", "Risky Crossing & Red Light Violation Rate by Continent")


# ========== 12. continent crosswalk coeff ==========
df = pd.read_csv("./summary_data/all_video_info.csv", encoding='utf-8')
df_filtered = df.dropna(subset=["crosswalk_usage_ratio", "crosswalk_prob"])
df_filtered = df_filtered[(df_filtered["crosswalk_usage_ratio"] != 0) & (df_filtered["crosswalk_prob"] != 0)]
continent_stats = df_filtered.groupby("continent")[["crosswalk_usage_ratio", "crosswalk_prob"]].mean()
continent_stats["crosswalk_coeff"] = continent_stats["crosswalk_usage_ratio"] / continent_stats["crosswalk_prob"]
continent_stats = continent_stats.sort_values("crosswalk_coeff", ascending=False)
print(continent_stats)
save_csv(continent_stats.reset_index(), "./summary_data/crosswalk_coeff.csv")
save_bar_chart_pdf(continent_stats.reset_index(), "continent", ["crosswalk_usage_ratio", "crosswalk_prob", "crosswalk_coeff"], "./summary_data/crosswalk_coeff.pdf", "Crosswalk Coefficient by Continent")


# ========== 13. continent speed and decision time ==========
df = pd.read_csv("./summary_data/all_video_info.csv")
# Prefer the MEASURED per-video walking speed (from [S1], via video_info) over the imported
# city-constant crossing_speed; fall back to the constant where no measurement exists.
if "measured_avg_walking_speed_mps" in df.columns:
    df["measured_walking_speed"] = df["measured_avg_walking_speed_mps"].fillna(df["crossing_speed"])
else:
    df["measured_walking_speed"] = df["crossing_speed"]
continents = df["continent"].dropna().unique()
results = []
for cont in continents:
    subset = df[df["continent"] == cont]
    if len(subset) == 0: continue
    results.append({
        "continent": cont,
        "avg_decision_time": subset["crossing_time"].mean(skipna=True),
        "avg_crossing_speed": subset["crossing_speed"].mean(skipna=True),
        "avg_measured_walking_speed": subset["measured_walking_speed"].mean(skipna=True),
    })
results_df = pd.DataFrame(results)
print(results_df)
save_csv(results_df, "./summary_data/crossing_stats.csv")
save_bar_chart_pdf(results_df, "continent", ["avg_decision_time", "avg_crossing_speed", "avg_measured_walking_speed"], "./summary_data/crossing_stats.pdf", "Decision Time, Imported & Measured Speed by Continent")


# ========== 13. others ==========
df = pd.read_csv("./summary_data/all_video_info.csv")
cols = ["population_city", "population_country", "traffic_mortality", "literacy_rate", "avg_height", "med_age", "gini","crossing_time", "crossing_speed", "risky_crossing_ratio", "run_red_light_ratio"]
# Pairwise: a listwise dropna across 11 columns lets one sparse column (risky_crossing_ratio
# is NULL for 9 of 28 videos) decide which rows every OTHER pair is computed on.
subset = df[cols].apply(pd.to_numeric, errors="coerce")
corr = subset.corr(min_periods=8)
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.tight_layout()
with PdfPages("./summary_data/correlation_heatmap.pdf") as pdf:
    pdf.savefig()
plt.close()
print("PDF saved: correlation_heatmap.pdf")
