import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools
import seaborn as sns

def save_csv(df, filename):
    df.to_csv(filename, index=False)
    print(f"CSV saved: {filename}")

def save_bar_chart_pdf(df, x_col, y_cols, filename, title="Bar Chart"):
    plt.figure(figsize=(10, 6))
    ax = df.set_index(x_col)[y_cols].plot(kind="bar")

    plt.title(title)
    plt.ylabel("Probability")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", label_type="edge", fontsize=8, padding=2)

    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(filename) as pdf:
        pdf.savefig()
    plt.close()
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


# ========== 2. gender ==========
df = pd.read_csv("./summary_data/all_pedestrian_info.csv")
df_gender = df.dropna(subset=["gender"])
gender_stats = df_gender.groupby("gender")[["risky_crossing", "run_red_light"]].mean().reset_index()
print(gender_stats)
save_csv(gender_stats, "./summary_data/gender_stats.csv")
save_bar_chart_pdf(gender_stats, "gender", ["risky_crossing", "run_red_light"], "./summary_data/gender_stats.pdf", "Risky Crossing & Red Light Violation Rate by Gender")


# ========== 3. age ==========
df_age = df.dropna(subset=["age"])
age_stats = df_age.groupby("age")[["risky_crossing", "run_red_light"]].mean().reset_index()
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
vehicle_cols = ["ambulance", "army vehicle", "bicycle", "bus", "car","garbagevan", "human", "hauler", "minibus", "minivan", "motorbike","pickup", "policecar", "scooter", "suv", "taxi", "truck", "van","wheelbarrow", "three wheelers"]
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
subset = df[["avg_vehicle_total", "avg_road_width", "risky_crossing", "run_red_light"]].dropna()
results_df = subset.corr()[["risky_crossing", "run_red_light"]].loc[["avg_vehicle_total", "avg_road_width"]]
print(results_df)
save_csv(results_df.reset_index(), "./summary_data/road_corr.csv")
save_bar_chart_pdf(results_df.reset_index(), "index", ["risky_crossing", "run_red_light"], "./summary_data/road_corr.pdf", "Correlation with Risky Crossing & Red Light Violation Rate")


# ========== 10. time of day and weather ==========
df = pd.read_csv("./summary_data/all_pedestrian_info.csv", encoding='latin1')
df_filtered = df.dropna(subset=["weather", "daytime", "run_red_light", "risky_crossing"])
weathers = df_filtered["weather"].unique()
daytimes = df_filtered["daytime"].unique()
combinations = list(itertools.product(weathers, daytimes))
results = []
for weather, daytime in combinations:
    subset = df_filtered[(df_filtered["weather"] == weather) & (df_filtered["daytime"] == daytime)]
    if len(subset) == 0: continue
    results.append({
        "weather": weather,
        "daytime": daytime,
        "run_red_light_prob": subset["run_red_light"].mean() * 100,
        "risky_crossing_prob": subset["risky_crossing"].mean() * 100,
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
df = pd.read_csv("./summary_data/all_video_info.csv", encoding='latin1')
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
continents = df["continent"].dropna().unique()
results = []
for cont in continents:
    subset = df[df["continent"] == cont]
    if len(subset) == 0: continue
    results.append({
        "continent": cont,
        "avg_decision_time": subset["crossing_time"].mean(skipna=True),
        "avg_crossing_speed": subset["crossing_speed"].mean(skipna=True),
    })
results_df = pd.DataFrame(results)
print(results_df)
save_csv(results_df, "./summary_data/crossing_stats.csv")
save_bar_chart_pdf(results_df, "continent", ["avg_decision_time", "avg_crossing_speed"], "./summary_data/crossing_stats.pdf", "Average Desicion Time & Crossing Speed by Continent")


# ========== 13. others ==========
df = pd.read_csv("./summary_data/all_video_info.csv")
cols = ["population_city", "population_country", "traffic_mortality", "literacy_rate", "avg_height", "med_age", "gini","crossing_time", "crossing_speed", "risky_crossing_ratio", "run_red_light_ratio"]
subset = df[cols].dropna()
corr = subset.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.tight_layout()
with PdfPages("./summary_data/correlation_heatmap.pdf") as pdf:
    pdf.savefig()
plt.close()
print("PDF saved: correlation_heatmap.pdf")
