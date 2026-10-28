import pandas as pd
import os
from modules.paths import RESULTS_ROOT

OUTPUT_COLUMNS = [
    'track_id', 'crossed', 'nearby_count_beginning', 'nearby_count_whole',
    'risky_crossing', 'run_red_light', 'crosswalk_use_or_not', 'gender', 'age',
    'phone_using', 'walking_speed_mps', 'crossing_speed_mps', 'decision_delay_s',
    'backpack', 'umbrella', 'handbag', 'suitcase', 'short_sleeved_shirt',
    'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest',
    'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress',
    'long_sleeved_dress', 'vest_dress', 'sling_dress', 'weather', 'daytime',
    'police_car', 'arrow_board', 'cones', 'accident', 'crack', 'potholes',
    'avg_vehicle_total', 'crossing_sign', 'avg_road_width', 'crosswalk', 'sidewalk',
]

def summary_all_info(video_path, personal_info_csv=None, env_info_csv=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(RESULTS_ROOT, video_name)
    os.makedirs(output_dir, exist_ok=True)

    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[A2]pedestrian_info.csv")
    if personal_info_csv is None:
        personal_info_csv = os.path.join(output_dir, "[C7]crossing_pe_info.csv")
    if env_info_csv is None:
        env_info_csv = os.path.join(output_dir, "[C9]crossing_env_info.csv")
    os.makedirs(os.path.dirname(os.path.abspath(output_csv_path)), exist_ok=True)

    def safe_read_csv(path):
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                pass
        return pd.DataFrame()

    per_df = safe_read_csv(personal_info_csv)
    env_df = safe_read_csv(env_info_csv)

    if per_df.empty or env_df.empty:
        columns = list(dict.fromkeys(OUTPUT_COLUMNS + list(per_df.columns) + list(env_df.columns)))
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False, encoding="utf-8")
        print(f"Input file missing/empty, created empty CSV at: {output_csv_path}")
        return

    if "crossed" in env_df.columns:
        env_df = env_df.drop(columns=["crossed"])

    merged_df = pd.merge(per_df, env_df, on="track_id", how="inner")
    merged_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Pedestrian stats summary saved to: {output_csv_path}")
