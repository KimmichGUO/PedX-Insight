import pandas as pd
import os

def summary_all_info(video_path, personal_info_csv = None, env_info_csv=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)

    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[A2]summary_info.csv")
    if personal_info_csv is None:
        personal_info_csv = os.path.join(output_dir, "[C7]crossing_pe_info.csv")
    if env_info_csv is None:
        env_info_csv = os.path.join(output_dir, "[C9]crossing_env_info.csv")

    per_df = pd.read_csv(personal_info_csv)
    env_df = pd.read_csv(env_info_csv).drop(columns=['crossed'])

    merged_df = pd.merge(per_df, env_df, on="track_id", how="inner")
    merged_df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Pedestrian stats summary saved to: {output_csv_path}")