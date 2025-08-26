import os
import pandas as pd

def update_video_status(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    for idx, row in df.iterrows():
        video_name = f"{row['city']}_{row['video']}"
        folder_path = os.path.join("analysis_results", video_name)
        target_file = os.path.join(folder_path, "[C10]nearby_count.csv")

        if os.path.isdir(folder_path) and os.path.exists(target_file):
            df.at[idx, "downloaded"] = True
            df.at[idx, "finished"] = True
        else:
            df.at[idx, "downloaded"] = None
            df.at[idx, "finished"] = None

    df.to_csv(output_csv, index=False)
    print(f"Result saved to {output_csv}")
