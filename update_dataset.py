import os
import pandas as pd

def update_video_status(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    for idx, row in df.iterrows():
        video_name = f"{row['city']}_{row['video']}"
        folder_path = os.path.join("analysis_results", video_name)
        target_file = os.path.join(folder_path, "[C10]nearby_count.csv")
        video_file = os.path.join("videos", f"{video_name}.mp4")

        if os.path.isdir(folder_path) and os.path.exists(target_file):
            df.at[idx, "downloaded"] = True
            df.at[idx, "finished"] = True
        else:
            df.at[idx, "downloaded"] = None
            df.at[idx, "finished"] = None

        if os.path.exists(video_file):
            df.at[idx, "downloaded"] = True


    videos_dir = "videos"
    for file in os.listdir(videos_dir):
        if file.endswith(".mp4.part"):
            file_path = os.path.join(videos_dir, file)
            os.remove(file_path)
            print(f"Deleted unfinished download: {file_path}")

    for idx, row in df.iterrows():
        if str(row.get("finished")).lower() == "true":
            video_name = f"{row['city']}_{row['video']}"
            video_file = os.path.join("videos", f"{video_name}.mp4")
            if os.path.exists(video_file):
                os.remove(video_file)
                print(f"Deleted analyzed video: {video_file}")

    df.to_csv(output_csv, index=False)
    print(f"Result saved to {output_csv}")
