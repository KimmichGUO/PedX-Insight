import os
import shutil
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
        else:
            df.at[idx, "downloaded"] = None

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

    analysis_dir = "analysis_results"
    incomplete_dir = "incomplete_results"
    os.makedirs(incomplete_dir, exist_ok=True)

    for folder in os.listdir(analysis_dir):
        folder_path = os.path.join(analysis_dir, folder)
        if os.path.isdir(folder_path):
            c10_file = os.path.join(folder_path, "[C10]nearby_count.csv")
            a2_file = os.path.join(folder_path, "[A2]pedestrian_info.csv")
            if os.path.exists(c10_file) and not os.path.exists(a2_file):
                target_path = os.path.join(incomplete_dir, folder)
                if os.path.exists(target_path):
                    shutil.rmtree(target_path)
                shutil.move(folder_path, incomplete_dir)
                print(f"Moved incomplete result folder: {folder_path} -> {incomplete_dir}")

    df.to_csv(output_csv, index=False)
    print(f"Result saved to {output_csv}")
