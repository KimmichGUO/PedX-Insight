import pandas as pd
import subprocess
import os
import argparse

def run(start_row: int = 1, start_step: int = 1, csv_file: str = "mapping_ex_filtered.csv"):
    """
    Process videos: download, analyze, and delete.

    Args:
        start_row (int): Start processing from which row in the CSV (1-based, not including header). Default = 1.
        start_step (int): Which step to start with.
                          1 = download, 2 = analysis, 3 = deletion. Default = 1.
        csv_file (str): Path to the CSV file containing video info. Default = "mapping_ex_filtered.csv".
    """
    df = pd.read_csv(csv_file)

    video_folder = './videos'
    os.makedirs(video_folder, exist_ok=True)

    for idx, row in df.iloc[start_row - 1:].iterrows():
        video_id = row['video']
        name = row['name']

        video_name = f"{name}.mp4"
        video_path = os.path.join(video_folder, video_name)
        url = f"https://www.youtube.com/watch?v={video_id}"

        # For subsequent rows, always start from step 1.
        current_start_step = start_step if idx == (start_row - 1) else 1

        # Step 1: Download video
        if current_start_step <= 1:
            download_cmd = [
                "yt-dlp",
                "-f", "bestvideo[height=2160]/bestvideo",
                "-o", video_path,
                url
            ]
            print(f"Downloading video {video_name} ...")
            try:
                subprocess.run(download_cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to download video {video_name}, skipping.")
                continue

        # Step 2: Run analysis
        if current_start_step <= 2:
            analysis_cmd = [
                "python",
                "main.py",
                "--mode", "analysis",
                "--source_video_path", "./videos"
            ]
            print(f"Analyzing video {video_name} ...")
            try:
                subprocess.run(analysis_cmd, check=True)
            except subprocess.CalledProcessError:
                print(f"Analysis failed for {video_id}")

        # Step 3: Delete video
        if current_start_step <= 3:
            if os.path.exists(video_path):
                os.remove(video_path)
                print(f"Deleted video {video_name}")

    print("All videos processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos: download, analyze, and delete.")
    parser.add_argument("--start_row", type=int, default=1, help="Row number to start from (1 = first data row)")
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3],
                        help="Step to start with: 1 = download, 2 = analysis, 3 = deletion")
    parser.add_argument("--csv", type=str, default="mapping_ex_filtered.csv", help="Path to CSV file with video info")

    args = parser.parse_args()

    run(start_row=args.start_row, start_step=args.start_step, csv_file=args.csv)
