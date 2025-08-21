import sys
import pandas as pd
import subprocess
import os
import argparse


def run(start_row: int = 1, start_step: int = 1, csv_file: str = "mapping_one_each.csv"):
    """
    Process videos: download, analyze, and delete.

    Args:
        start_row (int): Start processing from which row in the CSV (1-based, not including header). Default = 1.
        start_step (int): Which step to start with.
                          1 = download, 2 = analysis, 3 = deletion. Default = 1.
        csv_file (str): Path to the CSV file containing video info. Default = "mapping_each.csv".
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    # Ensure columns exist
    for col in ['finished', 'downloaded']:
        if col not in df.columns:
            print(f"Column '{col}' not found, creating it")
            df[col] = None

    video_folder = './videos'
    os.makedirs(video_folder, exist_ok=True)

    for idx, row in df.iloc[start_row - 1:].iterrows():
        video_id = row['video']
        name = row['name']
        is_finished = pd.notna(row['finished']) and str(row['finished']).upper() == 'TRUE'

        # Skip if already analyzed
        if is_finished:
            print(f"Video {name} already analyzed, skipping...")
            continue

        video_name = f"{name}_{video_id}.mp4"
        video_path = os.path.join(video_folder, video_name)
        url = f"https://www.youtube.com/watch?v={video_id}"

        # Check download status
        is_downloaded = pd.notna(row['downloaded']) and str(row['downloaded']).upper() == 'TRUE'

        # Step 1: Download if not downloaded
        if not is_downloaded:
            download_cmd = [
                "yt-dlp",
                "--cookies", "www.youtube.com_cookies.txt",
                "--user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--referer", "https://www.youtube.com/",
                "-f", "bestvideo[height<=720]/bestvideo",
                "-o", video_path,
                url
            ]
            print(f"Downloading video {video_name} ...")
            try:
                subprocess.run(download_cmd, check=True)
                print(f"Downloaded {video_name}")
                df.loc[idx, 'downloaded'] = 'TRUE'
                df.to_csv(csv_file, index=False)
            except subprocess.CalledProcessError:
                print(f"Failed to download video {video_name}, skipping.")
                continue

        # Step 2: Run analysis
        analysis_cmd = [
            sys.executable,
            "main.py",
            "--mode", "single",
            "--source_video_path", f"./videos/{video_name                                                                                                                                                                            }"
        ]
        print(f"Analyzing video {video_name} ...")
        try:
            subprocess.run(analysis_cmd, check=True)
            print(f"✓ Analysis completed for {video_name}")
            df.loc[idx, 'finished'] = 'TRUE'
            df.to_csv(csv_file, index=False)
        except subprocess.CalledProcessError:
            print(f"✗ Analysis failed for {video_name}")
            continue

        # Step 3: Delete video
        if os.path.exists(video_path):
            os.remove(video_path)
            print(f"✓ Deleted video {video_name}")

    print("All videos processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos: download, analyze, and delete.")
    parser.add_argument("--start_row", type=int, default=1, help="Row number to start from (1 = first data row)")
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3],
                        help="Step to start with: 1 = download, 2 = analysis, 3 = deletion")
    parser.add_argument("--csv", type=str, default="mapping_one_each.csv", help="Path to CSV file with video info")

    args = parser.parse_args()

    run(start_row=args.start_row, start_step=args.start_step, csv_file=args.csv)
