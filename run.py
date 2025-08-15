import pandas as pd
import subprocess
import os

df = pd.read_csv('mapping_ex_filtered.csv')

video_folder = './videos'
os.makedirs(video_folder, exist_ok=True)

for idx, row in df.iterrows():
    video_id = row['video']
    name = row['name']

    video_name = f"{name}.mp4"
    video_path = os.path.join(video_folder, video_name)
    url = f"https://www.youtube.com/watch?v={video_id}"

    download_cmd = [
        "yt-dlp",
        "-f", "bestvideo[height=2160]/bestvideo",
        "-o", video_path,
        url
    ]
    print(f"Downloading video {video_id} ...")
    try:
        subprocess.run(download_cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Failed to download video {video_id}, skipping.")
        continue

    analysis_cmd = [
        "python",
        "main.py",
        "--mode", "analysis",
        "--source_video_path", "./videos"
    ]
    print(f"Analyzing video {video_id} ...")
    try:
        subprocess.run(analysis_cmd, check=True)
    except subprocess.CalledProcessError:
        print(f"Analysis failed for {video_id}")

    if os.path.exists(video_path):
        os.remove(video_path)
        print(f"Deleted video {video_id}")

print("All videos processed!")
