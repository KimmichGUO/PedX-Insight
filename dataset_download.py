import os
import pandas as pd
import subprocess
import glob

df = pd.read_csv("mapping.csv")

for idx, row in df.iterrows():
    city = row['city']
    video_ids = row['videos'].strip("[]").split(",")
    video_ids = [v.strip() for v in video_ids if v.strip()]

    city_dir = os.path.join("videos", city)
    os.makedirs(city_dir, exist_ok=True)

    print(f"=== Downloading videos for {city} (4K preferred), total {len(video_ids)} videos ===")
    for vid in video_ids:
        existing_files = glob.glob(os.path.join(city_dir, f"*{vid}*"))
        if existing_files:
            print(f"⏩ Skipping {vid} (already exists)")
            continue

        url = f"https://www.youtube.com/watch?v={vid}"
        try:
            subprocess.run([
                "yt-dlp",
                "-f", "bestvideo[height=2160]+bestaudio/best[height=2160]",
                "-o", os.path.join(city_dir, f"%(title)s_{vid}.%(ext)s"),
                url
            ], check=True)
        except subprocess.CalledProcessError:
            print(f"❌ Failed to download: {url}")

print("All downloads completed")
