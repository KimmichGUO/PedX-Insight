import os
import pandas as pd
import subprocess
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--start-city", type=str, default='Toronto', help="Start downloading from this city")
args = parser.parse_args()

df = pd.read_csv("mapping.csv")

start_index = 0
if args.start_city:
    if args.start_city in df['city'].values:
        start_index = df.index[df['city'] == args.start_city][0]
    else:
        print(f"City not found: {args.start_city}, starting from the first city")

for idx, row in df.iloc[start_index:].iterrows():
    city = row['city']
    video_ids = row['videos'].strip("[]").split(",")
    video_ids = [v.strip() for v in video_ids if v.strip()]

    city_dir = os.path.join("videos", city)
    os.makedirs(city_dir, exist_ok=True)

    mapping_records = []
    video_count = 1

    print(f"=== Downloading videos for {city} (4K preferred), total {len(video_ids)} videos ===")
    for vid in video_ids:
        existing_files = glob.glob(os.path.join(city_dir, f"{city}{video_count}.*"))
        if existing_files:
            print(f"⏩ Skipping {vid} (already exists)")
            video_count += 1
            continue

        url = f"https://www.youtube.com/watch?v={vid}"
        new_video_name = f"{city}{video_count}"

        try:
            result = subprocess.run([
                "yt-dlp", "--get-title",
                "-f", "bestvideo[height=2160]/bestvideo",
                "-o", os.path.join(city_dir, f"{new_video_name}.%(ext)s"),
                url
            ], capture_output=True, text=True, check=True)

            original_title = result.stdout.strip()

            subprocess.run([
                "yt-dlp",
                "-f", "bestvideo[height=2160]/bestvideo",
                "-o", os.path.join(city_dir, f"{new_video_name}.%(ext)s"),
                url
            ], check=True)

            mapping_records.append([new_video_name, original_title, url])
            video_count += 1

        except subprocess.CalledProcessError:
            print(f"Failed to download: {url}")

    if mapping_records:
        mapping_df = pd.DataFrame(mapping_records, columns=["New_Name", "Original_Title", "URL"])
        mapping_df.to_csv(os.path.join(city_dir, f"{city}_mapping.csv"), index=False, encoding="utf-8-sig")

print("All downloads completed")
