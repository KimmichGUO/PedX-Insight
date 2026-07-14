import sys
import time
import pandas as pd
import subprocess
import os
import argparse


def _write_analysis_time(video_file_name, analysis_seconds):
    """Record per-video analysis wall time as [A3]analysis_time.csv.

    This is the producer for summary_data/all_time_info.csv (aggregated by
    get_all_video_info.py), keyed by the YouTube id so the Visualizer can join
    on videos.link instead of the old fragile duration-equality join.
    """
    folder = os.path.splitext(video_file_name)[0]
    out_dir = os.path.join("analysis_results", folder)
    os.makedirs(out_dir, exist_ok=True)
    # <name>_<video_id> with name underscore-free -> id = part after FIRST underscore
    link = folder.split("_", 1)[1] if "_" in folder else folder

    duration_seconds = None
    video_info_csv = os.path.join(out_dir, "[A1]video_info.csv")
    if os.path.exists(video_info_csv):
        try:
            vi = pd.read_csv(video_info_csv)
            match = vi.loc[vi["metric"] == "duration_seconds", "value"]
            if not match.empty:
                duration_seconds = match.iloc[0]
        except Exception as e:
            print(f"[warn] could not read duration from {video_info_csv}: {e}")

    pd.DataFrame([{
        "link": link,
        "video_name": folder,
        "duration_seconds": duration_seconds,
        "analysis_seconds": round(analysis_seconds, 1),
    }]).to_csv(os.path.join(out_dir, "[A3]analysis_time.csv"), index=False)


def run(start_row: int = 1, start_step: int = 1, csv_file: str = "mapping_one_each.csv",
        localize: bool = False):
    """
    Process videos: download, analyze, and delete.

    Args:
        start_row (int): Start processing from which row in the CSV (1-based, not including header). Default = 1.
        start_step (int): Which step to start with.
                          1 = download, 2 = analysis, 3 = deletion. Default = 1.
        csv_file (str): Path to the CSV file containing video info. Default = "mapping_each.csv".
        localize (bool): Also geolocate each video (--mode localize) after analysis and
                         BEFORE deletion (localization needs the video file). Requires the
                         Monocular-OSM-Localization environment; failures never block the run.
    """
    # Read CSV
    df = pd.read_csv(csv_file)

    # Ensure columns exist, and force object dtype: an all-empty CSV column is inferred
    # as float64, and pandas 3 removed the silent upcast on setitem — assigning the
    # 'TRUE' status strings below would raise LossySetitemError.
    for col in ['finished', 'downloaded']:
        if col not in df.columns:
            print(f"Column '{col}' not found, creating it")
            df[col] = None
        df[col] = df[col].astype('object')

    video_folder = './videos'
    os.makedirs(video_folder, exist_ok=True)

    for i, (idx, row) in enumerate(df.iloc[start_row - 1:].iterrows()):
        # start_step applies to the first processed row only (the resume point): 1=download,
        # 2=analysis, 3=deletion. Subsequent rows always run the full pipeline.
        effective_start_step = start_step if i == 0 else 1
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
        if effective_start_step <= 1 and not is_downloaded:
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
        if effective_start_step <= 2:
            analysis_cmd = [
                sys.executable,
                "main.py",
                "--mode", "single_all",
                "--source_video_path", f"./videos/{video_name}"
            ]
            print(f"Analyzing video {video_name} ...")
            try:
                analysis_started = time.time()
                subprocess.run(analysis_cmd, check=True)
                _write_analysis_time(video_name, time.time() - analysis_started)
                print(f"[OK] Analysis completed for {video_name}")
                df.loc[idx, 'finished'] = 'TRUE'
                df.to_csv(csv_file, index=False)
            except subprocess.CalledProcessError:
                print(f"[FAIL] Analysis failed for {video_name}")
                continue

        # Step 2.5: Geolocate (optional) — must run BEFORE deletion, needs the video file.
        if localize and effective_start_step <= 2:
            localize_cmd = [
                sys.executable, "main.py",
                "--mode", "localize",
                "--source_video_path", video_path,
            ]
            print(f"Localizing video {video_name} ...")
            # check=False: a failed/unconfigured localization must never block the pipeline;
            # localize writes its own status row into [L1]localization.csv either way.
            result = subprocess.run(localize_cmd, check=False)
            print(f"[{'OK' if result.returncode == 0 else 'WARN'}] Localization "
                  f"{'completed' if result.returncode == 0 else 'failed (continuing)'} for {video_name}")

        # Step 3: Delete video
        if effective_start_step <= 3 and os.path.exists(video_path):
            os.remove(video_path)
            print(f"[OK] Deleted video {video_name}")

    print("All videos processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos: download, analyze, and delete.")
    parser.add_argument("--start_row", type=int, default=1, help="Row number to start from (1 = first data row)")
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3],
                        help="Step to start with: 1 = download, 2 = analysis, 3 = deletion")
    parser.add_argument("--csv", type=str, default="mapping_one_each.csv", help="Path to CSV file with video info")
    parser.add_argument("--localize", action="store_true",
                        help="Also geolocate each video (--mode localize) after analysis, before deletion. "
                             "Requires the Monocular-OSM-Localization environment (see README).")

    args = parser.parse_args()

    run(start_row=args.start_row, start_step=args.start_step, csv_file=args.csv, localize=args.localize)
