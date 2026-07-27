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
        localize: bool = False, limit: int = 0, clip_seconds: int = 0):
    """
    Process videos: download, analyze, and delete.

    Args:
        start_row (int): Start processing from which row in the CSV (1-based, not including header). Default = 1.
        start_step (int): Which step to start with.
                          1 = download, 2 = analysis, 3 = deletion. Default = 1.
        csv_file (str): Path to the CSV file containing video info. Default = "mapping_each.csv".
        localize (bool): Also geolocate each video (--mode localize) after analysis and
                         BEFORE deletion (localization needs the video file). Requires the
                         monocular_osm package (pip install -r requirements-localize.txt);
                         failures never block the run.
        limit (int): Stop after processing this many videos (0 = no limit). Counts videos
                         actually processed, not rows skipped as already-finished.
        clip_seconds (int): Download only the first N seconds of each video (0 = whole
                         video, the default). Analysis cost is 3.5-90x realtime and scales
                         with pedestrian density, so an unbounded batch over dense cities
                         can run for weeks; this bounds it. Needs ffmpeg on PATH.
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

    processed = 0
    for i, (idx, row) in enumerate(df.iloc[start_row - 1:].iterrows()):
        if limit and processed >= limit:
            print(f"Reached --limit {limit}; stopping.")
            break
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
                # node runtime lets yt-dlp solve YouTube's JS challenges (node v24 installed);
                # without it extraction is deprecated and bot-checks trigger sooner.
                "--js-runtimes", "node",
                "--user-agent",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "--referer", "https://www.youtube.com/",
                "-f", "bestvideo[height<=720]/bestvideo",
                "-o", video_path,
            ]
            if clip_seconds and clip_seconds > 0:
                # Fetch only the opening window. yt-dlp needs ffmpeg for this, and
                # --force-keyframes-at-cuts makes the cut frame-accurate so the analysis
                # modules do not start on a partial GOP.
                download_cmd += ["--download-sections", f"*0-{int(clip_seconds)}",
                                 "--force-keyframes-at-cuts"]
            download_cmd.append(url)
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
            # Call the wrapper directly (it forwards unknown flags to the OSM tool);
            # main.py's strict argparse can't carry tool flags. The flags below are the
            # proven fast/safe configuration: no splat/aerial refinement, ~0.7s frame
            # spacing for real VO baselines, capped frame budget.
            localize_cmd = [
                sys.executable, os.path.join("modules", "localization", "localize.py"),
                "--source_video_path", video_path,
                "--no-splat", "--no-aerial",
                "--frame-stride", "20", "--max-frames", "600",
                "--vo-segment", "60:900",
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

        processed += 1

    print("All videos processed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos: download, analyze, and delete.")
    parser.add_argument("--start_row", type=int, default=1, help="Row number to start from (1 = first data row)")
    parser.add_argument("--start_step", type=int, default=1, choices=[1, 2, 3],
                        help="Step to start with: 1 = download, 2 = analysis, 3 = deletion")
    parser.add_argument("--csv", type=str, default="mapping_one_each.csv", help="Path to CSV file with video info")
    parser.add_argument("--localize", action="store_true",
                        help="Also geolocate each video (--mode localize) after analysis, before deletion. "
                             "Requires the monocular_osm package (see README / requirements-localize.txt).")
    parser.add_argument("--limit", type=int, default=0,
                        help="Stop after processing this many videos (0 = no limit).")
    parser.add_argument("--clip-seconds", dest="clip_seconds", type=int, default=0,
                        help="Download only the first N seconds of each video (0 = whole video). "
                             "Analysis runs 3.5-90x realtime and scales with pedestrian density, "
                             "so this is how you bound a batch's cost. Requires ffmpeg on PATH.")

    args = parser.parse_args()

    run(start_row=args.start_row, start_step=args.start_step, csv_file=args.csv, localize=args.localize,
        limit=args.limit, clip_seconds=args.clip_seconds)
