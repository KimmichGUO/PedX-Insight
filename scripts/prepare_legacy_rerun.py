"""Prepare a re-run of the legacy_1hz videos on the current (dense_v2) pipeline.

13 of the 28 analysed videos were processed by the old 1 Hz tracker. Their counts are not
comparable with dense_v2 videos, and they have no measured speeds -- so they sit in the
dataset as a second, incompatible tier. Re-analysing them needs the source videos again
(run.py deletes each video after analysis), i.e. a YouTube download with fresh cookies.

This script does the preparation only; it downloads and analyses nothing:

  1. legacy_rerun.csv         run.py driver for every legacy_1hz video, with the SAME
                              segment (start/end) it was originally analysed on, taken from
                              mapping_one_each.csv; downloaded/finished cleared.
  2. legacy_rerun_review.csv  title / channel / duration per video, from PedX-Crawler's
                              outputs, for an eligibility check before spending GPU time:
                              the crawler files videos under the city it searched for, and
                              some titles show they are not footage of that city.
  3. --archive                moves each legacy folder out of analysis_results/ (to
                              analysis_results_legacy_archive/<date>/) so stale 1 Hz files
                              cannot mix with the new outputs. Without the flag nothing moves.

Then run (after refreshing www.youtube.com_cookies.txt from a logged-in browser):
    python run.py --csv legacy_rerun.csv [--localize]
"""

import argparse
import datetime as dt
import glob
import os
import shutil
import sys

import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CRAWLER_OUTPUTS = os.path.join(os.path.dirname(REPO), "PedX-Crawler", "data", "outputs")


def legacy_videos():
    v = pd.read_csv(os.path.join(REPO, "summary_data", "all_video_info.csv"), encoding="utf-8")
    legacy = v[v["pipeline_version"] == "legacy_1hz"].copy()
    # Folder = <name>_<youtube id>; the id is everything after the FIRST underscore.
    legacy["name"] = legacy["video_name"].str.split("_", n=1).str[0]
    legacy["video"] = legacy["video_name"].str.split("_", n=1).str[1]
    return legacy[["video_name", "name", "video", "city", "duration_seconds"]]


def crawler_metadata():
    frames = []
    for path in glob.glob(os.path.join(CRAWLER_OUTPUTS, "*.csv")):
        try:
            df = pd.read_csv(path, encoding="utf-8")
        except Exception:
            continue
        # Crawler schemas differ by file: discovery-style (video, name=title, channel_name)
        # or older (video_id, title, channel).
        cols = set(df.columns)
        vid = "video_id" if "video_id" in cols else "video" if "video" in cols else None
        title = "title" if "title" in cols else "name" if "name" in cols else None
        if vid is None or title is None:
            continue
        chan = "channel" if "channel" in cols else "channel_name" if "channel_name" in cols else None
        part = pd.DataFrame({"video_id": df[vid], "title": df[title],
                             "channel": df[chan] if chan else None,
                             "source_file": os.path.basename(path)})
        frames.append(part)
    if not frames:
        return pd.DataFrame(columns=["video_id", "title", "channel"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("video_id")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exclude", default="", help="comma-separated names to leave out (e.g. Paris2,NewYork3)")
    ap.add_argument("--archive", action="store_true", help="move the legacy folders out of analysis_results/")
    args = ap.parse_args()
    excluded = {x.strip() for x in args.exclude.split(",") if x.strip()}

    legacy = legacy_videos()
    legacy = legacy[~legacy["name"].isin(excluded)]
    mapping = pd.read_csv(os.path.join(REPO, "mapping_one_each.csv"), encoding="utf-8")
    driver = mapping[mapping["video"].isin(legacy["video"])].copy()
    missing = sorted(set(legacy["video"]) - set(driver["video"]))
    if missing:
        print(f"WARNING: not in mapping_one_each.csv, cannot re-run: {missing}")
    driver["downloaded"] = ""
    driver["finished"] = ""
    out = os.path.join(REPO, "legacy_rerun.csv")
    driver.to_csv(out, index=False)
    total_min = (driver["end_time"] - driver["start_time"]).sum() / 60
    # Runtime estimate from this machine's own record (run.py's [A3] -> all_time_info.csv).
    ratio_txt = ""
    try:
        t = pd.read_csv(os.path.join(REPO, "summary_data", "all_time_info.csv"))
        r = (t["analysis_seconds"] / t["duration_seconds"]).replace([float("inf")], pd.NA).dropna()
        if len(r):
            ratio_txt = (f"; median past analysis time = {r.median():.2f}x video length "
                         f"(n={len(r)}), so ~{total_min * r.median():.0f} min of GPU time")
    except Exception:
        pass
    print(f"Wrote {out}: {len(driver)} videos, {total_min:.0f} min of footage{ratio_txt}")

    meta = crawler_metadata()
    review = legacy.merge(meta, left_on="video", right_on="video_id", how="left").drop(columns="video_id")
    review_path = os.path.join(REPO, "legacy_rerun_review.csv")
    review.to_csv(review_path, index=False, encoding="utf-8")
    print(f"Wrote {review_path} -- check each title actually shows street footage of its city:")
    for r in review.itertuples():
        print(f"  {r.name:<10} {r.city:<10} {str(getattr(r, 'title', '')).encode('ascii', 'replace').decode()[:70]}")

    if args.archive:
        dest_root = os.path.join(REPO, "analysis_results_legacy_archive", dt.date.today().isoformat())
        os.makedirs(dest_root, exist_ok=True)
        for name in legacy["video_name"]:
            src = os.path.join(REPO, "analysis_results", name)
            if os.path.isdir(src):
                shutil.move(src, os.path.join(dest_root, name))
                print(f"  archived {name} -> {dest_root}")
        print("Archived. These videos are absent from the dataset until the re-run completes.")
    else:
        print("Nothing moved (pass --archive right before starting the re-run).")
    print("\nNext: refresh www.youtube.com_cookies.txt, then:\n  python run.py --csv legacy_rerun.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
