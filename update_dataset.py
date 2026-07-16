import os
import pandas as pd

def update_video_status(input_csv, output_csv):
    if not (os.path.exists(input_csv) and os.path.getsize(input_csv) > 0):
        print(f"Mapping CSV missing or empty, nothing to update: {input_csv}")
        return

    df = pd.read_csv(input_csv)

    # Ensure the status columns exist, and force object dtype: an all-empty CSV column is
    # inferred as float64, and pandas 3 removed the silent upcast on setitem — assigning
    # True below would raise on a float64 column. Same guard as run.py.
    for col in ['downloaded', 'finished']:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype('object')

    for idx, row in df.iterrows():
        # Use 'name' (not 'city') to match run.py's f"{name}_{video_id}" folder/file naming;
        # otherwise status detection silently fails whenever name != city.
        video_name = f"{row['name']}_{row['video']}"
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
    if os.path.isdir(videos_dir):
        for file in os.listdir(videos_dir):
            if file.endswith(".mp4.part"):
                file_path = os.path.join(videos_dir, file)
                os.remove(file_path)
                print(f"Deleted unfinished download: {file_path}")

    for idx, row in df.iterrows():
        if str(row.get("finished")).lower() == "true":
            video_name = f"{row['name']}_{row['video']}"
            video_file = os.path.join("videos", f"{video_name}.mp4")
            if os.path.exists(video_file):
                os.remove(video_file)
                print(f"Deleted analyzed video: {video_file}")

    df.to_csv(output_csv, index=False)
    print(f"Result saved to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Refresh downloaded/finished flags in the mapping CSV from analysis_results/ and videos/."
    )
    parser.add_argument("--input_csv", type=str, default="mapping_one_each.csv",
                        help="Path to the mapping CSV to read.")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Where to write the updated CSV (defaults to overwriting input_csv).")
    args = parser.parse_args()

    update_video_status(args.input_csv, args.output_csv or args.input_csv)
