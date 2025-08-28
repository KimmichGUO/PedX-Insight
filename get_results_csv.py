import os
import shutil
from pathlib import Path


def copy_csv_files():
    source_dir = Path('./analysis_results')
    target_dir = Path('./final_results')

    if not source_dir.exists():
        print(f"Source directory {source_dir} does not exist!")
        return

    target_dir.mkdir(exist_ok=True)

    for folder_path in source_dir.iterdir():
        if folder_path.is_dir():
            folder_name = folder_path.name

            target_folder = target_dir / folder_name
            target_folder.mkdir(exist_ok=True)

            csv_files = list(folder_path.glob('*.csv'))

            if csv_files:
                print(f"Processing folder: {folder_name}")

                for csv_file in csv_files:
                    target_file = target_folder / csv_file.name
                    shutil.copy2(csv_file, target_file)
                    print(f"  Copied: {csv_file.name} -> {target_file}")

            else:
                print(f"No CSV files found in folder: {folder_name}")


if __name__ == "__main__":
    print("Starting CSV file copying...")
    copy_csv_files()
    print("Copying completed!")