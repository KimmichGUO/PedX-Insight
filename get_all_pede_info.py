import os
import re
import pandas as pd
import ast
from pathlib import Path
from modules.completeness import missing_outputs
from modules.mapping_utils import clean_mapping, find_city_in_mapping



def main():
    # Define paths
    final_results_path = Path('./analysis_results')
    mapping_csv_path = Path('./mapping.csv')
    output_csv_path = Path('./summary_data/all_pedestrian_info.csv')

    # Check if paths exist
    if not final_results_path.exists():
        print("Error: ./analysis_results/ folder does not exist")
        return

    if not mapping_csv_path.exists():
        print("Error: mapping.csv file does not exist")
        return

    # Read mapping.csv
    try:
        mapping_df = pd.read_csv(mapping_csv_path)
        mapping_df = clean_mapping(mapping_df)  # zeros->NA, unanimous country backfill
        print(f"Successfully read mapping.csv with {len(mapping_df)} records")
    except Exception as e:
        print(f"Error reading mapping.csv: {e}")
        return

    # Define columns from mapping.csv
    mapping_columns = [
        'state', 'country', 'iso3', 'continent', 'lat', 'lon', 'gmp',
        'population_city', 'population_country', 'traffic_mortality',
        'literacy_rate', 'avg_height', 'med_age', 'gini'
    ]

    # List to collect all pedestrian data
    all_pedestrian_data = []

    # Iterate through all folders under analysis_results
    for folder_path in final_results_path.iterdir():
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name
        print(f"Processing folder: {folder_name}")

        # Parse city name and link
        if '_' not in folder_name:
            print(f"Warning: Folder name {folder_name} does not contain an underscore, skipped")
            continue

        first_underscore_index = folder_name.index('_')
        # Folder prefix is the underscore-free name slug (e.g. 'London1' = city + running
        # index from the crawler bridge); strip the numeric suffix to recover the city.
        city = re.sub(r'\d+$', '', folder_name[:first_underscore_index])
        link = folder_name[first_underscore_index + 1:]
        city_link = folder_name  # Full folder name as city_link

        print(f"  City: {city}, Link: {link}")

        # Same completeness gate get_all_video_info.py applies. Without it, a folder from a
        # run that died mid-pipeline can still contribute person-level rows -- inferred age
        # and gender for a city that has no behavioural data -- as soon as anything
        # regenerates its [A2]. Luanda_z3Gx2hp3Vo8 is excluded today only by the accident of
        # an empty [A2]; one re-run of the crossing modes would let it through.
        missing = missing_outputs(folder_path)
        if missing:
            print(f"  SKIPPED (incomplete analysis): {folder_name} is missing "
                  f"{len(missing)} required output(s): {', '.join(missing)}")
            continue

        # Read [A2]pedestrian_info.csv
        pedestrian_info_path = folder_path / '[A2]pedestrian_info.csv'
        if not pedestrian_info_path.exists():
            print(f"  Warning: {pedestrian_info_path} does not exist, skipped")
            continue

        try:
            pedestrian_df = pd.read_csv(pedestrian_info_path)
            print(f"  Read {len(pedestrian_df)} pedestrian records")
        except Exception as e:
            print(f"  Error reading {pedestrian_info_path}: {e}")
            continue

        # Find corresponding city info from mapping.csv
        city_info = find_city_in_mapping(city, link, mapping_df)
        if city_info is not None:
            city = city_info['city']  # canonical name (NewYork -> New York)

        # Create records for each pedestrian
        for _, ped_row in pedestrian_df.iterrows():
            # Build data row
            row_data = {
                'city': city,
                'link': link,
                'city_link': city_link
            }

            # Add pedestrian information columns
            pedestrian_columns = [
                'track_id', 'crossed', 'nearby_count_beginning', 'nearby_count_whole',
                'risky_crossing', 'run_red_light', 'crosswalk_use_or_not', 'gender',
                'age', 'phone_using', 'walking_speed_mps', 'crossing_speed_mps', 'decision_delay_s',
                'backpack', 'umbrella', 'handbag', 'suitcase',
                'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
                'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
                'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress',
                'weather', 'daytime', 'police_car', 'arrow_board', 'cones', 'accident',
                'crack', 'potholes', 'avg_vehicle_total', 'crossing_sign', 'avg_road_width',
                # NOTE: the real [A2]/[V6] class is the single name 'human hauler'
                # (with a space); the legacy 'human'/'hauler' entries never matched a
                # real column and stay only so existing consumers of
                # all_pedestrian_info.csv (statistics_with_pdf_save.py vehicle_cols)
                # keep finding the columns they index. They remain empty.
                'crosswalk', 'sidewalk', 'ambulance', 'army vehicle', 'auto rickshaw',
                'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 'human',
                'hauler', 'minibus',
                'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter',
                'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
            ]

            for col in pedestrian_columns:
                row_data[col] = ped_row.get(col, None)

            # Add mapping info
            if city_info is not None:
                for col in mapping_columns:
                    row_data[col] = city_info.get(col, None)
            else:
                # If city info not found, fill with None
                for col in mapping_columns:
                    row_data[col] = None

            all_pedestrian_data.append(row_data)

        print(f"  Finished processing folder {folder_name}, added {len(pedestrian_df)} records")

    # Convert to DataFrame and save
    if all_pedestrian_data:
        result_df = pd.DataFrame(all_pedestrian_data)

        # Define column order
        pedestrian_columns = [
            'track_id', 'crossed', 'nearby_count_beginning', 'nearby_count_whole',
            'risky_crossing', 'run_red_light', 'crosswalk_use_or_not', 'gender',
            'age', 'phone_using', 'walking_speed_mps', 'crossing_speed_mps', 'decision_delay_s',
            'backpack', 'umbrella', 'handbag', 'suitcase',
            'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
            'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
            'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress',
            'weather', 'daytime', 'police_car', 'arrow_board', 'cones', 'accident',
            'crack', 'potholes', 'avg_vehicle_total', 'crossing_sign', 'avg_road_width',
            # 'human hauler' is the real [A2] column; 'human'/'hauler' are legacy
            # always-empty columns kept for downstream consumers (see note above).
            'crosswalk', 'sidewalk', 'ambulance', 'army vehicle', 'auto rickshaw',
            'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 'human',
            'hauler', 'minibus',
            'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter',
            'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
        ]

        columns_order = ['city', 'link', 'city_link'] + pedestrian_columns + mapping_columns
        result_df = result_df.reindex(columns=columns_order)

        # Keep the 0/1 flag columns INTEGRAL across the CSV boundary. These columns are
        # nullable now (an undetermined observation is NA, not 0), and pandas widens an
        # int column to float64 as soon as one NA appears -- so to_csv would write '1.0'
        # instead of '1'. The Visualizer's importer parses these with a strict
        # `value === '1'` boolean check, so on 2026-09-07 that widening silently published
        # 156 risky pedestrians as NOT risky. Int64 keeps the tokens 1 / 0 / '' (empty).
        for _flag in ('risky_crossing', 'run_red_light', 'crosswalk_use_or_not', 'phone_using'):
            if _flag in result_df.columns:
                result_df[_flag] = pd.to_numeric(result_df[_flag], errors='coerce').astype('Int64')

        # Ensure the output directory exists (./summary_data is not tracked in the repo).
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        # Save to CSV
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"\nSuccessfully processed {len(set([row['city_link'] for row in all_pedestrian_data]))} folders")
        print(f"Total collected {len(all_pedestrian_data)} pedestrian records")
        print(f"Results saved to {output_csv_path}")
        print(f"Output file contains {len(result_df)} rows and {len(result_df.columns)} columns")
    else:
        print("No valid data found")


if __name__ == "__main__":
    main()
