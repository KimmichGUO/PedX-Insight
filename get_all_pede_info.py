import os
import pandas as pd
import ast
from pathlib import Path


def find_city_in_mapping(city, link, mapping_df):
    """
    Find the corresponding city information from mapping.csv
    """
    # First, find all rows matching the city name
    city_matches = mapping_df[mapping_df['city'] == city]

    if len(city_matches) == 0:
        print(f"Warning: City {city} not found in mapping.csv")
        return None
    elif len(city_matches) == 1:
        # Only one match, return directly
        return city_matches.iloc[0]
    else:
        # Multiple matches, need to check using link
        for _, row in city_matches.iterrows():
            try:
                # Parse the 'videos' column
                videos_str = row['videos']
                if pd.isna(videos_str):
                    continue

                # Parse list format manually [item1,item2,item3]
                if isinstance(videos_str, str):
                    if videos_str.startswith('[') and videos_str.endswith(']'):
                        # Remove brackets and split by comma
                        videos_content = videos_str[1:-1]
                        videos_list = [v.strip() for v in videos_content.split(',')]
                    else:
                        # If not list format, assume comma-separated
                        videos_list = [v.strip() for v in videos_str.split(',')]
                else:
                    videos_list = videos_str

                if link in videos_list:
                    return row
            except Exception as e:
                print(f"Error parsing 'videos' column: {e}, value: {videos_str}")
                continue

        print(f"Warning: Multiple matches for city {city}, but no record contains link {link}")
        return city_matches.iloc[0]  # Return the first match as default


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
        city = folder_name[:first_underscore_index]
        link = folder_name[first_underscore_index + 1:]
        city_link = folder_name  # Full folder name as city_link

        print(f"  City: {city}, Link: {link}")

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
                'age', 'phone_using', 'backpack', 'umbrella', 'handbag', 'suitcase',
                'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
                'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
                'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress',
                'weather', 'daytime', 'police_car', 'arrow_board', 'cones', 'accident',
                'crack', 'potholes', 'avg_vehicle_total', 'crossing_sign', 'avg_road_width',
                'crosswalk', 'sidewalk', 'ambulance', 'army vehicle', 'auto rickshaw',
                'bicycle', 'bus', 'car', 'garbagevan', 'human', 'hauler', 'minibus',
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
            'age', 'phone_using', 'backpack', 'umbrella', 'handbag', 'suitcase',
            'short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
            'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt',
            'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress',
            'weather', 'daytime', 'police_car', 'arrow_board', 'cones', 'accident',
            'crack', 'potholes', 'avg_vehicle_total', 'crossing_sign', 'avg_road_width',
            'crosswalk', 'sidewalk', 'ambulance', 'army vehicle', 'auto rickshaw',
            'bicycle', 'bus', 'car', 'garbagevan', 'human', 'hauler', 'minibus',
            'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter',
            'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
        ]

        columns_order = ['city', 'link', 'city_link'] + pedestrian_columns + mapping_columns
        result_df = result_df.reindex(columns=columns_order)

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
