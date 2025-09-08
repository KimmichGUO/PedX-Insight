import os
import pandas as pd
import ast
from pathlib import Path


def find_city_in_mapping(city, link, mapping_df):
    """
    Find the corresponding city information in mapping.csv
    """
    # First find all rows that match the city name
    city_matches = mapping_df[mapping_df['city'] == city]

    if len(city_matches) == 0:
        print(f"Warning: City {city} not found")
        return None
    elif len(city_matches) == 1:
        # Only one match, return it
        return city_matches.iloc[0]
    else:
        # Multiple matches, need to check using link
        for _, row in city_matches.iterrows():
            try:
                videos_str = row['videos']
                if pd.isna(videos_str):
                    continue

                if isinstance(videos_str, str):
                    if videos_str.startswith('[') and videos_str.endswith(']'):
                        videos_content = videos_str[1:-1]
                        videos_list = [v.strip() for v in videos_content.split(',')]
                    else:
                        videos_list = [v.strip() for v in videos_str.split(',')]
                else:
                    videos_list = videos_str

                if link in videos_list:
                    return row
            except Exception as e:
                print(f"Error parsing videos column: {e}, value: {videos_str}")
                continue

        print(f"Warning: City {city} has multiple matches but no record containing link {link} was found")
        return city_matches.iloc[0]  # Return the first match by default


def parse_video_info_csv(file_path):
    """
    Parse video_info.csv and return a dictionary of all metrics
    """
    data_dict = {}
    try:
        df = pd.read_csv(file_path)
        for _, row in df.iterrows():
            data_dict[row['metric']] = row['value']
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    return data_dict


def get_top_vehicles(file_path):
    """
    Read [V6]vehicle_count.csv and return the top three vehicle types
    """
    try:
        df = pd.read_csv(file_path)

        # Filter out "Total" row (case insensitive)
        df_filtered = df[df['Vehicle_Type'].str.lower() != 'total']

        # Sort by count descending
        df_sorted = df_filtered.sort_values('Count', ascending=False)

        top_vehicles = df_sorted['Vehicle_Type'].head(3).tolist()

        while len(top_vehicles) < 3:
            top_vehicles.append(None)

        return top_vehicles[0], top_vehicles[1], top_vehicles[2]

    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None, None, None


def main():
    # Define paths
    final_results_path = Path('./analysis_results')
    mapping_csv_path = Path('./mapping.csv')
    crossing_time_csv_path = Path('./crossing_time.csv')
    crossing_speed_csv_path = Path('./crossing_speed.csv')
    output_csv_path = Path('./summary_data/all_video_info.csv')

    if not final_results_path.exists():
        print("Error: ./analysis_results/ folder does not exist")
        return

    if not mapping_csv_path.exists():
        print("Error: mapping.csv does not exist")
        return

    # Load mapping.csv
    try:
        mapping_df = pd.read_csv(mapping_csv_path)
        print(f"Successfully loaded mapping.csv with {len(mapping_df)} records")
    except Exception as e:
        print(f"Error reading mapping.csv: {e}")
        return

    # Load crossing_time.csv
    crossing_time_df = None
    if crossing_time_csv_path.exists():
        try:
            crossing_time_df = pd.read_csv(crossing_time_csv_path)
            print(f"Successfully loaded crossing_time.csv with {len(crossing_time_df)} records")
        except Exception as e:
            print(f"Error reading crossing_time.csv: {e}")
    else:
        print("Warning: crossing_time.csv does not exist")

    # Load crossing_speed.csv
    crossing_speed_df = None
    if crossing_speed_csv_path.exists():
        try:
            crossing_speed_df = pd.read_csv(crossing_speed_csv_path)
            print(f"Successfully loaded crossing_speed.csv with {len(crossing_speed_df)} records")
        except Exception as e:
            print(f"Error reading crossing_speed.csv: {e}")
    else:
        print("Warning: crossing_speed.csv does not exist")

    # Define required columns
    video_info_columns = [
        'video_name', 'duration_seconds', 'total_frames', 'total_pedestrians',
        'total_crossed_pedestrians', 'average_age', 'phone_usage_ratio',
        'risky_crossing_ratio', 'run_red_light_ratio', 'crosswalk_usage_ratio',
        'traffic_signs_ratio', 'total_vehicles', 'top3_vehicles', 'main_weather',
        'sidewalk_prob', 'crosswalk_prob', 'traffic_light_prob', 'avg_road_width',
        'Crack_prob', 'Potholes_prob', 'police_car_prob', 'Arrow Board_prob',
        'cones_prob', 'accident_prob'
    ]

    mapping_columns = [
        'state', 'country', 'iso3', 'continent', 'lat', 'lon', 'gmp',
        'population_city', 'population_country', 'traffic_mortality',
        'literacy_rate', 'avg_height', 'med_age', 'gini'
    ]

    aggregated_data = []

    for folder_path in final_results_path.iterdir():
        if not folder_path.is_dir():
            continue

        folder_name = folder_path.name
        print(f"Processing folder: {folder_name}")

        if '_' not in folder_name:
            print(f"Warning: Folder name {folder_name} does not contain an underscore, skipped")
            continue

        first_underscore_index = folder_name.index('_')
        city = folder_name[:first_underscore_index]
        link = folder_name[first_underscore_index + 1:]

        print(f"  City: {city}, Link: {link}")

        video_info_path = folder_path / '[A1]video_info.csv'
        if not video_info_path.exists():
            print(f"  Warning: {video_info_path} does not exist, skipped")
            continue

        video_data = parse_video_info_csv(video_info_path)
        if video_data is None:
            continue

        vehicle_count_path = folder_path / '[V6]vehicle_count.csv'
        top_vehicle1, top_vehicle2, top_vehicle3 = None, None, None
        if vehicle_count_path.exists():
            top_vehicle1, top_vehicle2, top_vehicle3 = get_top_vehicles(vehicle_count_path)
        else:
            print(f"  Warning: {vehicle_count_path} does not exist")

        city_info = find_city_in_mapping(city, link, mapping_df)

        crossing_time = None
        if crossing_time_df is not None:
            time_match = crossing_time_df[crossing_time_df['city'] == city]
            if len(time_match) > 0:
                crossing_time = time_match.iloc[0]['time']

        crossing_speed = None
        if crossing_speed_df is not None:
            speed_match = crossing_speed_df[crossing_speed_df['city'] == city]
            if len(speed_match) > 0:
                crossing_speed = speed_match.iloc[0]['speed']

        row_data = {'city': city, 'link': link}

        for col in video_info_columns:
            row_data[col] = video_data.get(col, None)

        row_data['crossing_time'] = crossing_time
        row_data['crossing_speed'] = crossing_speed

        if city_info is not None:
            for col in mapping_columns:
                row_data[col] = city_info.get(col, None)
        else:
            for col in mapping_columns:
                row_data[col] = None

        aggregated_data.append(row_data)
        print(f"  Successfully processed folder {folder_name}")

    if aggregated_data:
        result_df = pd.DataFrame(aggregated_data)

        columns_order = ['city', 'link'] + video_info_columns + [
            'crossing_time', 'crossing_speed'
        ] + mapping_columns
        result_df = result_df.reindex(columns=columns_order)

        result_df.to_csv(output_csv_path, index=False, encoding='latin-1')
        print(f"\nSuccessfully processed {len(aggregated_data)} folders")
        print(f"Results saved to {output_csv_path}")
        print(f"Output file contains {len(result_df)} rows and {len(result_df.columns)} columns")
    else:
        print("No valid data found")


if __name__ == "__main__":
    main()
