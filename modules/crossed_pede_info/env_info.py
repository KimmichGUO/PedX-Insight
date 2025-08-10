import os
import pandas as pd

def merge_env_info(video_path, crossing_csv_path=None, weather_csv_path=None, road_defect_csv_path=None, daytime_csv_path=None, accident_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C9]crossing_env_info.csv")
    if crossing_csv_path is None:
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if weather_csv_path is None:
        weather_csv_path = os.path.join(output_dir, "[E1]weather.csv")
    if road_defect_csv_path is None:
        road_defect_csv_path = os.path.join(output_dir, "[E4]road_condition.csv")
    if daytime_csv_path is None:
        daytime_csv_path = os.path.join(output_dir, "[E6]daytime.csv")
    if accident_csv_path is None:
        accident_csv_path = os.path.join(output_dir, "[E8]accident_detection.csv")

    df_crossing = pd.read_csv(crossing_csv_path)
    df_crossing = df_crossing[df_crossing['crossed'] == True]

    df_weather = pd.read_csv(weather_csv_path)
    df_daytime = pd.read_csv(daytime_csv_path)
    df_road = pd.read_csv(road_defect_csv_path)
    df_accident = pd.read_csv(accident_csv_path)

    merged_rows = []

    for _, row in df_crossing.iterrows():
        track_id = row['track_id']
        start_frame = int(row['started_frame'])
        end_frame = int(row['ended_frame'])

        weather_labels = df_weather[(df_weather['frame_id'] >= start_frame) & (df_weather['frame_id'] <= end_frame)]
        daytime_labels = df_daytime[(df_daytime['frame_id'] >= start_frame) & (df_daytime['frame_id'] <= end_frame)]
        accident_labels = df_accident[(df_accident['frame_id'] >= start_frame) & (df_accident['frame_id'] <= end_frame)]
        road_labels = df_road[(df_road['frame_id'] >= start_frame) & (df_road['frame_id'] <= end_frame)]

        weather = weather_labels['weather_label'].mode()[0] if not weather_labels.empty else 'Unknown'
        daytime = daytime_labels['daytime_label'].mode()[0] if not daytime_labels.empty else 'Unknown'

        if not accident_labels.empty:
            police_car = accident_labels['police_car'].mode()[0]
            arrow_board = accident_labels['Arrow Board'].mode()[0]
            cones = accident_labels['cones'].mode()[0]
            accident = accident_labels['accident'].mode()[0]
        else:
            police_car = arrow_board = cones = accident = -1

        if not road_labels.empty:
            long_crack = road_labels['Longitudinal Crack'].mode()[0]
            trans_crack = road_labels['Transverse Crack'].mode()[0]
            alligator = road_labels['Alligator Crack'].mode()[0]
            potholes = road_labels['Potholes'].mode()[0]
            if long_crack == 1 or alligator == 1 or trans_crack == 1:
                crack = 1
            else:
                crack = 0
        else:
            long_crack = trans_crack = alligator = potholes = -1

        merged_rows.append({
            'track_id': track_id,
            'crossed': True,
            'weather': weather,
            'daytime': daytime,
            'police_car': police_car,
            'arrow_board': arrow_board,
            'cones': cones,
            'accident': accident,
            # 'longitudinal_crack': long_crack,
            # 'transverse_crack': trans_crack,
            # 'alligator_crack': alligator,
            'crack': crack,
            'potholes': potholes
        })

    df_result = pd.DataFrame(merged_rows)
    df_result.to_csv(output_csv_path, index=False)
    print(f"Crossed Pedestrian Environment analysis Result saved to {output_csv_path}")
