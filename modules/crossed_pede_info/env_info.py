import os
import pandas as pd

def merge_env_info(video_path, crossing_csv_path=None, weather_csv_path=None, road_defect_csv_path=None,
                   daytime_csv_path=None, accident_csv_path=None, output_csv_path=None,
                   vehicle_type_csv=None, traffic_sign_csv=None, road_width_csv=None,
                   crosswalk_csv=None, sidewalk_csv_path=None, speed_trend_csv=None,
                   on_lane_csv=None
                   ):
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
    if vehicle_type_csv is None:
        vehicle_type_csv = os.path.join(output_dir, "[V1]vehicle_type.csv")
    if traffic_sign_csv is None:
        traffic_sign_csv = os.path.join(output_dir, "[E3]traffic_sign.csv")
    if road_width_csv is None:
        road_width_csv = os.path.join(output_dir, "[E5]road_width.csv")
    if crosswalk_csv is None:
        crosswalk_csv = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    if sidewalk_csv_path is None:
        sidewalk_csv_path = os.path.join(output_dir, "[E9]sidewalk_detection.csv")
    if speed_trend_csv is None:
        speed_trend_csv = os.path.join(output_dir, "[C2]pedestrian_acc.csv")
    if on_lane_csv is None:
        on_lane_csv = os.path.join(output_dir, "[C8]pedestrian_on_lane.csv")

    df_crossing = pd.read_csv(crossing_csv_path)
    df_crossing = df_crossing[df_crossing['crossed'] == True]

    df_weather = pd.read_csv(weather_csv_path)
    df_daytime = pd.read_csv(daytime_csv_path)
    df_road = pd.read_csv(road_defect_csv_path)
    df_accident = pd.read_csv(accident_csv_path)
    df_ve_type = pd.read_csv(vehicle_type_csv)

    df_sign = pd.read_csv(traffic_sign_csv)
    df_road_width = pd.read_csv(road_width_csv)
    df_crosswalk = pd.read_csv(crosswalk_csv)
    df_sidewalk = pd.read_csv(sidewalk_csv_path)
    df_speed = pd.read_csv(speed_trend_csv)
    df_onlane = pd.read_csv(on_lane_csv)

    merged_rows = []

    for _, row in df_crossing.iterrows():
        track_id = row['track_id']
        start_frame = int(row['started_frame'])
        end_frame = int(row['ended_frame'])

        weather_labels = df_weather[(df_weather['frame_id'] >= start_frame) & (df_weather['frame_id'] <= end_frame)]
        daytime_labels = df_daytime[(df_daytime['frame_id'] >= start_frame) & (df_daytime['frame_id'] <= end_frame)]
        accident_labels = df_accident[(df_accident['frame_id'] >= start_frame) & (df_accident['frame_id'] <= end_frame)]
        road_labels = df_road[(df_road['frame_id'] >= start_frame) & (df_road['frame_id'] <= end_frame)]
        ve_type_labels = df_ve_type[(df_ve_type['frame_id'] >= start_frame) & (df_ve_type['frame_id'] <= end_frame)]

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
            crack = 1 if (long_crack == 1 or alligator == 1 or trans_crack == 1) else 0
        else:
            long_crack = trans_crack = alligator = potholes = -1
            crack = -1

        if not ve_type_labels.empty:
            avg_vehicle_total = ve_type_labels['total'].mean()
            vehicle_presence = {}
            total_frames = len(ve_type_labels)
            for col in ve_type_labels.columns:
                if col not in ['frame_id', 'total']:
                    presence_ratio = (ve_type_labels[col] > 0).sum() / total_frames
                    vehicle_presence[col] = 1 if presence_ratio > 0.1 else 0
        else:
            avg_vehicle_total = 0
            vehicle_presence = {col: -1 for col in df_ve_type.columns if col not in ['frame_id', 'total']}


        special_not_risky_signs = {"w57", "pg", "i1"}
        signs_in_range = df_sign[(df_sign['frame_id'] >= start_frame) & (df_sign['frame_id'] <= end_frame)]
        special_not_risky_flag = 0
        if not signs_in_range.empty:
            for _, srow in signs_in_range.iterrows():
                if any(sign in str(srow['sign_classes_1']) for sign in special_not_risky_signs) \
                   or ("Pedestrian Crossing" in str(srow['sign_classes_2'])):
                    special_not_risky_flag = 1
                    break

        # road width
        roadwidth_in_range = df_road_width[(df_road_width['Frame Index'] >= start_frame) & (df_road_width['Frame Index'] <= end_frame)]
        avg_road_width = roadwidth_in_range['Road Width (m)'].mean() if not roadwidth_in_range.empty else -1

        # crosswalk
        crosswalk_in_range = df_crosswalk[(df_crosswalk['frame_id'] >= start_frame) & (df_crosswalk['frame_id'] <= end_frame)]
        if not crosswalk_in_range.empty:
            ratio = (crosswalk_in_range['crosswalk_detected'] == "Yes").sum() / len(crosswalk_in_range)
            crosswalk_flag = 1 if ratio > 0.2 else 0
        else:
            crosswalk_flag = -1

        # sidewalk
        sidewalk_in_range = df_sidewalk[(df_sidewalk['frame_id'] >= start_frame) & (df_sidewalk['frame_id'] <= end_frame)]
        if not sidewalk_in_range.empty:
            ratio = (sidewalk_in_range['polygons'].notna() & (sidewalk_in_range['polygons'] != "")).sum() / len(sidewalk_in_range)
            sidewalk_flag = 1 if ratio > 0.5 else 0
        else:
            sidewalk_flag = -1

        # speed trend
        speed_in_range = df_speed[df_speed['track_id'] == track_id]
        if not speed_in_range.empty:
            trend_mode = speed_in_range['trend'].mode()[0]
        else:
            trend_mode = "Unknown"

        # on lane
        onlane_in_range = df_onlane[df_onlane['track_id'] == track_id]
        if not onlane_in_range.empty and onlane_in_range['entered_lane'].iloc[0] == True:
            overlap = 0
            for _, lrow in onlane_in_range.iterrows():
                overlap_start = max(start_frame, lrow['start_frame'])
                overlap_end = min(end_frame, lrow['end_frame'])
                if overlap_start <= overlap_end:
                    overlap += (overlap_end - overlap_start + 1)
            total_cross_frames = end_frame - start_frame + 1
            onlane_flag = 1 if (overlap / total_cross_frames) > 0.1 else 0
        else:
            onlane_flag = 0

        merged_row = {
            'track_id': track_id,
            'crossed': True,
            'weather': weather,
            'daytime': daytime,
            'police_car': police_car,
            'arrow_board': arrow_board,
            'cones': cones,
            'accident': accident,
            'crack': crack,
            'potholes': potholes,
            'avg_vehicle_total': avg_vehicle_total,
            'crossing_sign': special_not_risky_flag,
            'avg_road_width': avg_road_width,
            'crosswalk': crosswalk_flag,
            'sidewalk': sidewalk_flag,
            'speed_trend': trend_mode,
            'on_lane': onlane_flag
        }

        merged_row.update(vehicle_presence)
        merged_rows.append(merged_row)

    df_result = pd.DataFrame(merged_rows)
    df_result.to_csv(output_csv_path, index=False)
    print(f"Crossed Pedestrian Environment analysis Result saved to {output_csv_path}")