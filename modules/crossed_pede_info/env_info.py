import os
import pandas as pd

def merge_env_info(video_path,
                   crossing_csv_path=None, weather_csv_path=None, road_defect_csv_path=None,
                   daytime_csv_path=None, accident_csv_path=None, output_csv_path=None,
                   vehicle_type_csv=None, traffic_sign_csv=None, road_width_csv=None,
                   crosswalk_csv=None, sidewalk_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C9]crossing_env_info.csv")

    crossing_csv_path = crossing_csv_path or os.path.join(output_dir, "[C3]crossing_judge.csv")
    weather_csv_path = weather_csv_path or os.path.join(output_dir, "[E1]weather.csv")
    road_defect_csv_path = road_defect_csv_path or os.path.join(output_dir, "[E4]road_condition.csv")
    daytime_csv_path = daytime_csv_path or os.path.join(output_dir, "[E6]daytime.csv")
    accident_csv_path = accident_csv_path or os.path.join(output_dir, "[E8]accident_detection.csv")
    vehicle_type_csv = vehicle_type_csv or os.path.join(output_dir, "[V1]vehicle_type.csv")
    traffic_sign_csv = traffic_sign_csv or os.path.join(output_dir, "[E3]traffic_sign.csv")
    road_width_csv = road_width_csv or os.path.join(output_dir, "[E5]road_width.csv")
    crosswalk_csv = crosswalk_csv or os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    sidewalk_csv_path = sidewalk_csv_path or os.path.join(output_dir, "[E9]sidewalk_detection.csv")

    def safe_read_csv(path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df if not df.empty else pd.DataFrame()
        return pd.DataFrame()

    df_crossing = safe_read_csv(crossing_csv_path)

    if df_crossing.empty:
        columns = [
            'track_id', 'crossed', 'weather', 'daytime', 'police_car', 'arrow_board', 'cones', 'accident',
            'crack', 'potholes', 'avg_vehicle_total', 'crossing_sign', 'avg_road_width', 'crosswalk', 'sidewalk'
        ]
        if os.path.exists(vehicle_type_csv):
            df_ve_type = safe_read_csv(vehicle_type_csv)
            vehicle_cols = [col for col in df_ve_type.columns if col not in ['frame_id', 'total']]
            columns.extend(vehicle_cols)
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
        print(f"Crossing CSV empty or missing. Created empty CSV at: {output_csv_path}")
        return

    df_weather = safe_read_csv(weather_csv_path)
    df_daytime = safe_read_csv(daytime_csv_path)
    df_road = safe_read_csv(road_defect_csv_path)
    df_accident = safe_read_csv(accident_csv_path)
    df_ve_type = safe_read_csv(vehicle_type_csv)
    df_sign = safe_read_csv(traffic_sign_csv)
    df_road_width = safe_read_csv(road_width_csv)
    df_crosswalk = safe_read_csv(crosswalk_csv)
    df_sidewalk = safe_read_csv(sidewalk_csv_path)

    merged_rows = []

    for _, row in df_crossing[df_crossing['crossed'] == True].iterrows():
        track_id = row['track_id']
        start_frame = int(row.get('started_frame', 0))
        end_frame = int(row.get('ended_frame', start_frame))

        weather_labels = df_weather[(df_weather.get('frame_id', -1) >= start_frame) &
                                    (df_weather.get('frame_id', -1) <= end_frame)]
        weather = weather_labels['weather_label'].mode()[0] if not weather_labels.empty else 'Unknown'

        daytime_labels = df_daytime[(df_daytime.get('frame_id', -1) >= start_frame) &
                                    (df_daytime.get('frame_id', -1) <= end_frame)]
        daytime = daytime_labels['daytime_label'].mode()[0] if not daytime_labels.empty else 'Unknown'

        accident_labels = df_accident[(df_accident.get('frame_id', -1) >= start_frame) &
                                      (df_accident.get('frame_id', -1) <= end_frame)]
        if not accident_labels.empty:
            police_car = accident_labels.get('police_car', pd.Series([-1])).mode()[0]
            arrow_board = accident_labels.get('Arrow Board', pd.Series([-1])).mode()[0]
            cones = accident_labels.get('cones', pd.Series([-1])).mode()[0]
            accident = accident_labels.get('accident', pd.Series([-1])).mode()[0]
        else:
            police_car = arrow_board = cones = accident = -1

        road_labels = df_road[(df_road.get('frame_id', -1) >= start_frame) &
                              (df_road.get('frame_id', -1) <= end_frame)]
        if not road_labels.empty:
            long_crack = road_labels.get('Longitudinal Crack', pd.Series([-1])).mode()[0]
            trans_crack = road_labels.get('Transverse Crack', pd.Series([-1])).mode()[0]
            alligator = road_labels.get('Alligator Crack', pd.Series([-1])).mode()[0]
            potholes = road_labels.get('Potholes', pd.Series([-1])).mode()[0]
            crack = 1 if (long_crack == 1 or alligator == 1 or trans_crack == 1) else 0
        else:
            long_crack = trans_crack = alligator = potholes = -1
            crack = -1

        ve_type_labels = df_ve_type[(df_ve_type.get('frame_id', -1) >= start_frame) &
                                    (df_ve_type.get('frame_id', -1) <= end_frame)]
        if not ve_type_labels.empty:
            avg_vehicle_total = ve_type_labels.get('total', pd.Series([0])).mean()
            total_frames = len(ve_type_labels)
            vehicle_presence = {}
            for col in ve_type_labels.columns:
                if col not in ['frame_id', 'total']:
                    presence_ratio = (ve_type_labels[col] > 0).sum() / total_frames
                    vehicle_presence[col] = 1 if presence_ratio > 0.1 else 0
        else:
            avg_vehicle_total = 0
            vehicle_presence = {col: -1 for col in df_ve_type.columns if col not in ['frame_id', 'total']} if not df_ve_type.empty else {}

        special_not_risky_signs = {"w57", "pg", "i1"}
        signs_in_range = df_sign[(df_sign.get('frame_id', -1) >= start_frame) &
                                 (df_sign.get('frame_id', -1) <= end_frame)]
        special_not_risky_flag = 0
        if not signs_in_range.empty:
            for _, srow in signs_in_range.iterrows():
                if any(sign in str(srow.get('sign_classes_1', '')) for sign in special_not_risky_signs) \
                   or ("Pedestrian Crossing" in str(srow.get('sign_classes_2', ''))):
                    special_not_risky_flag = 1
                    break

        roadwidth_in_range = df_road_width[(df_road_width.get('Frame Index', -1) >= start_frame) &
                                           (df_road_width.get('Frame Index', -1) <= end_frame)]
        avg_road_width = roadwidth_in_range.get('Road Width (m)', pd.Series([-1])).mean() if not roadwidth_in_range.empty else -1

        crosswalk_in_range = df_crosswalk[(df_crosswalk.get('frame_id', -1) >= start_frame) &
                                          (df_crosswalk.get('frame_id', -1) <= end_frame)]
        if not crosswalk_in_range.empty:
            ratio = (crosswalk_in_range.get('crosswalk_detected', pd.Series([])) == "Yes").sum() / len(crosswalk_in_range)
            crosswalk_flag = 1 if ratio > 0.2 else 0
        else:
            crosswalk_flag = -1

        sidewalk_in_range = df_sidewalk[(df_sidewalk.get('frame_id', -1) >= start_frame) &
                                        (df_sidewalk.get('frame_id', -1) <= end_frame)]
        if not sidewalk_in_range.empty:
            ratio = (sidewalk_in_range.get('polygons', pd.Series([])).notna() &
                     (sidewalk_in_range['polygons'] != "")).sum() / len(sidewalk_in_range)
            sidewalk_flag = 1 if ratio > 0.5 else 0
        else:
            sidewalk_flag = -1

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
            'sidewalk': sidewalk_flag
        }
        merged_row.update(vehicle_presence)
        merged_rows.append(merged_row)

    df_result = pd.DataFrame(merged_rows)

    if df_result.empty:
        df_result = pd.DataFrame(columns=[
                                             'track_id', 'crossed', 'weather', 'daytime', 'police_car', 'arrow_board',
                                             'cones', 'accident', 'crack', 'potholes', 'avg_vehicle_total',
                                             'crossing_sign', 'avg_road_width', 'crosswalk', 'sidewalk'
                                         ])

    df_result.to_csv(output_csv_path, index=False)
    print(f"Crossed Pedestrian Environment analysis Result saved to {output_csv_path}")
