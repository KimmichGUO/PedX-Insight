import os
import pandas as pd
import cv2

def generate_video_env_stats(video_path,
                             tracked_pedestrian_csv=None,
                             vehicle_count_csv=None,
                             weather_csv_path=None,
                             sidewalk_csv=None,
                             crosswalk_csv=None,
                             traffic_light_csv=None,
                             road_width_csv=None,
                             road_condition_csv=None,
                             accident_csv_path=None,
                             output_csv_path=None,
                             run_red_csv=None,
                             risky_csv_path=None,
                             traffic_sign_path=None,
                             phone_csv_path=None,
                             age_csv_path=None,
                             crosswalk_usage_csv=None,
                             ):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)

    tracked_pedestrian_csv = tracked_pedestrian_csv or os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    vehicle_count_csv = vehicle_count_csv or os.path.join(output_dir, "[V6]vehicle_count.csv")
    weather_csv_path = weather_csv_path or os.path.join(output_dir, "[E1]weather.csv")
    sidewalk_csv = sidewalk_csv or os.path.join(output_dir, "[E9]sidewalk_detection.csv")
    crosswalk_csv = crosswalk_csv or os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    traffic_light_csv = traffic_light_csv or os.path.join(output_dir, "[E2]traffic_light.csv")
    road_width_csv = road_width_csv or os.path.join(output_dir, "[E5]road_width.csv")
    road_condition_csv = road_condition_csv or os.path.join(output_dir, "[E4]road_condition.csv")
    accident_csv_path = accident_csv_path or os.path.join(output_dir, "[E8]accident_detection.csv")
    run_red_csv = run_red_csv or os.path.join(output_dir, "[C5]red_light_runner.csv")
    risky_csv_path = risky_csv_path or os.path.join(output_dir, "[C1]risky_crossing.csv")
    traffic_sign_path = traffic_sign_path or os.path.join(output_dir, "[E3]traffic_sign.csv")
    phone_csv_path = phone_csv_path or os.path.join(output_dir, "[P5]phone_usage.csv")
    age_csv_path = age_csv_path or os.path.join(output_dir, "[P6]age_gender.csv")
    crosswalk_usage_csv = crosswalk_usage_csv or os.path.join(output_dir, "[C4]crosswalk_usage.csv")
    output_csv_path = output_csv_path or os.path.join(output_dir, "[A1]video_info.csv")

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else None
    cap.release()

    def safe_read_csv(path):
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df if not df.empty else None
        return None

    tracked_df = safe_read_csv(tracked_pedestrian_csv)
    vehicle_df = safe_read_csv(vehicle_count_csv)
    weather_df = safe_read_csv(weather_csv_path)
    sidewalk_df = safe_read_csv(sidewalk_csv)
    crosswalk_df = safe_read_csv(crosswalk_csv)
    traffic_light_df = safe_read_csv(traffic_light_csv)
    road_width_df = safe_read_csv(road_width_csv)
    road_condition_df = safe_read_csv(road_condition_csv)
    accident_df = safe_read_csv(accident_csv_path)
    runred_df = safe_read_csv(run_red_csv)
    risky_df = safe_read_csv(risky_csv_path)
    traffic_sign_df = safe_read_csv(traffic_sign_path)
    phone_df = safe_read_csv(phone_csv_path)
    age_df = safe_read_csv(age_csv_path)
    crosswalk_usage_df = safe_read_csv(crosswalk_usage_csv)

    # 总行人
    min_frames_threshold = fps * 0.5 if fps else 0
    valid_pedestrians = []
    if tracked_df is not None:
        for tid, group in tracked_df.groupby('track_id'):
            if len(group) >= min_frames_threshold:
                valid_pedestrians.append(tid)
    total_pedestrians = len(valid_pedestrians) if tracked_df is not None else None

    risky_crossing_ratio = None
    if risky_df is not None and 'track_id' in risky_df.columns and 'risk' in risky_df.columns:
        crossed_ids = risky_df['track_id'].unique()
        risky_count = 0
        for tid in crossed_ids:
            person_df = risky_df[risky_df['track_id'] == tid]
            risky_ratio = (person_df['risk'].str.lower() == 'risky').sum() / len(person_df)
            if risky_ratio > 0.3:
                risky_count += 1
        risky_crossing_ratio = risky_count / len(crossed_ids) if len(crossed_ids) > 0 else None

    runred_ratio = None
    if runred_df is not None and 'track_id' in runred_df.columns and 'ran_red_light' in runred_df.columns:
        total_crossers_runred = runred_df['track_id'].nunique()
        runred_ids = runred_df.loc[runred_df['ran_red_light'] == True, 'track_id'].unique()
        runred_ratio = len(runred_ids) / total_crossers_runred if total_crossers_runred > 0 else None

    total_vehicles = int(vehicle_df['Count'].sum()) if vehicle_df is not None else None
    top3_vehicles = vehicle_df.sort_values(by='Count', ascending=False).head(3)['Vehicle_Type'].tolist() if vehicle_df is not None else None

    main_weather = weather_df['weather_label'].mode().iloc[0] if weather_df is not None and 'weather_label' in weather_df.columns else None

    sidewalk_prob = (sidewalk_df['polygons'].astype(str).str.strip() != "").sum() / total_frames if sidewalk_df is not None else None

    crosswalk_prob = (crosswalk_df['crosswalk_detected'].str.lower() == "yes").sum() / total_frames if crosswalk_df is not None else None

    traffic_light_prob = None
    if traffic_light_df is not None and 'main_light_color' in traffic_light_df.columns:
        colors = ['yellow', 'red', 'green']
        traffic_light_frames = traffic_light_df['main_light_color'].isin(colors).sum()
        traffic_light_prob = traffic_light_frames / total_frames if total_frames > 0 else None

    avg_road_width = road_width_df['Road Width (m)'].mean() if road_width_df is not None and 'Road Width (m)' in road_width_df.columns else None

    crack_prob = None
    pothole_prob = None
    if road_condition_df is not None:
        if all(col in road_condition_df.columns for col in ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']):
            crack_prob = (road_condition_df[['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']].sum(axis=1) > 0).sum() / total_frames
        if 'Potholes' in road_condition_df.columns:
            pothole_prob = road_condition_df['Potholes'].sum() / total_frames

    accident_probs = {}
    for cls in ['police_car', 'Arrow Board', 'cones', 'accident']:
        if accident_df is not None and cls in accident_df.columns:
            accident_probs[cls] = accident_df[cls].sum() / total_frames
        else:
            accident_probs[cls] = None

    total_traffic_signs = None
    signs_rate = None
    if traffic_sign_df is not None and 'sign_classes_1' in traffic_sign_df.columns and 'sign_classes_2' in traffic_sign_df.columns:
        count_1 = traffic_sign_df['sign_classes_1'].fillna('').apply(lambda x: len([s for s in str(x).split(';') if s.strip() != ''])).sum()
        count_2 = traffic_sign_df['sign_classes_2'].fillna('').apply(lambda x: len([s for s in str(x).split(';') if s.strip() != ''])).sum()
        total_traffic_signs = int(count_1 + count_2)
        signs_rate = total_traffic_signs / total_frames if total_frames > 0 else None

    total_crossed_pedestrians = runred_df['track_id'].nunique() if runred_df is not None and 'track_id' in runred_df.columns else None

    phone_usage_ratio = None
    if phone_df is not None and 'phone_using' in phone_df.columns:
        phone_summary = phone_df.groupby('track_id')['phone_using'].mean()
        phone_using_true_count = (phone_summary > 0.1).sum()
        total_track_ids = phone_summary.shape[0]
        phone_usage_ratio = phone_using_true_count / total_track_ids if total_track_ids > 0 else None

    crosswalk_ratio = None
    if crosswalk_usage_df is not None and 'used_crosswalk' in crosswalk_usage_df.columns:
        crosswalk_ratio = (crosswalk_usage_df['used_crosswalk'] == True).sum() / len(crosswalk_usage_df) if len(crosswalk_usage_df) > 0 else None

    age_mode = None
    if age_df is not None and 'age' in age_df.columns:
        age_mode = age_df['age'].mode().iloc[0] if not age_df['age'].mode().empty else None


    data = [
        ["video_name", video_name],
        ["duration_seconds", duration],
        ["total_frames", total_frames],
        ["total_pedestrians", total_pedestrians],
        ["total_crossed_pedestrians", total_crossed_pedestrians],
        ["average_age", age_mode],
        ["phone_usage_ratio", phone_usage_ratio],
        ["risky_crossing_ratio", risky_crossing_ratio],
        ["run_red_light_ratio", runred_ratio],
        ["crosswalk_usage_ratio", crosswalk_ratio],
        ["traffic_signs_ratio", signs_rate],
        ["total_vehicles", total_vehicles],
        ["top3_vehicles", top3_vehicles],
        ["main_weather", main_weather],
        ["sidewalk_prob", sidewalk_prob],
        ["crosswalk_prob", crosswalk_prob],
        ["traffic_light_prob", traffic_light_prob],
        ["avg_road_width", avg_road_width],
        ["Crack_prob", crack_prob],
        ["Potholes_prob", pothole_prob],
    ]

    for cls, prob in accident_probs.items():
        data.append([f"{cls}_prob", prob])

    output_df = pd.DataFrame(data, columns=["metric", "value"])
    output_df.to_csv(output_csv_path, index=False)
    print(f"Video environment stats saved to: {output_csv_path}")