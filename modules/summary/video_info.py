import os
import math
import pandas as pd
import cv2

def generate_video_env_stats(video_path,
                             analysis_interval=1.0,
                             speed_csv_path=None,
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
    speed_csv_path = speed_csv_path or os.path.join(output_dir, "[S1]pedestrian_speed.csv")
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
    speed_df = safe_read_csv(speed_csv_path)

    # Total pedestrians. [B1] holds ~1 row per analysis_interval seconds (NOT per native
    # frame), so the old `fps*0.5` frame count demanded ~15-30 s of continuous same-id
    # presence and severely undercounted. Threshold on the number of SAMPLED rows that
    # corresponds to a minimum visible duration instead.
    min_visible_seconds = 1.5
    interval = analysis_interval if analysis_interval and analysis_interval > 0 else 1.0
    min_rows_threshold = max(1, math.ceil(min_visible_seconds / interval))
    valid_pedestrians = []
    if tracked_df is not None:
        for tid, group in tracked_df.groupby('track_id'):
            if len(group) >= min_rows_threshold:
                valid_pedestrians.append(tid)
    total_pedestrians = len(valid_pedestrians) if tracked_df is not None else None

    # Measured (not imported) mean pedestrian walking speed in m/s, from [S1]. Prefer
    # tracks flagged reliable; fall back to all tracks if none qualify.
    measured_walking_speed = None
    if speed_df is not None and 'walking_speed_mps' in speed_df.columns and not speed_df.empty:
        reliable = speed_df[speed_df['reliable'] == True] if 'reliable' in speed_df.columns else speed_df
        use = reliable if not reliable.empty else speed_df
        val = use['walking_speed_mps'].median()
        measured_walking_speed = round(float(val), 3) if val == val else None

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

    if vehicle_df is not None:
        # Exclude the producer's 'Total' summary row (written with a capital 'T' by
        # count_vehicle.py); a case-sensitive '!= "total"' filter would leave it in and
        # both double-count total_vehicles and let 'Total' lead top3_vehicles.
        non_total_df = vehicle_df[vehicle_df['Vehicle_Type'].str.lower() != 'total']
        total_vehicles = int(non_total_df['Count'].sum())
        top3_vehicles = non_total_df.sort_values(by='Count', ascending=False).head(3)['Vehicle_Type'].tolist()
    else:
        total_vehicles = None
        top3_vehicles = None

    main_weather = weather_df['weather_label'].mode().iloc[0] if weather_df is not None and 'weather_label' in weather_df.columns else None

    # Each source CSV is divided by its own row count, not total_frames: some producers
    # write one row per analyzed (every-Nth) frame (traffic_light, traffic_sign, accident,
    # sidewalk) while others write one row per frame. Using len(df) yields the correct
    # fraction in both cases and avoids a divide-by-zero when the video is unreadable
    # (safe_read_csv already returns None for empty inputs, so len(df) >= 1 here).
    # Count only rows with a real, non-empty polygons string. astype(str) alone would
    # turn NaN (empty cell = no sidewalk that frame) into the truthy string 'nan' and
    # pin sidewalk_prob at ~1.0 regardless of content; require .notna() first
    # (mirroring env_info.py's handling of the same column).
    sidewalk_prob = None
    if sidewalk_df is not None and 'polygons' in sidewalk_df.columns:
        sidewalk_prob = (sidewalk_df['polygons'].notna() &
                         (sidewalk_df['polygons'].astype(str).str.strip() != "")).sum() / len(sidewalk_df)

    crosswalk_prob = (crosswalk_df['crosswalk_detected'].str.lower() == "yes").sum() / len(crosswalk_df) if crosswalk_df is not None else None

    traffic_light_prob = None
    if traffic_light_df is not None and 'main_light_color' in traffic_light_df.columns:
        colors = ['yellow', 'red', 'green']
        traffic_light_frames = traffic_light_df['main_light_color'].isin(colors).sum()
        traffic_light_prob = traffic_light_frames / len(traffic_light_df)

    avg_road_width = road_width_df['Road Width (m)'].mean() if road_width_df is not None and 'Road Width (m)' in road_width_df.columns else None

    crack_prob = None
    pothole_prob = None
    if road_condition_df is not None:
        if all(col in road_condition_df.columns for col in ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']):
            crack_prob = (road_condition_df[['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']].sum(axis=1) > 0).sum() / len(road_condition_df)
        if 'Potholes' in road_condition_df.columns:
            pothole_prob = road_condition_df['Potholes'].sum() / len(road_condition_df)

    accident_probs = {}
    for cls in ['police_car', 'Arrow Board', 'cones', 'accident']:
        if accident_df is not None and cls in accident_df.columns:
            accident_probs[cls] = accident_df[cls].sum() / len(accident_df)
        else:
            accident_probs[cls] = None

    total_traffic_signs = None
    signs_rate = None
    if traffic_sign_df is not None and 'sign_classes_1' in traffic_sign_df.columns and 'sign_classes_2' in traffic_sign_df.columns:
        count_1 = traffic_sign_df['sign_classes_1'].fillna('').apply(lambda x: len([s for s in str(x).split(';') if s.strip() != ''])).sum()
        count_2 = traffic_sign_df['sign_classes_2'].fillna('').apply(lambda x: len([s for s in str(x).split(';') if s.strip() != ''])).sum()
        total_traffic_signs = int(count_1 + count_2)
        signs_rate = total_traffic_signs / len(traffic_sign_df)

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
        ["measured_avg_walking_speed_mps", measured_walking_speed],
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