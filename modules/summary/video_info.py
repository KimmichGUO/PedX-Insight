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
                             on_lane_csv=None
                             ):

    # 1 name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[A1]video_info.csv")
    if tracked_pedestrian_csv is None:
        tracked_pedestrian_csv = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
    if vehicle_count_csv is None:
        vehicle_count_csv = os.path.join(output_dir, "[V6]vehicle_count.csv")
    if sidewalk_csv is None:
        sidewalk_csv = os.path.join(output_dir, "[E9]sidewalk_detection.csv")
    if crosswalk_csv is None:
        crosswalk_csv = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    if traffic_light_csv is None:
        traffic_light_csv = os.path.join(output_dir, "[E2]traffic_light.csv")
    if road_width_csv is None:
        road_width_csv = os.path.join(output_dir, "[E5]road_width.csv")
    if weather_csv_path is None:
        weather_csv_path = os.path.join(output_dir, "[E1]weather.csv")
    if road_condition_csv is None:
        road_condition_csv = os.path.join(output_dir, "[E4]road_condition.csv")
    if accident_csv_path is None:
        accident_csv_path = os.path.join(output_dir, "[E8]accident_detection.csv")
    if run_red_csv is None:
        run_red_csv = os.path.join(output_dir, "[C5]red_light_runner.csv")
    if risky_csv_path is None:
        risky_csv_path = os.path.join(output_dir, "[C1]risky_crossing.csv")
    if traffic_sign_path is None:
        traffic_sign_path = os.path.join(output_dir, "[E3]traffic_sign.csv")
    if phone_csv_path is None:
        phone_csv_path = os.path.join(output_dir, "[P5]phone_usage.csv")
    if age_csv_path is None:
        age_csv_path = os.path.join(output_dir, "[P6]age_gender.csv")
    if crosswalk_usage_csv is None:
        crosswalk_usage_csv = os.path.join(output_dir, "[C4]crosswalk_usage.csv")
    if on_lane_csv is None:
        on_lane_csv = os.path.join(output_dir, "[C8]pedestrian_on_lane.csv")




    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else None
    cap.release()

    tracked_df = pd.read_csv(tracked_pedestrian_csv)
    vehicle_df = pd.read_csv(vehicle_count_csv)
    weather_df = pd.read_csv(weather_csv_path)
    sidewalk_df = pd.read_csv(sidewalk_csv)
    crosswalk_df = pd.read_csv(crosswalk_csv)
    traffic_light_df = pd.read_csv(traffic_light_csv)
    road_width_df = pd.read_csv(road_width_csv)
    road_condition_df = pd.read_csv(road_condition_csv)
    accident_df = pd.read_csv(accident_csv_path)
    runred_df = pd.read_csv(run_red_csv)
    risky_df = pd.read_csv(risky_csv_path)
    traffic_sign_df = pd.read_csv(traffic_sign_path)

    # 4 total pedestrians
    min_frames_threshold = fps * 0.5

    valid_pedestrians = []
    for tid, group in tracked_df.groupby('track_id'):
        if len(group) >= min_frames_threshold:
            valid_pedestrians.append(tid)
    # total_pedestrians = tracked_df['track_id'].nunique()
    total_pedestrians = len(valid_pedestrians)

    if 'track_id' in risky_df.columns and 'risk' in risky_df.columns:
        crossed_ids = risky_df['track_id'].unique()
        total_crossers = len(crossed_ids)

        risky_count = 0
        for tid in crossed_ids:
            person_df = risky_df[risky_df['track_id'] == tid]
            risky_ratio = (person_df['risk'].str.lower() == 'risky').sum() / len(person_df)
            if risky_ratio > 0.3:
                risky_count += 1

        risky_crossing_ratio = risky_count / total_crossers if total_crossers > 0 else 0
    else:
        risky_crossing_ratio = None

        # 4+2 runred
    if 'track_id' in runred_df.columns and 'ran_red_light' in runred_df.columns:
        total_crossers_runred = runred_df['track_id'].nunique()
        runred_ids = runred_df.loc[runred_df['ran_red_light'] == True, 'track_id'].unique()
        runred_ratio = len(runred_ids) / total_crossers_runred if total_crossers_runred > 0 else 0
    else:
        runred_ratio = None


    # 5-6 vehicle total + top3
    vehicle_df = vehicle_df[vehicle_df['Vehicle_Type'].str.lower() != 'total']
    total_vehicles = int(vehicle_df['Count'].sum())
    top3_vehicles = vehicle_df.sort_values(by='Count', ascending=False).head(3)['Vehicle_Type'].tolist()

    # 7 weather
    main_weather = weather_df['weather_label'].mode().iloc[0]

    # 8 sidewalk prob
    sidewalk_prob = (sidewalk_df['polygons'].astype(str).str.strip() != "").sum() / total_frames

    # 9 crosswalk prob
    crosswalk_prob = (crosswalk_df['crosswalk_detected'].str.lower() == "yes").sum() / total_frames

    # 10 traffic light prob
    colors = ['yellow', 'red', 'green']
    traffic_light_frames = traffic_light_df['main_light_color'].isin(colors).sum()
    traffic_light_prob = traffic_light_frames / total_frames if total_frames > 0 else 0

    # 11 avg road width
    avg_road_width = road_width_df['Road Width (m)'].mean() if 'Road Width (m)' in road_width_df.columns else None

    # 12-13 condition: Crack + Potholes
    crack_prob = (
        road_condition_df[['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']].sum(axis=1) > 0
    ).sum() / total_frames if all(col in road_condition_df.columns for col in ['Longitudinal Crack', 'Transverse Crack', 'Alligator Crack']) else None

    pothole_prob = road_condition_df['Potholes'].sum() / total_frames if 'Potholes' in road_condition_df.columns else None

    # 14-17 accident
    accident_probs = {}
    for cls in ['police_car', 'Arrow Board', 'cones', 'accident']:
        if cls in accident_df.columns:
            accident_probs[cls] = accident_df[cls].sum() / total_frames
        else:
            accident_probs[cls] = None

    # 18 traffic sign total count
    total_traffic_signs = 0
    if 'sign_classes_1' in traffic_sign_df.columns and 'sign_classes_2' in traffic_sign_df.columns:
        count_1 = traffic_sign_df['sign_classes_1'].fillna('').apply(lambda x: len([s for s in str(x).split(';') if s.strip() != ''])).sum()
        count_2 = traffic_sign_df['sign_classes_2'].fillna('').apply(lambda x: len([s for s in str(x).split(';') if s.strip() != ''])).sum()
        total_traffic_signs = int(count_1 + count_2)

    else:
        total_traffic_signs = None
    signs_rate = total_traffic_signs / total_frames

    # 19 total crossed pedestrians
    total_crossed_pedestrians = runred_df['track_id'].nunique() if 'track_id' in runred_df.columns else 0

    # 20 phone usage ratio
    phone_df = pd.read_csv(phone_csv_path)
    phone_usage_ratio = 0
    if not phone_df.empty:
        phone_summary = phone_df.groupby('track_id')['phone_using'].mean()
        phone_using_true_count = (phone_summary > 0.1).sum()
        total_track_ids = phone_summary.shape[0]
        phone_usage_ratio = phone_using_true_count / total_track_ids if total_track_ids > 0 else 0

    # 21 crosswalk usage ratio
    crosswalk_usage_df = pd.read_csv(crosswalk_usage_csv)
    crosswalk_ratio = 0
    if 'used_crosswalk' in crosswalk_usage_df.columns:
        crosswalk_ratio = (crosswalk_usage_df['used_crosswalk'] == True).sum() / len(crosswalk_usage_df) if len(
            crosswalk_usage_df) > 0 else 0

    # 22 age mode
    age_df = pd.read_csv(age_csv_path)
    age_mode = None
    if 'age' in age_df.columns:
        age_mode = age_df['age'].mode().iloc[0] if not age_df['age'].mode().empty else None

    # 23 on_lane ratio
    on_lane_df = pd.read_csv(on_lane_csv)
    on_lane_ratio = 0
    # if 'entered_lane' in on_lane_df.columns:
    #     on_lane_ratio = (on_lane_df['entered_lane'] == True).sum() / len(on_lane_df) if len(on_lane_df) > 0 else 0
    if 'entered_lane' in on_lane_df.columns and len(on_lane_df) > 0:
        track_entered = on_lane_df.groupby('track_id')['entered_lane'].any()
        on_lane_ratio = track_entered.sum() / len(track_entered)

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
        ["walk_on_lane_ratio", on_lane_ratio],
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
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    output_df.to_csv(output_csv_path, index=False)
    print(f"Video environment stats saved to: {output_csv_path}")