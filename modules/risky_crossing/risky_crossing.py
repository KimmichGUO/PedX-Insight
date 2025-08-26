import cv2
import os
import pandas as pd


def detect_crossing_risk(
        video_path,
        traffic_light_csv=None,
        crosswalk_csv=None,
        traffic_sign_csv=None,
        crossing_judge_csv=None,
        output_csv_path=None
):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    os.makedirs(output_dir, exist_ok=True)

    if output_csv_path is None:
        output_csv_path = os.path.join(output_dir, "[C1]risky_crossing.csv")
    if traffic_light_csv is None:
        traffic_light_csv = os.path.join(output_dir, "[E2]traffic_light.csv")
    if crosswalk_csv is None:
        crosswalk_csv = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
    if traffic_sign_csv is None:
        traffic_sign_csv = os.path.join(output_dir, "[E3]traffic_sign.csv")
    if crossing_judge_csv is None:
        crossing_judge_csv = os.path.join(output_dir, "[C3]crossing_judge.csv")

    df_crossing = pd.read_csv(crossing_judge_csv)
    crossed_pedestrians = df_crossing[df_crossing['crossed'] == True].copy()

    if crossed_pedestrians.empty:
        columns = [
            'track_id', 'crossed', 'started_frame', 'ended_frame',
            'total_frames', 'risky_frames', 'risky_ratio', 'risk'
        ]
        pd.DataFrame(columns=columns).to_csv(output_csv_path, index=False)
        print(f"No crossed pedestrians found. Empty results saved to {output_csv_path}")
        return

    df_light = pd.read_csv(traffic_light_csv)
    light_state_map = dict(zip(df_light["frame_id"], df_light["main_light_color"]))

    df_crosswalk = pd.read_csv(crosswalk_csv)
    crosswalk_map = {
        row["frame_id"]: row["crosswalk_detected"].strip().lower() == "yes"
        for _, row in df_crosswalk.iterrows()
    }

    df_sign = pd.read_csv(traffic_sign_csv)
    sign_map = {}
    for _, row in df_sign.iterrows():
        sign_map[row["frame_id"]] = {
            "sign_classes_1": str(row["sign_classes_1"]).split(";") if pd.notna(row["sign_classes_1"]) else [],
            "sign_classes_2": str(row["sign_classes_2"]).split(";") if pd.notna(row["sign_classes_2"]) else []
        }

    results = []

    for _, pedestrian in crossed_pedestrians.iterrows():
        track_id = pedestrian['track_id']
        started_frame = int(pedestrian['started_frame'])
        ended_frame = int(pedestrian['ended_frame'])

        risky_frame_count = 0
        total_frame_count = ended_frame - started_frame + 1

        for frame_id in range(started_frame, ended_frame + 1):
            light_state = light_state_map.get(frame_id, "None")
            crosswalk_present = crosswalk_map.get(frame_id, False)

            sign_classes_1 = sign_map.get(frame_id, {}).get("sign_classes_1", [])
            sign_classes_2 = sign_map.get(frame_id, {}).get("sign_classes_2", [])

            special_not_risky_signs = {"w57", "pg", "i1"}
            special_not_risky_flag = (
                    any(sign in special_not_risky_signs for sign in sign_classes_1) or
                    "Pedestrian Crossing" in sign_classes_2
            )
            special_risky_flag = "p9" in sign_classes_1

            frame_risk = "not risky"

            if special_risky_flag:
                frame_risk = "risky"
            elif light_state == "green":
                frame_risk = "risky"
            elif special_not_risky_flag:
                frame_risk = "not risky"
            else:
                if light_state == "yellow" and not crosswalk_present:
                    frame_risk = "risky"
                elif light_state == "None" and not crosswalk_present:
                    frame_risk = "risky"
                else:
                    frame_risk = "not risky"

            if frame_risk == "risky":
                risky_frame_count += 1

        risky_ratio = risky_frame_count / total_frame_count if total_frame_count > 0 else 0.0
        final_risk = "risky" if risky_ratio > 0.1 else "not risky"

        results.append({
            "track_id": track_id,
            "crossed": True,
            "started_frame": started_frame,
            "ended_frame": ended_frame,
            "total_frames": total_frame_count,
            "risky_frames": risky_frame_count,
            "risky_ratio": round(risky_ratio, 3),
            "risk": final_risk
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(output_csv_path, index=False)

# import cv2
# import os
# import pandas as pd
# from collections import defaultdict, deque
#
# def detect_crossing_risk(
#     video_path,
#     direction_csv_path=None,
#     traffic_light_csv=None,
#     crosswalk_csv=None,
#     traffic_sign_csv=None,
#     output_csv_path=None
# ):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     output_dir = os.path.join("analysis_results", video_name)
#     os.makedirs(output_dir, exist_ok=True)
#
#     if output_csv_path is None:
#         output_csv_path = os.path.join(output_dir, "[C1]risky_crossing.csv")
#     if direction_csv_path is None:
#         direction_csv_path = os.path.join(output_dir, "[B1]tracked_pedestrians.csv")
#     if traffic_light_csv is None:
#         traffic_light_csv = os.path.join(output_dir, "[E2]traffic_light.csv")
#     if crosswalk_csv is None:
#         crosswalk_csv = os.path.join(output_dir, "[E7]crosswalk_detection.csv")
#     if traffic_sign_csv is None:
#         traffic_sign_csv = os.path.join(output_dir, "[E3]traffic_sign.csv")
#
#     df = pd.read_csv(direction_csv_path)
#     df.set_index(["frame_id", "track_id"], inplace=True)
#
#     df_light = pd.read_csv(traffic_light_csv)
#     light_state_map = dict(zip(df_light["frame_id"], df_light["main_light_color"]))
#
#     df_crosswalk = pd.read_csv(crosswalk_csv)
#     crosswalk_map = {
#         row["frame_id"]: row["crosswalk_detected"].strip().lower() == "yes"
#         for _, row in df_crosswalk.iterrows()
#     }
#
#     df_sign = pd.read_csv(traffic_sign_csv)
#     sign_map = {}
#     for _, row in df_sign.iterrows():
#         sign_map[row["frame_id"]] = {
#             "sign_classes_1": str(row["sign_classes_1"]).split(";") if pd.notna(row["sign_classes_1"]) else [],
#             "sign_classes_2": str(row["sign_classes_2"]).split(";") if pd.notna(row["sign_classes_2"]) else []
#         }
#
#     cap = cv2.VideoCapture(video_path)
#     results = []
#
#     frame_id = -1
#     id_x_history = defaultdict(lambda: deque(maxlen=30))
#
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame_id += 1
#         frame_center_x = frame.shape[1] // 2
#
#         light_state = light_state_map.get(frame_id, "None")
#         crosswalk_present = crosswalk_map.get(frame_id, False)
#
#         sign_classes_1 = sign_map.get(frame_id, {}).get("sign_classes_1", [])
#         sign_classes_2 = sign_map.get(frame_id, {}).get("sign_classes_2", [])
#
#         if frame_id not in df.index.get_level_values(0):
#             continue
#
#         frame_rows = df.loc[frame_id]
#         if isinstance(frame_rows, pd.Series):
#             frame_rows = frame_rows.to_frame().T
#
#         for track_id, row in frame_rows.iterrows():
#             x_center = (row["x1"] + row["x2"]) / 2
#             id_x_history[track_id].append(x_center)
#
#             if len(id_x_history[track_id]) < 5:
#                 continue
#
#             x_list = list(id_x_history[track_id])
#             crossed = (x_list[0] < frame_center_x and x_list[-1] > frame_center_x) or \
#                       (x_list[0] > frame_center_x and x_list[-1] < frame_center_x)
#
#             if not crossed:
#                 continue
#
#             special_not_risky_signs = {"w57", "pg", "i1"}
#             special_not_risky_flag = (
#                 any(sign in special_not_risky_signs for sign in sign_classes_1) or
#                 "Pedestrian Crossing" in sign_classes_2
#             )
#             special_risky_flag = "p9" in sign_classes_1
#
#             if special_risky_flag:
#                 risk = "risky"
#             elif light_state == "green":
#                 risk = "risky"
#             elif special_not_risky_flag:
#                 risk = "not risky"
#             else:
#                 if light_state == "yellow" and not crosswalk_present:
#                     risk = "risky"
#                 elif light_state == "None" and not crosswalk_present:
#                     risk = "risky"
#                 else:
#                     risk = "not risky"
#
#             results.append({
#                 "frame_id": frame_id,
#                 "track_id": track_id if isinstance(track_id, int) else track_id[1],
#                 "x_start": round(x_list[0], 2),
#                 "x_end": round(x_list[-1], 2),
#                 "crossed": True,
#                 "light_state": light_state,
#                 "crosswalk_present": crosswalk_present,
#                 "Crossing_signal": special_not_risky_flag,
#                 "No_Crossing_signal": special_risky_flag,
#                 "risk": risk
#             })
#
#     cap.release()
#     cv2.destroyAllWindows()
#     pd.DataFrame(results).to_csv(output_csv_path, index=False)
#     print(f"Crossing risk analysis completed. Results saved to {output_csv_path}")