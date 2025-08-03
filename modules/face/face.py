# import cv2
# import pandas as pd
# from deepface import DeepFace
# import os
#
# def run_face_analysis(video_path, tracking_csv_path = None, output_csv_path = None):
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     if tracking_csv_path is None:
#         tracking_csv_path = os.path.join(".", "analysis_results", video_name, "tracked_pedestrians.csv")
#     if output_csv_path is None:
#         output_dir = os.path.join(".", "analysis_results", video_name)
#         os.makedirs(output_dir, exist_ok=True)
#         output_csv_path = os.path.join(output_dir, "face_analysis.csv")
#
#     df = pd.read_csv(tracking_csv_path)
#     cap = cv2.VideoCapture(video_path)
#     fps = cap.get(cv2.CAP_PROP_FPS)
#
#     results = []
#
#     print("Start analyzing age, gender, and race frame by frame...")
#
#     for idx, row in df.iterrows():
#         frame_id = int(row["frame_id"])
#         track_id = row["track_id"]
#         x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
#
#         cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
#         ret, frame = cap.read()
#         if not ret:
#             print(f"Frame {frame_id} not readable.")
#             continue
#
#         person_crop = frame[y1:y2, x1:x2]
#         if person_crop.size == 0:
#             print(f"Track ID {track_id} - empty crop at frame {frame_id}.")
#             continue
#
#         try:
#             analysis = DeepFace.analyze(
#                 person_crop,
#                 actions=['age', 'gender', 'race'],
#                 enforce_detection=False
#             )
#         except Exception as e:
#             print(f"Track ID {track_id} @ Frame {frame_id} - Detection failed:", e)
#             continue
#
#         if isinstance(analysis, dict):
#             analysis = [analysis]
#
#         face = analysis[0]
#         results.append({
#             "frame_id": frame_id,
#             "track_id": track_id,
#             "age": int(face.get("age", -1)),
#             "gender": face.get("dominant_gender", "Unknown"),
#             "race": face.get("dominant_race", "Unknown")
#         })
#         print(f"Track ID {track_id} @ Frame {frame_id} analyzed.")
#
#     cap.release()
#
#     results_df = pd.DataFrame(results)
#     results_df.to_csv(output_csv_path, index=False)
#     print(f"\nAll attributes saved to {output_csv_path}")
import cv2
import pandas as pd
from deepface import DeepFace
import os

def run_face_analysis(video_path, tracking_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # 默认路径
    if tracking_csv_path is None:
        tracking_csv_path = os.path.join(".", "analysis_results", video_name, "tracked_pedestrians.csv")
    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "face_analysis.csv")

    # 加载 tracking 结果
    df = pd.read_csv(tracking_csv_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    results = []
    print("Start analyzing age, gender, and race frame by frame...")

    for idx, row in df.iterrows():
        frame_id = int(row["frame_id"])
        track_id = row["track_id"]
        x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id - 1)
        ret, frame = cap.read()
        if not ret:
            print(f"Frame {frame_id} not readable.")
            continue

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            print(f"Track ID {track_id} - empty crop at frame {frame_id}.")
            continue

        try:
            analysis = DeepFace.analyze(
                person_crop,
                actions=['age', 'gender', 'race'],
                enforce_detection=False
            )
        except Exception as e:
            print(f"Track ID {track_id} @ Frame {frame_id} - Detection failed:", e)
            continue

        if isinstance(analysis, dict):
            analysis = [analysis]

        face = analysis[0]

        # 添加文字标签
        gender = face.get('dominant_gender', 'Unknown')
        age = int(face.get('age', -1))
        race = face.get('dominant_race', 'Unknown')
        label = f"{gender}, {age}, {race}"

        # 在原图上画框和文字
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 显示整帧视频图像
        cv2.imshow("Face Analysis", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            print("User interrupted.")
            break

        # 保存当前分析结果
        results.append({
            "frame_id": frame_id,
            "track_id": track_id,
            "age": age,
            "gender": gender,
            "race": race
        })

        print(f"Track ID {track_id} @ Frame {frame_id} analyzed.")

    cap.release()
    cv2.destroyAllWindows()

    # 保存为 CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv_path, index=False)
    print(f"\nAll attributes saved to {output_csv_path}")
