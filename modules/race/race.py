import cv2
import csv
import os
from deepface import DeepFace

def run_race_detection(source_video_path):
    cap = cv2.VideoCapture(source_video_path)
    frame_index = 0

    video_basename = os.path.splitext(os.path.basename(source_video_path))[0]
    output_dir = os.path.join("analysis_results", video_basename)
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"{video_basename}_pedestrian_speed.csv")

    with open(csv_filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "detections"]) 

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1

        if frame_index % 30 != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            results = DeepFace.analyze(
                frame_rgb,
                actions=['age', 'gender', 'race'],
                enforce_detection=False
            )
        except Exception as e:
            print(f"Frame {frame_index} - Detection failed:", e)
            results = []

        if isinstance(results, dict):
            results = [results]

        face_data_list = []
        for face in results:
            age = int(face.get('age', -1))
            gender = face.get('dominant_gender', 'Unknown')
            race = face.get('dominant_race', 'Unknown')

            face_data = {
                'age': age,
                'gender': gender,
                'race': race
            }
            face_data_list.append(face_data)

        with open(csv_filename, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([frame_index, face_data_list])

        for face in results:
            region = face.get('region', {})
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            label = f"{face.get('dominant_gender', '')}, {int(face.get('age', 0))}, {face.get('dominant_race', '')}"
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("Demographics", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
