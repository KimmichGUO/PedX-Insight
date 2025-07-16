import os
import cv2
import time
import pandas as pd


def estimate_distance(bbox_width, focal_length=1000, known_width=0.5):
    if bbox_width == 0:
        return 0.0
    return (known_width * focal_length) / bbox_width


def visualize_and_estimate_distance(video_path, tracked_csv_path=None, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "pedestrian_distance.csv")
    if tracked_csv_path is None:
        tracked_csv_path = "./analysis_results/pedestrian/tracked_pedestrians.csv"

    tracked_df = pd.read_csv(tracked_csv_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return

    frame_id = -1
    target_fps = 30
    frame_time = 1.0 / target_fps
    results_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1
        # 不进行 resize，保持原始尺寸
        frame_detections = tracked_df[tracked_df['frame_id'] == frame_id]

        for _, row in frame_detections.iterrows():
            tid = int(row['track_id'])
            x1, y1, x2, y2 = int(row['x1']), int(row['y1']), int(row['x2']), int(row['y2'])
            bbox_width = x2 - x1

            distance = estimate_distance(bbox_width)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f"ID: {tid}", (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f"Dist: {distance:.2f}m", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            results_list.append({
                "frame_id": frame_id,
                "track_id": tid,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "distance(m)": round(distance, 2)
            })

        cv2.imshow("Distance Estimation from Tracking", frame)

        time.sleep(frame_time)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    pd.DataFrame(results_list).to_csv(output_csv_path, index=False)
    print(f"Distance estimation completed. Results saved to {output_csv_path}")


# import cv2
# import numpy as np
# import math
# import time
# from ultralytics import YOLO  # YOLOv8 module


# # Function to estimate distance based on bounding box size
# def estimate_distance(bbox_width, bbox_height):
#     # For simplicity, assume the distance is inversely proportional to the box size
#     # This is a basic estimation, you may use camera calibration for more accuracy
#     focal_length = 1000  # Example focal length, modify based on camera setup
#     known_width = 0.5  # Approximate width of the person or car (in meters)
#     distance = (known_width * focal_length) / bbox_width  # Basic distance estimation
#     return distance


# # Main function to read and process video with YOLOv8
# def process_video():
#     # Load the YOLOv8 model
#     model = YOLO('yolov8n.pt')

#     # Open the video file
#     cap = cv2.VideoCapture('pedestrian.mp4')

#     # Check if video opened successfully
#     if not cap.isOpened():
#         print("Error: Unable to open video file.")
#         return

#     # Set the desired frame rate
#     target_fps = 30
#     frame_time = 1.0 / target_fps  # Time per frame to maintain 30fps

#     # Resize to 720p (1280x720)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

#     # Loop through each frame
#     while cap.isOpened():
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Resize frame to 720p
#         resized_frame = cv2.resize(frame, (1280, 720))

#         # Run YOLOv8 to detect objects in the current frame
#         results = model(resized_frame)

#         # Process the detections from YOLOv8
#         for result in results:
#             boxes = result.boxes
#             for box in boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
#                 conf = box.conf[0]  # Confidence score
#                 cls = int(box.cls[0])  # Class ID

#                 # Only draw bounding boxes for cars or people with confidence >= 0.5
#                 if model.names[cls] =='person' and conf >= 0.5:
#                     label = f'{model.names[cls]} {conf:.2f}'

#                     # Draw the bounding box
#                     cv2.rectangle(resized_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
#                     cv2.putText(resized_frame, label, (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

#                     # Estimate the distance
#                     bbox_width = x2 - x1
#                     bbox_height = y2 - y1
#                     distance = estimate_distance(bbox_width, bbox_height)

#                     # Display the estimated distance
#                     distance_label = f'Distance: {distance:.2f}m'
#                     cv2.putText(resized_frame, distance_label, (x1, y2 + 20),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

#         # Display the resulting frame
#         cv2.imshow('Object Detection', resized_frame)

#         # Limit the frame rate to 30fps
#         time.sleep(frame_time)

#         # Break the loop when 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release video capture and close windows
#     cap.release()
#     cv2.destroyAllWindows()


# # Run the video processing function
# process_video()
