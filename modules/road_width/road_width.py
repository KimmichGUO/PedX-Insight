import cv2
import numpy as np
from collections import deque
import csv
import os

def run_road_width_analysis(video_path, output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    if output_csv_path is None:
        output_dir = os.path.join(".", "analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "road_width_analysis.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    src_points = np.float32([
        [frame_width * 0.45, frame_height * 0.6], 
        [frame_width * 0.55, frame_height * 0.6],  
        [frame_width * 0.1, frame_height * 0.95], 
        [frame_width * 0.9, frame_height * 0.95]  
    ])
    dest_points = np.float32([
        [frame_width * 0.25, 0],
        [frame_width * 0.75, 0],
        [frame_width * 0.25, frame_height],
        [frame_width * 0.75, frame_height]
    ])
    # src_points = np.float32([[100, 300], [500, 300], [50, 400], [550, 400]])
    # dest_points = np.float32([[100, 100], [500, 100], [100, 400], [500, 400]])
    road_width_history = deque(maxlen=10)

    def perspective_transform(frame, src_pts, dest_pts):
        matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
        return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))

    def detect_road_width(birdseye_frame):
        mask = np.zeros_like(birdseye_frame)
        roi_vertices = np.array([[(50, 300), (550, 300), (550, 400), (50, 400)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)
        masked_frame = cv2.bitwise_and(birdseye_frame, mask)
        edges = cv2.Canny(masked_frame, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=20)

        left_boundary = right_boundary = None
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if slope < 0:
                    if left_boundary is None or x1 < left_boundary[0]:
                        left_boundary = (x1, y1, x2, y2)
                elif slope > 0:
                    if right_boundary is None or x1 > right_boundary[0]:
                        right_boundary = (x1, y1, x2, y2)

        road_width_pixels = None
        if left_boundary and right_boundary:
            left_x = (left_boundary[0] + left_boundary[2]) // 2
            right_x = (right_boundary[0] + right_boundary[2]) // 2
            road_width_pixels = abs(right_x - left_x)
            road_width_history.append(road_width_pixels)
        elif road_width_history:
            road_width_pixels = np.mean(road_width_history)

        calibration_factor = 0.05  # 可调节
        return road_width_pixels * calibration_factor if road_width_pixels else None

    with open(output_csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Frame Index', 'Time (s)', 'Road Width (m)'])

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            birdseye = perspective_transform(enhanced, src_points, dest_points)

            road_width_m = detect_road_width(birdseye)
            frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            time_sec = frame_idx / fps

            if road_width_m is not None:
                writer.writerow([frame_idx, f"{time_sec:.2f}", f"{road_width_m:.2f}"])
            else:
                writer.writerow([frame_idx, f"{time_sec:.2f}", "NaN"])

    cap.release()
    print(f"Road width analysis saved to: {output_csv_path}")

# import cv2
# import numpy as np
# from collections import deque

# # Load the video file
# video_path = "1.mp4"  # Replace with your video path
# output_path = "output.mp4"  # Path to save the output video
# cap = cv2.VideoCapture(video_path)

# # Video writer to save the processed video
# frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
# out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width * 2, frame_height))

# # Perspective transformation points
# src_points = np.float32([[100, 300], [500, 300], [50, 400], [550, 400]])
# dest_points = np.float32([[100, 100], [500, 100], [100, 400], [500, 400]])

# # Queue to store road width measurements for smoothing
# road_width_history = deque(maxlen=10)

# # Perspective transformation function
# def perspective_transform(frame, src_pts, dest_pts):
#     frame_height, frame_width = frame.shape[:2]
#     matrix = cv2.getPerspectiveTransform(src_pts, dest_pts)
#     birdseye_frame = cv2.warpPerspective(frame, matrix, (frame_width, frame_height))
#     return birdseye_frame

# # Road width detection function
# def detect_road_width(birdseye_frame):
#     mask = np.zeros_like(birdseye_frame)
#     roi_vertices = np.array([[(50, 300), (550, 300), (550, 400), (50, 400)]], dtype=np.int32)
#     cv2.fillPoly(mask, roi_vertices, 255)
#     masked_frame = cv2.bitwise_and(birdseye_frame, mask)

#     edges = cv2.Canny(masked_frame, 50, 150)
#     lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)

#     left_boundary = None
#     right_boundary = None
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line[0]
#             slope = (y2 - y1) / (x2 - x1 + 1e-6)
#             if slope < 0:
#                 if left_boundary is None or x1 < left_boundary[0]:
#                     left_boundary = (x1, y1, x2, y2)
#             elif slope > 0:
#                 if right_boundary is None or x1 > right_boundary[0]:
#                     right_boundary = (x1, y1, x2, y2)

#     road_width_pixels = None
#     if left_boundary and right_boundary:
#         left_x = (left_boundary[0] + left_boundary[2]) // 2
#         right_x = (right_boundary[0] + right_boundary[2]) // 2
#         road_width_pixels = abs(right_x - left_x)
#         road_width_history.append(road_width_pixels)
#     elif road_width_history:
#         road_width_pixels = np.mean(road_width_history)

#     calibration_factor = 0.05  # Adjust based on calibration
#     road_width_meters = road_width_pixels * calibration_factor if road_width_pixels else None

#     visualization = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
#     if left_boundary:
#         cv2.line(visualization, (left_boundary[0], left_boundary[1]), (left_boundary[2], left_boundary[3]), (0, 255, 0), 2)
#     if right_boundary:
#         cv2.line(visualization, (right_boundary[0], right_boundary[1]), (right_boundary[2], right_boundary[3]), (0, 0, 255), 2)
#     if road_width_meters:
#         cv2.putText(visualization, f"Width: {road_width_meters:.2f} m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

#     return visualization, road_width_meters

# # Process and visualize frames
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Preprocessing (from Step 2)
#     blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
#     gray_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     enhanced_frame = clahe.apply(gray_frame)
#     birdseye_frame = perspective_transform(enhanced_frame, src_points, dest_points)

#     # Detect road width and boundaries
#     visualization, road_width_meters = detect_road_width(birdseye_frame)

#     # Combine original and processed views side by side
#     combined_frame = np.hstack((frame, visualization))

#     # Save processed frame to output video
#     out.write(combined_frame)


#     def resize_frame(frame, max_width=1280, max_height=720):
#         h, w = frame.shape[:2]
#         scale_w = max_width / w
#         scale_h = max_height / h
#         scale = min(scale_w, scale_h, 1)  
#         new_w, new_h = int(w * scale), int(h * scale)
#         resized_frame = cv2.resize(frame, (new_w, new_h))
#         return resized_frame

#     display_frame = resize_frame(combined_frame)
#     cv2.imshow("Road Width Measurement", display_frame)
#     # Display the combined frame
#     # cv2.imshow("Road Width Measurement", combined_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# out.release()
# cv2.destroyAllWindows()
