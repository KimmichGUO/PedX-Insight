import os
import cv2
import csv
import sys
from time import time as t
from datetime import timedelta
from collections import deque
from elements.SGD import SGDepth_Model
from elements.asset import apply_mask
from SGDepth.arguments import InferenceEvaluationArguments
import math

# ---------------------------
# 参数
# ---------------------------
opt = InferenceEvaluationArguments().parse()

depth_seg_estimator = SGDepth_Model(opt.disp_detector)

cap = cv2.VideoCapture(opt.video)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
fps = math.ceil(fps) if fps > 0 else 30

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ---------------------------
# 输出设置
# ---------------------------
video_name = os.path.splitext(os.path.basename(opt.video))[0]
sidewalk_csv_path = os.path.join('./analysis_results', video_name, '[E9]sidewalk_detection.csv')
os.makedirs(os.path.dirname(sidewalk_csv_path), exist_ok=True)

csvfile = open(sidewalk_csv_path, 'w', newline='')
writer = csv.writer(csvfile)
writer.writerow(['frame_id', 'polygons'])

if opt.save:
    output_video_folder = os.path.join('outputs', video_name)
    os.makedirs(output_video_folder, exist_ok=True)
    if len(opt.output_name.split('.')) == 1:
        opt.output_name += '.mp4'
    output_video_name = os.path.join(output_video_folder, opt.output_name)
    if opt.save_frames:
        output_frames_folder = os.path.join(output_video_folder, 'frames')
        os.makedirs(output_frames_folder, exist_ok=True)
    out = cv2.VideoWriter(output_video_name,
                          cv2.VideoWriter_fourcc(*'mp4v'),
                          opt.outputfps, (w, h))

sidewalk_class_id = 1
MIN_AREA_THRESHOLD = 500
frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    if frame_num % 60 != 0:
        continue

    tc = t()
    main_frame = frame.copy()

    depth, seg_img = depth_seg_estimator.inference(frame)

    sidewalk_mask = (seg_img == sidewalk_class_id).astype('uint8') * 255
    contours, _ = cv2.findContours(sidewalk_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons_for_frame = []

    for cnt in contours:
        if cv2.contourArea(cnt) < MIN_AREA_THRESHOLD:
            continue

        epsilon = 0.01 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        seg_h, seg_w = seg_img.shape[:2]
        frame_h, frame_w = frame.shape[:2]
        if seg_w != frame_w or seg_h != frame_h:
            scale_x = frame_w / seg_w
            scale_y = frame_h / seg_h
            approx = approx.astype('float32')
            approx[:, 0, 0] *= scale_x
            approx[:, 0, 1] *= scale_y
            approx = approx.astype('int32')

        polygon = approx.reshape(-1, 2)
        polygon_str = ';'.join([f"{x},{y}" for x, y in polygon])
        polygons_for_frame.append(polygon_str)

        cv2.polylines(frame, [approx], isClosed=True, color=(0, 255, 0), thickness=2)

    if polygons_for_frame:
        writer.writerow([frame_num, '|'.join(polygons_for_frame)])
    else:
        writer.writerow([frame_num, ""])

    if opt.mode != 'night':
        frame = apply_mask(frame, seg_img, mode=opt.mode)

    t2 = t()
    fps_real = 1 / (t2 - tc)
    estimated_time = str(timedelta(seconds=(frame_count - frame_num) / fps_real)).split('.')[0]

    if opt.fps:
        cv2.putText(frame, f"FPS : {fps_real:.2f}", (40, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), thickness=2)

    if opt.save:
        out.write(frame)
        if opt.save_frames:
            cv2.imwrite(os.path.join(output_frames_folder, f'{frame_num:04d}.jpg'), frame)

    noshow = 1
    if not noshow:
        cv2.imshow('Sidewalk Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    sys.stdout.write(f"\r[Input Video: {opt.video}] [{frame_num}/{frame_count} Analyzed Frames] [FPS: {fps_real:.2f}] [ET: {estimated_time}]")
    sys.stdout.flush()

cap.release()
csvfile.close()
if opt.save:
    out.release()
if not opt.noshow:
    cv2.destroyAllWindows()

print(f"\nSidewalk detection results saved to {sidewalk_csv_path}")
