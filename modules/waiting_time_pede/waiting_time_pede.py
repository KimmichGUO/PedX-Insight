import argparse
from typing import List
import csv
import os

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from utils.timers import FPSBasedTimer

COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)

def waiting_time_pede(
    source_video_path: str,
    weights: str,
    device: str,
    confidence: float,
    iou: float,
    classes: List[int],
    save_interval_frames: int = 30,
) -> None:
    base_name = os.path.splitext(os.path.basename(source_video_path))[0]
    csv_output_path = f"{base_name}_waiting_times.csv"

    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)
    video_info = sv.VideoInfo.from_video_path(video_path=source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)
    timer = FPSBasedTimer(video_info.fps)

    frame_count = 0

    with open(csv_output_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "data"])

    for frame in frames_generator:
        frame_count += 1

        results = model(frame, verbose=False, device=device, conf=confidence)[0]
        detections = sv.Detections.from_ultralytics(results)

        if classes:
            mask = np.isin(detections.class_id, classes)
            detections = detections[mask]

        detections = detections.with_nms(threshold=iou)
        detections = tracker.update_with_detections(detections)

        times_in_sec = timer.tick(detections)

        if frame_count % save_interval_frames == 0:
            filtered = {
                tid: round(t, 2)
                for tid, t in zip(detections.tracker_id, times_in_sec)
                if t >= 2.0
            }

            if filtered:
                with open(csv_output_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([frame_count, str(filtered)])

        annotated_frame = frame.copy()
        annotated_frame = COLOR_ANNOTATOR.annotate(annotated_frame, detections)

        labels = [
            f"#{tid} {int(t // 60):02d}:{int(t % 60):02d}"
            for tid, t in zip(detections.tracker_id, times_in_sec)
        ]
        annotated_frame = LABEL_ANNOTATOR.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels,
        )

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate and display waiting times for pedestrians.")
    parser.add_argument("--source_video_path", type=str, required=True, help="Path to source video")
    parser.add_argument("--weights", type=str, default="yolov8s.pt", help="Model weights path")
    parser.add_argument("--device", type=str, default="cpu", help="Device: cpu/cuda/mps")
    parser.add_argument("--confidence_threshold", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou_threshold", type=float, default=0.7, help="IOU threshold")
    parser.add_argument("--classes", nargs="*", type=int, default=[0], help="Classes to detect (empty=all)")
    parser.add_argument("--save_interval_frames", type=int, default=30, help="Interval (frames) to save data")

    args = parser.parse_args()

    waiting_time_pede(
        source_video_path=args.source_video_path,
        weights=args.weights,
        device=args.device,
        confidence=args.confidence_threshold,
        iou=args.iou_threshold,
        classes=args.classes,
        save_interval_frames=args.save_interval_frames,
    )
