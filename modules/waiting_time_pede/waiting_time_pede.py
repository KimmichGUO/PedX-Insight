import cv2
from typing import List, Dict
from collections import defaultdict

import numpy as np
from ultralytics import YOLO
import supervision as sv

from utils.general import find_in_list


COLORS = sv.ColorPalette.from_hex(["#E6194B", "#3CB44B", "#FFE119", "#3C76D1"])
COLOR_ANNOTATOR = sv.ColorAnnotator(color=COLORS)
LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=COLORS, text_color=sv.Color.from_hex("#000000")
)


def count_pedestrian_waiting_time(
    source_video_path: str,
    weights: str = "yolov8s.pt",
    device: str = "cpu",
    confidence: float = 0.3,
    iou: float = 0.7,
    classes: List[int] = [0],
) -> None:
    model = YOLO(weights)
    tracker = sv.ByteTrack(minimum_matching_threshold=0.5)

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    frames_generator = sv.get_video_frames_generator(source_video_path)
  
    id_to_frame_count: Dict[int, int] = defaultdict(int)

    for frame in frames_generator:
        results = model(frame, verbose=False, device=device, conf=confidence)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[find_in_list(detections.class_id, classes)]
        detections = detections.with_nms(threshold=iou)
        detections = tracker.update_with_detections(detections)

        annotated_frame = frame.copy()

        for tracker_id in detections.tracker_id:
            id_to_frame_count[tracker_id] += 1

        annotated_frame = COLOR_ANNOTATOR.annotate(scene=annotated_frame, detections=detections)

        labels = [
            f"#{tracker_id} {int((id_to_frame_count[tracker_id]) / video_info.fps):02d}s"
            for tracker_id in detections.tracker_id
        ]
        annotated_frame = LABEL_ANNOTATOR.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )

        cv2.imshow("Pedestrian Waiting Time (Full Video)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
