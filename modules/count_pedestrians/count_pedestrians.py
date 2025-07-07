import json
from typing import List, Tuple
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import supervision as sv

COLORS = sv.ColorPalette.DEFAULT

def load_zones_config(file_path: str) -> List[np.ndarray]:
    with open(file_path, "r") as file:
        data = json.load(file)
        return [np.array(polygon, np.int32) for polygon in data["polygons"]]

def initiate_annotators(polygons: List[np.ndarray], resolution_wh: Tuple[int, int]):
    line_thickness = sv.calculate_optimal_line_thickness(resolution_wh=resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=resolution_wh)

    zones, zone_annotators, box_annotators = [], [], []
    for index, polygon in enumerate(polygons):
        zone = sv.PolygonZone(polygon=polygon)
        zone_annotator = sv.PolygonZoneAnnotator(
            zone=zone,
            color=COLORS.by_idx(index),
            thickness=line_thickness,
            text_thickness=line_thickness * 2,
            text_scale=text_scale * 2,
        )
        box_annotator = sv.BoxAnnotator(
            color=COLORS.by_idx(index), thickness=line_thickness
        )
        zones.append(zone)
        zone_annotators.append(zone_annotator)
        box_annotators.append(box_annotator)

    return zones, zone_annotators, box_annotators

def detect(frame: np.ndarray, model: YOLO, confidence_threshold: float = 0.5):
    results = model(frame, imgsz=1280, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    filter_by_class = detections.class_id == 0
    filter_by_confidence = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_confidence]

def annotate(frame, zones, zone_annotators, box_annotators, detections):
    annotated_frame = frame.copy()
    counts = []
    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        detections_in_zone = detections[zone.trigger(detections=detections)]
        count = len(detections_in_zone)
        counts.append(count)
        annotated_frame = zone_annotator.annotate(scene=annotated_frame)
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame, detections=detections_in_zone
        )
    return annotated_frame, counts


def count_pedestrians(
    source_video_path: str,
    zone_configuration_path: str,
    source_weights_path: str = "yolov8x.pt",
    target_video_path: str = None,
    confidence_threshold: float = 0.3,
    iou_threshold: float = 0.7,
):

    video_info = sv.VideoInfo.from_video_path(source_video_path)
    polygons = load_zones_config(zone_configuration_path)
    zones, zone_annotators, box_annotators = initiate_annotators(
        polygons=polygons, resolution_wh=video_info.resolution_wh
    )

    model = YOLO(source_weights_path)

    frames_generator = sv.get_video_frames_generator(source_video_path)

    if target_video_path is not None:
        with sv.VideoSink(target_video_path, video_info) as sink:
            for frame in tqdm(frames_generator, total=video_info.total_frames):
                detections = detect(frame, model, confidence_threshold)
                annotated_frame = annotate(frame, zones, zone_annotators, box_annotators, detections)
                sink.write_frame(annotated_frame)
    else:
        for frame in tqdm(frames_generator, total=video_info.total_frames):
            detections = detect(frame, model, confidence_threshold)
            annotated_frame, counts = annotate(frame, zones, zone_annotators, box_annotators, detections)

            # 在帧上添加人数信息
            for i, count in enumerate(counts):
                position = (30, 50 + i * 40)
                text = f"Zone {i + 1} Count: {count}"
                cv2.putText(
                    annotated_frame, text, position,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 255, 0),
                    thickness=2
                )

            print(f"Frame: Zone counts = {counts}")

            cv2.imshow("Processed Video", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cv2.destroyAllWindows()
