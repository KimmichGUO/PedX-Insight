import argparse
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

def tracking_pede(
        source_video_path: str,
        weights: str = "yolov8m-pose.pt",
) -> None:
    model = YOLO(weights)
    edge_annotator = sv.EdgeAnnotator()
    vertex_annotator = sv.VertexAnnotator()
    box_annotator = sv.BoxAnnotator()
    tracker = sv.ByteTrack()
    smoother = sv.DetectionsSmoother()
    trace_annotator = sv.TraceAnnotator()

    def callback(frame: np.ndarray, _: int) -> np.ndarray:
        results = model(frame)[0]
        key_points = sv.KeyPoints.from_ultralytics(results)
        detections = key_points.as_detections()
        detections = tracker.update_with_detections(detections)
        detections = smoother.update_with_detections(detections)

        annotated_frame = edge_annotator.annotate(frame.copy(), key_points=key_points)
        annotated_frame = vertex_annotator.annotate(annotated_frame, key_points=key_points)
        annotated_frame = box_annotator.annotate(annotated_frame, detections=detections)
        annotated_frame = trace_annotator.annotate(annotated_frame, detections=detections)

        cv2.imshow("Processed Video", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            import os
            os._exit(0)

        return annotated_frame

    sv.process_video(
        source_path=source_video_path,
        target_path="result.mp4",
        callback=callback
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose tracking on a video using YOLO and Supervision.")
    parser.add_argument(
        "--source_video_path", type=str, required=True, help="Path to input video."
    )
    parser.add_argument(
        "--weights", type=str, default="yolov8m-pose.pt", help="YOLO pose weights file."
    )
    args = parser.parse_args()

    tracking_pede(
        source_video_path=args.source_video_path,
        weights=args.weights,
    )
