import os
import cv2
import pandas as pd
from collections import Counter, defaultdict
from ultralytics import YOLO
import supervision as sv


def run_vehicle_frame_analysis(video_path, weights="modules/type_vehicle/best.pt", output_csv_path=None):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[V1]vehicle_type.csv ")

    model = YOLO(weights)

    cap = cv2.VideoCapture(video_path)
    class_name_list = list(model.model.names.values())
    box_annotator = sv.BoxAnnotator()

    results_list = []
    frame_id = -1

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_id += 1

        result = model.predict(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        frame_counter = Counter(detections.class_id)
        stat = defaultdict(int)

        for class_id, count in frame_counter.items():
            class_name = model.model.names[class_id]
            stat[class_name] = count

        for class_name in class_name_list:
            stat[class_name] = stat.get(class_name, 0)

        stat["frame_id"] = frame_id
        stat["total"] = sum(stat[class_name] for class_name in class_name_list)

        results_list.append(stat)

    cap.release()

    df = pd.DataFrame(results_list)
    columns = ["frame_id"] + sorted(class_name_list) + ["total"]
    df = df[columns]
    df.to_csv(output_csv_path, index=False)
    print(f"Vehicle frame-level statistics saved to {output_csv_path}")
# import cv2
# import numpy as np
# from collections import Counter
# from ultralytics import YOLO
# import supervision as sv


# def type_vehicle_analysis(
#         source_video_path: str,
#         weights: str = "best.pt",
#         confidence: float = 0.3,
#         iou: float = 0.7,
#         device: str = "cpu",
#         classes: list = [],
#         show: bool = True,
#         target_video_path: str = None
# ) -> None:
#     # 加载模型
#     model = YOLO(weights)
#     model.fuse()
#     model.to(device)

#     cap = cv2.VideoCapture(source_video_path)
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     if target_video_path:
#         fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#         out = cv2.VideoWriter(target_video_path, fourcc, fps, (width, height))
#     else:
#         out = None

#     total_counter = Counter()
#     box_annotator = sv.BoxAnnotator()

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         results = model.predict(
#             source=frame,
#             conf=confidence,
#             iou=iou,
#             classes=classes,
#             device=device,
#             verbose=False
#         )

#         result = results[0]
#         detections = sv.Detections.from_ultralytics(result)

#         frame_counter = Counter(detections.class_id)
#         total_counter.update(frame_counter)

#         labels = [
#             f"{model.model.names[class_id]} {conf:.2f}"
#             for class_id, conf in zip(detections.class_id, detections.confidence)
#         ]

#         annotated_frame = box_annotator.annotate(
#             scene=frame.copy(),
#             detections=detections,
#             #labels=labels
#         )

#         if show:
#             cv2.imshow("Vehicle Type Detection", annotated_frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         if out:
#             out.write(annotated_frame)

#     cap.release()
#     if out:
#         out.release()
#     cv2.destroyAllWindows()

#     # print("\n Vehicle Class Statistics:")
#     # for class_id, count in total_counter.items():
#     #     print(f"  {model.model.names[class_id]}: {count}")
