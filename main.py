import argparse

from modules.count_pedestrians.count_pedestrians import count_pedestrians
from modules.pedestrian_waiting.count_pedestrian_waiting_time import count_pedestrian_waiting_time

def main():
    parser = argparse.ArgumentParser(description="Pedestrian Analysis Toolbox")

    parser.add_argument(
        "--task",
        type=str,
        choices=["number", "waitingtime"],
        required=True,
        help="Choose the task to run: 'number' (count pedestrians) or 'waitingtime' (analyze waiting time).",
    )

    parser.add_argument(
        "--source_video_path",
        required=True,
        type=str,
        help="Path to the source video file",
    )
    parser.add_argument(
        "--source_weights_path",
        type=str,
        default="yolov8x.pt",
        help="Path to the YOLO weights file (default: yolov8x.pt)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.3,
        help="Model confidence threshold (default: 0.3)",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.7,
        help="IOU threshold (default: 0.7)",
    )

    # 仅当 task 为 number 时需要
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default="modules/count_pedestrians/vertical-zone-config.json",
        help="Path to the zone configuration JSON file (only for 'number' task)",
    )
    parser.add_argument(
        "--target_video_path",
        type=str,
        default=None,
        help="Path to save the processed video (optional, only for 'number' task)",
    )

    args = parser.parse_args()

    if args.task == "number":
        count_pedestrians(
            source_video_path=args.source_video_path,
            zone_configuration_path=args.zone_configuration_path,
            source_weights_path=args.source_weights_path,
            target_video_path=args.target_video_path,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
        )
    elif args.task == "waitingtime":
        count_pedestrian_waiting_time(
            source_video_path=args.source_video_path,
            weights=args.source_weights_path,
            device="cuda",  # 可改成 args.device，如果你想支持参数控制
            confidence=args.confidence_threshold,
            iou=args.iou_threshold,
            classes=[0]  # 默认追踪 pedestrian
        )

if __name__ == "__main__":
    main()
