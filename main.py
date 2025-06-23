import argparse
from modules.count_pedestrians.count_pedestrians import count_pedestrians
from modules.waiting_time_pede.waiting_time_pede import waiting_time_pede
from modules.tracking_pede.tracking_pede import tracking_pede

def main():
    parser = argparse.ArgumentParser(description="Pedestrian Analysis Toolbox")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["count", "waiting", "tracking"],  # 新增tracking选项
        help="Choose the analysis mode: 'count', 'waiting', or 'tracking'",
    )
    parser.add_argument(
        "--zone_configuration_path",
        type=str,
        default="modules/count_pedestrians/vertical-zone-config.json",
        help="Path to the zone configuration JSON file (only used in 'count' mode)",
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        type=str,
        help="Path to the source video file",
    )
    parser.add_argument(
        "--target_video_path",
        type=str,
        default=None,
        help="Path to save the processed video (optional, if supported by the mode)",
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
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on: 'cpu', 'cuda', or 'mps'. Default is 'cpu'.",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        type=int,
        default=[],
        help="List of class IDs to detect (e.g. 0 for person). Leave empty to detect all classes.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8m-pose.pt",
        help="Weights file for tracking mode (default: yolov8m-pose.pt)",
    )

    args = parser.parse_args()

    if args.mode == "count":
        count_pedestrians(
            source_video_path=args.source_video_path,
            zone_configuration_path=args.zone_configuration_path,
            source_weights_path=args.source_weights_path,
            target_video_path=args.target_video_path,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
        )
    elif args.mode == "waiting":
        waiting_time_pede(
            source_video_path=args.source_video_path,
            weights=args.source_weights_path,
            device=args.device,
            confidence=args.confidence_threshold,
            iou=args.iou_threshold,
            classes=args.classes,
        )
    elif args.mode == "tracking":
        tracking_pede(
            source_video_path=args.source_video_path,
            #target_video_path=args.target_video_path,
            weights=args.weights,
        )
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()
