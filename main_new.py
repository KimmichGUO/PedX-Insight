import argparse
from modules.count_pedestrians.count_pedestrians import count_pedestrians
from modules.waiting_time_pede.waiting_time_pede import run_waiting_time_analysis
from modules.tracking_pede.tracking_pede import run_pede_direction_analysis
from modules.type_vehicle.type_vehicle import type_vehicle_analysis
from modules.traffic_analysis.traffic_analysis import run_traffic_analysis
from modules.age_gender.age_gender import run_age_gender_detection
from modules.weather.weather import run_weather_detection
from modules.race.race import run_race_detection
from modules.traffic_total.traffic_total import run_traffic_total_detection
from modules.traffic_light.traffic_light import traffic_light
# from modules.traffic_sign.traffic_sign import traffic_sign_detection
from modules.head_phone.head_phone import run_head_detection
from modules.daynight.daytime import run_daytime_detection
from modules.crosswalk.crosswalk import run_crosswalk_detection
from modules.speed_pedestrian.speed_pedestrian import run_pede_speed_estimation
import subprocess
import os
from modules.face.face import run_face_analysis
from modules.phone.phone import run_phone_detection
from modules.clothing.clothing import run_clothing_detection
from modules.belongings.belongings import run_belongings_detection

def main():
    def run_mode(mode, video_path, extra_args=""):
        cmd = f"python main.py --mode {mode} --source_video_path \"{video_path}\" {extra_args}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)
    parser = argparse.ArgumentParser(description="Pedestrian Analysis Toolbox")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["count", "waiting", "tracking_pede", "type", "speed_pede","traffic", "agegender", "belongings", "weather", "phone", "clothing", "face", "total", "light", "head", "daytime", "crosswalk","all"],
        help="Choose the analysis mode: 'count', 'waiting', 'tracking', 'type', 'traffic', or 'agegender'",
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
        default="yolov8n.pt",
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
        default=[0],
        help="List of class IDs to detect (e.g. 0 for person). Leave empty to detect all classes.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8m-pose.pt",
        help="Weights file for tracking mode (default: yolov8m-pose.pt)",
    )
    parser.add_argument(
        "--weights_yolo",
        type=str,
        default="yolov8n.pt",
        help="Weights file for tracking mode",
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
        run_waiting_time_analysis(
            video_path=args.source_video_path,
        )
    elif args.mode == "tracking_pede":
        run_pede_direction_analysis(
            video_path=args.source_video_path,
        )
    elif args.mode == "speed_pede":
        run_pede_speed_estimation(
            video_path=args.source_video_path,
        )
    elif args.mode == "type":
        type_vehicle_analysis(
            source_video_path=args.source_video_path,
            weights=args.source_weights_path,
            confidence=args.confidence_threshold,
            iou=args.iou_threshold,
            device=args.device,
            classes=args.classes,
            show=True,
            target_video_path=args.target_video_path
        )
    elif args.mode == "traffic":
        run_traffic_analysis(
            source_video_path=args.source_video_path,
            source_weights_path=args.source_weights_path,
            target_video_path=args.target_video_path,
            confidence_threshold=args.confidence_threshold,
            iou_threshold=args.iou_threshold,
        )
    elif args.mode == "agegender":
        run_age_gender_detection(args.source_video_path)

    elif args.mode == "weather":
        run_weather_detection(
            source_video_path=args.source_video_path
        )
    elif args.mode == "clothing":
        run_clothing_detection(
            video_path=args.source_video_path
        )
    elif args.mode == "face":
        # run_race_detection(
        #     source_video_path=args.source_video_path
        # )
        run_face_analysis(video_path=args.source_video_path)
    elif args.mode == "total":
        run_traffic_total_detection(
            source_video_path=args.source_video_path,
            weights=args.source_weights_path,
            target_video_path=args.target_video_path
        )
    elif args.mode == "light":
        traffic_light(
            source_video_path=args.source_video_path,
        )
    elif args.mode == "phone":
        run_phone_detection(
            video_path=args.source_video_path,
            weights = args.weights_yolo
        )
    elif args.mode == "belongings":
        run_belongings_detection(
            video_path=args.source_video_path,
            weights = args.weights_yolo
        )
    # elif args.mode == "sign":
    #     traffic_sign_detection(
    #         source_video_path=args.source_video_path,
    #         weights=args.source_weights_path,
    #         confidence=args.confidence_threshold,
    #         iou=args.iou_threshold,
    #         device=args.device,
    #         classes=args.classes,
    #         target_video_path=args.target_video_path
    #     )
    elif args.mode == "head":
        run_head_detection(
            # source_video_path=args.source_video_path
            video_path=args.source_video_path
        )
    elif args.mode == "daytime":
        run_daytime_detection(
            source_video_path=args.source_video_path
        )
    elif args.mode == "crosswalk":
        run_crosswalk_detection(
            source_video_path=args.source_video_path,
            conf=args.confidence_threshold
        )
    # elif args.mode == "all":
    #     subprocess.run("python main.py --mode count --source_video_path pedestrian.mp4", shell=True)
    #     subprocess.run("python main.py --mode count --source_video_path pedestrian.mp4", shell=True)
    elif args.mode == "all":
        video_dir = args.source_video_path
        if not os.path.isdir(video_dir):
            print(f"Error: {video_dir} is not a valid directory.")
            return

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)

            run_mode("count", video_path)
            run_mode("waiting", video_path)
            run_mode("tracking_pede", video_path)
            run_mode("speed_pede",video_path)
            run_mode("clothing", video_path)
            run_mode("phone", video_path)
            run_mode("belongings", video_path)
            run_mode("face", video_path)
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
