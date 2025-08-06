import argparse
# 0
from track_id_pedestrians import run_pedestrian_tracking
from track_id_pedestrians_with_imgsave import run_pedestrian_tracking_with_imgsave
# 1.1
from modules.count_pedestrians.count_pedestrians import pedestrian_count
# 1.2
from modules.speed_pedestrian.speed_pedestrian import run_pede_speed_estimation
# 1.3
from modules.waiting_time_pede.waiting_time_pede import run_waiting_time_analysis
# 1.4
from modules.tracking_pede.tracking_pede import run_pede_direction_analysis
# 1.5
from modules.phone.phone import run_phone_detection
# 1.6 ~ 1.8
from modules.face.face import run_face_analysis
# 1.6_new
from modules.gender.gender import gender_analysis
# 1.9
from modules.clothing.clothing import run_clothing_detection
# 1.10
from modules.belongings.belongings import run_belongings_detection
# 2.1
from modules.type_vehicle.type_vehicle import run_vehicle_frame_analysis
# 2.2
from modules.speed_estimate.track import run_speed_estimate
# 2.3
from modules.distance_vehicle.distance_vehicle import run_car_detection_with_distance
# 2.4
from modules.distance_pedestrian.distance_pedestrian import visualize_and_estimate_distance
# 2.5
from modules.lane_detection.lane_detection import run_lane_detection
# 3.1
from modules.weather.weather import run_weather_detection
# 3.2
from modules.traffic_light.traffic_light import run_traffic_light_detection
# 3.3
from modules.traffic_sign.traffic_sign import run_traffic_sign
# 3.4
from modules.road_condition.road_condition import run_road_defect_detection
# 3.5
from modules.road_width.road_width import run_road_width_analysis
# 3.6
from modules.daynight.daytime import run_daytime_detection
# 3.7
from modules.crosswalk.crosswalk import run_crosswalk_detection
# 3.8
from modules.accident.accident import run_accident_scene_detection

from modules.count_vehicle.count_vehicle import vehicle_count

import subprocess
import os

from modules.risky_crossing.risky_crossing import detect_crossing_risk
from modules.acceleration_pede.acceleration import analyze_acceleration
from modules.crossing_judge.crossing import detect_crossing
from modules.crosswalk_usage.crosswalk_usage import determine_crosswalk_usage
from modules.run_redlight.run_redlight import determine_red_light_violation
from modules.count_vehicle.count_vehicle_when_crossing import analyze_vehicle_during_crossing
from modules.crossed_pede_info.crossed_info import extract_pedestrian_info
from modules.pede_on_lane.pede_on_lane import pedestrian_on_lane

import numpy as np
np.float = float


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
        choices=["id", "id_img","count", "waiting", "tracking_pede", "speed_pede", "clothing", "phone", "belongings", "face","gender",
                 "vehicle_type", "car_distance", "pede_distance", "lane", "speed", "count_vehicle",
                 "weather", "traffic_sign", "width", "light", "road_defect", "daytime", "crosswalk", "accident", "sidewalk",
                 "risky", "acc", "cross_pede", "crosswalk_usage", "run_red", "crossing_vehicle_count", "personal_info", "on_lane",
                 "all", "pedestrian", "vehicle", "environment"],
        help="Choose the analysis mode",
    )
    parser.add_argument(
        "--source_video_path",
        required=True,
        type=str,
        help="Path to the source video file",
    )
    parser.add_argument(
        "--weights_yolo",
        type=str,
        default="yolov8n.pt",
        help="Weights file for tracking mode",
    )
    args = parser.parse_args()

    if args.mode == "count":
        pedestrian_count(
            video_path=args.source_video_path,
        )
    elif args.mode == "id":
        run_pedestrian_tracking(
            video_path=args.source_video_path,
        )
    elif args.mode == "id_img":
        run_pedestrian_tracking_with_imgsave(
            video_path=args.source_video_path,
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
    elif args.mode == "speed":
        run_speed_estimate(
            source=args.source_video_path,
        )
    elif args.mode == "acc":
        analyze_acceleration(
            video_path=args.source_video_path,
        )
    elif args.mode == "traffic_sign":
        run_traffic_sign(
            video_path=args.source_video_path,
        )
    elif args.mode == "risky":
        detect_crossing_risk(
            video_path=args.source_video_path,
        )
    elif args.mode == "weather":
        run_weather_detection(
            video_path=args.source_video_path
        )
    elif args.mode == "clothing":
        run_clothing_detection(
            video_path=args.source_video_path
        )
    elif args.mode == "face":
        run_face_analysis(
            video_path=args.source_video_path
        )
    elif args.mode == "gender":
        gender_analysis(
            video_path=args.source_video_path
        )
    elif args.mode == "light":
        run_traffic_light_detection(
            video_path=args.source_video_path,
        )
    elif args.mode == "defect":
        run_road_defect_detection(
            video_path=args.source_video_path
        )
    elif args.mode == "width":
        run_road_width_analysis(
            video_path=args.source_video_path
        )
    elif args.mode == "car_distance":
        run_car_detection_with_distance(
            video_path=args.source_video_path
        )
    elif args.mode == "count_vehicle":
        vehicle_count(
            video_path=args.source_video_path
        )
    elif args.mode == "crossing_vehicle_count":
        analyze_vehicle_during_crossing(
            video_path=args.source_video_path
        )
    elif args.mode == "sidewalk":
        cmd = [
            "python", "modules/sidewalk/sidewalk_detect.py",
            "--video", args.source_video_path
        ]
        subprocess.run(cmd)
    elif args.mode == "cross_pede":
        detect_crossing(
            video_path=args.source_video_path
        )
    elif args.mode == "pede_distance":
        visualize_and_estimate_distance(
            video_path=args.source_video_path
        )
    elif args.mode == "lane":
        run_lane_detection(
            video_path=args.source_video_path
        )
    elif args.mode == "vehicle_type":
        run_vehicle_frame_analysis(
            video_path=args.source_video_path
        )
    elif args.mode == "phone":
        run_phone_detection(
            video_path=args.source_video_path,
            weights=args.weights_yolo
        )
    elif args.mode == "belongings":
        run_belongings_detection(
            video_path=args.source_video_path,
            weights=args.weights_yolo
        )
    elif args.mode == "daytime":
        run_daytime_detection(
            video_path=args.source_video_path
        )
    elif args.mode == "crosswalk":
        run_crosswalk_detection(
            video_path=args.source_video_path,
        )
    elif args.mode == "crosswalk_usage":
        determine_crosswalk_usage(
            video_path=args.source_video_path,
        )
    elif args.mode == "accident":
        run_accident_scene_detection(
            video_path=args.source_video_path,
        )
    elif args.mode == "run_red":
        determine_red_light_violation(
            video_path=args.source_video_path,
        )
    elif args.mode == "personal_info":
        extract_pedestrian_info(
            video_path=args.source_video_path,
        )
    elif args.mode == "on_lane":
        pedestrian_on_lane(
            video_path=args.source_video_path,
        )
    elif args.mode == "all":
        video_dir = args.source_video_path
        if not os.path.isdir(video_dir):
            print(f"Error: {video_dir} is not a valid directory.")
            return

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)

            # pedestrian
            run_mode("id", video_path)
            run_mode("count", video_path)
            run_mode("waiting", video_path)
            run_mode("tracking_pede", video_path)
            run_mode("speed_pede",video_path)
            run_mode("clothing", video_path)
            run_mode("phone", video_path)
            run_mode("belongings", video_path)
            run_mode("face", video_path)
            # vehicle
            run_mode("vehicle_type", video_path)
            run_mode("car_distance", video_path)
            run_mode("pede_distance", video_path)
            run_mode("lane", video_path)
            # environment
            run_mode("weather", video_path)
            run_mode("traffic_sign", video_path)
            run_mode("width", video_path)
            run_mode("light", video_path)
            run_mode("road_defect", video_path)
            run_mode("daytime", video_path)
            run_mode("crosswalk", video_path)
            run_mode("accident", video_path)

    elif args.mode == "pedestrian":
        video_dir = args.source_video_path
        if not os.path.isdir(video_dir):
            print(f"Error: {video_dir} is not a valid directory.")
            return

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)

            # pedestrian
            run_mode("id", video_path)
            run_mode("count", video_path)
            run_mode("waiting", video_path)
            run_mode("tracking_pede", video_path)
            run_mode("speed_pede",video_path)
            run_mode("clothing", video_path)
            run_mode("phone", video_path)
            run_mode("belongings", video_path)
            run_mode("face", video_path)

    elif args.mode == "vehicle":
        video_dir = args.source_video_path
        if not os.path.isdir(video_dir):
            print(f"Error: {video_dir} is not a valid directory.")
            return

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)

            run_mode("vehicle_type", video_path)
            run_mode("car_distance", video_path)
            run_mode("pede_distance", video_path)
            run_mode("lane", video_path)


    elif args.mode == "environment":
        video_dir = args.source_video_path
        if not os.path.isdir(video_dir):
            print(f"Error: {video_dir} is not a valid directory.")
            return

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)

            run_mode("weather", video_path)
            run_mode("traffic_sign", video_path)
            run_mode("width", video_path)
            run_mode("light", video_path)
            run_mode("road_defect", video_path)
            run_mode("daytime", video_path)
            run_mode("crosswalk", video_path)
            run_mode("accident", video_path)
    else:
        print(f"Unknown mode: {args.mode}")



if __name__ == "__main__":
    main()