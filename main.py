import argparse
import subprocess
import os

# 1.1
from new_track_id_with_imgs import ultralytics_pedestrian_tracking_with_imgsave
# 1.2
from modules.waiting_time_pede.waiting_time_pede import run_waiting_time_analysis
# 1.3
from modules.age_gender.age_gender_detect import run_age_gender
# 1.4
from modules.clothing.clothing import run_clothing_detection
# 1.5
from modules.phone.phone import run_phone_detection
# 1.6
from modules.belongings.belongings import run_belongings_detection
# 1.7
from modules.weather.weather import run_weather_detection
# 1.8
from modules.daynight.daytime import run_daytime_detection
# 1.9
from modules.traffic_light.traffic_light import run_traffic_light_detection
# 1.10
from modules.traffic_sign.traffic_sign import run_traffic_sign
# 1.12
from modules.crosswalk.crosswalk import run_crosswalk_detection
# 1.13
from modules.accident.accident import run_accident_scene_detection
# 1.14
from modules.road_condition.road_condition import run_road_defect_detection
# 1.15
from modules.road_width.road_width import run_road_width_analysis
# 1.16
from modules.type_vehicle.type_vehicle import run_vehicle_frame_analysis
from modules.count_vehicle.count_vehicle import vehicle_count
from modules.count_vehicle.count_vehicle_when_crossing import analyze_vehicle_during_crossing
# 1.17
from modules.lane_detection.lane_detection import run_lane_detection

# 2.1
from modules.crossing_judge.crossing import detect_crossing
# 2.2
from modules.run_redlight.run_redlight import determine_red_light_violation
# 2.3
from modules.crosswalk_usage.crosswalk_usage import determine_crosswalk_usage
# 2.4
from modules.risky_crossing.risky_crossing import detect_crossing_risk
# 2.5
from modules.pede_on_lane.pede_on_lane import pedestrian_on_lane
# 2.6
from modules.pede_around_count.pede_around import calculate_nearby_count

# 3
from modules.crossed_pede_info.crossed_info import extract_pedestrian_info
from modules.crossed_pede_info.env_info import merge_env_info
from modules.summary.video_info import generate_video_env_stats
from modules.summary.pede_info import summary_all_info

import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')


def main():
    def run_mode(mode, video_path, analysis_interval=1.0, weights="yolo11n.pt"):
        cmd = f"python main.py --mode {mode} --source_video_path \"{video_path}\" --analysis_interval {analysis_interval} --weights_yolo \"{weights}\""
        cmd_print = f"{mode} using video \"{video_path}\""
        print(f"Running: {cmd_print}")
        subprocess.run(cmd, shell=True)

    parser = argparse.ArgumentParser(description="Pedestrian Analysis Toolbox")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["id_img", "waiting", "clothing", "phone", "belongings", "ag",
                 "vehicle_type", "lane", "count_vehicle",
                 "weather", "traffic_sign", "width", "light", "road_condition", "daytime", "crosswalk", "accident",
                 "sidewalk",
                 "risky", "cross_pede", "crosswalk_usage", "run_red", "crossing_vehicle_count", "on_lane", "nearby",
                 "pedestrian", "vehicle", "environment",
                 "sum_video", "sum_pede", "personal_info", "env_info",
                 "mul_all", "single_all"
                 ],
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
        default="yolo11n.pt",
        help="Weights file for tracking mode",
    )
    parser.add_argument(
        "--analysis_interval",
        type=float,
        default=1.0,
        help="Analysis interval second",
    )
    args = parser.parse_args()

    if args.mode == "id_img":
        ultralytics_pedestrian_tracking_with_imgsave(
            video_path=args.source_video_path,
            weights=args.weights_yolo,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "waiting":
        run_waiting_time_analysis(
            video_path=args.source_video_path,
        )
    elif args.mode == "ag":
        run_age_gender(
            video_path=args.source_video_path
        )
    elif args.mode == "clothing":
        run_clothing_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "phone":
        run_phone_detection(
            video_path=args.source_video_path,
            weights=args.weights_yolo,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "belongings":
        run_belongings_detection(
            video_path=args.source_video_path,
            weights=args.weights_yolo,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "weather":
        run_weather_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "daytime":
        run_daytime_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "light":
        run_traffic_light_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "traffic_sign":
        run_traffic_sign(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "sidewalk":
        cmd = [
            "python", "modules/sidewalk/sidewalk_detect.py",
            "--video", args.source_video_path,
            "--analyze_interval_sec", str(args.analysis_interval)
        ]
        subprocess.run(cmd)
    elif args.mode == "crosswalk":
        run_crosswalk_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "accident":
        run_accident_scene_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "road_condition":
        run_road_defect_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "width":
        run_road_width_analysis(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "count_vehicle":
        vehicle_count(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "crossing_vehicle_count":
        analyze_vehicle_during_crossing(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "vehicle_type":
        run_vehicle_frame_analysis(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "lane":
        run_lane_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "cross_pede":
        detect_crossing(
            video_path=args.source_video_path
        )
    elif args.mode == "run_red":
        determine_red_light_violation(
            video_path=args.source_video_path,
        )
    elif args.mode == "crosswalk_usage":
        determine_crosswalk_usage(
            video_path=args.source_video_path,
        )
    elif args.mode == "risky":
        detect_crossing_risk(
            video_path=args.source_video_path,
        )
    elif args.mode == "on_lane":
        pedestrian_on_lane(
            video_path=args.source_video_path,
        )
    elif args.mode == "nearby":
        calculate_nearby_count(
            video_path=args.source_video_path,
        )
    elif args.mode == "personal_info":
        extract_pedestrian_info(
            video_path=args.source_video_path,
        )
    elif args.mode == "env_info":
        merge_env_info(
            video_path=args.source_video_path,
        )
    elif args.mode == "sum_video":
        generate_video_env_stats(
            video_path=args.source_video_path,
        )
    elif args.mode == "sum_pede":
        summary_all_info(
            video_path=args.source_video_path,
        )

    elif args.mode == "mul_all":
        video_dir = args.source_video_path
        if not os.path.isdir(video_dir):
            print(f"Error: {video_dir} is not a valid directory.")
            return

        video_files = [f for f in os.listdir(video_dir) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]

        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            print(f"Processing {video_path} ...")
            # basic function
            # pedestrian
            run_mode("id_img", video_path, args.analysis_interval, args.weights_yolo)
            run_mode("waiting", video_path, args.analysis_interval)
            run_mode("phone", video_path, args.analysis_interval, args.weights_yolo)
            run_mode("ag", video_path)
            run_mode("clothing", video_path, args.analysis_interval)
            run_mode("belongings", video_path, args.analysis_interval, args.weights_yolo)

            # vehicle
            run_mode("vehicle_type", video_path, args.analysis_interval)
            run_mode("lane", video_path, args.analysis_interval)
            run_mode("count_vehicle", video_path, args.analysis_interval)

            # environment
            run_mode("weather", video_path, args.analysis_interval)
            run_mode("light", video_path, args.analysis_interval)
            run_mode("traffic_sign", video_path, args.analysis_interval)
            run_mode("road_condition", video_path, args.analysis_interval)
            run_mode("width", video_path, args.analysis_interval)
            run_mode("daytime", video_path, args.analysis_interval)
            run_mode("crosswalk", video_path, args.analysis_interval)
            run_mode("accident", video_path, args.analysis_interval)
            run_mode("sidewalk", video_path, args.analysis_interval)

            # advanced
            run_mode("cross_pede", video_path)
            run_mode("risky", video_path)
            run_mode("crosswalk_usage", video_path)
            run_mode("run_red", video_path)
            run_mode("crossing_vehicle_count", video_path, args.analysis_interval)
            run_mode("on_lane", video_path)
            run_mode("nearby", video_path)

            # summary
            run_mode("personal_info", video_path)
            run_mode("env_info", video_path)
            run_mode("sum_video", video_path)
            run_mode("sum_pede", video_path)

    elif args.mode == "single_all":
        video_path = args.source_video_path
        # basic function
        # pedestrian
        run_mode("id_img", video_path, args.analysis_interval, args.weights_yolo)
        run_mode("waiting", video_path, args.analysis_interval)
        run_mode("phone", video_path, args.analysis_interval, args.weights_yolo)
        run_mode("ag", video_path)
        run_mode("clothing", video_path, args.analysis_interval)
        run_mode("belongings", video_path, args.analysis_interval, args.weights_yolo)

        # vehicle
        run_mode("vehicle_type", video_path, args.analysis_interval)
        run_mode("lane", video_path, args.analysis_interval)
        run_mode("count_vehicle", video_path, args.analysis_interval)

        # environment
        run_mode("weather", video_path, args.analysis_interval)
        run_mode("light", video_path, args.analysis_interval)
        run_mode("traffic_sign", video_path, args.analysis_interval)
        run_mode("road_condition", video_path, args.analysis_interval)
        run_mode("width", video_path, args.analysis_interval)
        run_mode("daytime", video_path, args.analysis_interval)
        run_mode("crosswalk", video_path, args.analysis_interval)
        run_mode("accident", video_path, args.analysis_interval)
        run_mode("sidewalk", video_path, args.analysis_interval)

        # advanced
        run_mode("cross_pede", video_path)
        run_mode("risky", video_path)
        run_mode("crosswalk_usage", video_path)
        run_mode("run_red", video_path)
        run_mode("crossing_vehicle_count", video_path, args.analysis_interval)
        run_mode("on_lane", video_path)
        run_mode("nearby", video_path)

        # summary
        run_mode("personal_info", video_path)
        run_mode("env_info", video_path)
        run_mode("sum_video", video_path)
        run_mode("sum_pede", video_path)

    elif args.mode == "pedestrian":
        video_path = args.source_video_path
        # basic function
        # pedestrian
        run_mode("id_img", video_path, args.analysis_interval, args.weights_yolo)
        run_mode("waiting", video_path, args.analysis_interval)
        run_mode("phone", video_path, args.analysis_interval, args.weights_yolo)
        run_mode("ag", video_path)
        run_mode("clothing", video_path, args.analysis_interval)
        run_mode("belongings", video_path, args.analysis_interval, args.weights_yolo)

    elif args.mode == "vehicle":
        video_path = args.source_video_path
        run_mode("vehicle_type", video_path, args.analysis_interval)
        run_mode("lane", video_path, args.analysis_interval)
        run_mode("count_vehicle", video_path, args.analysis_interval)

    elif args.mode == "environment":
        video_path = args.source_video_path
        run_mode("weather", video_path, args.analysis_interval)
        run_mode("light", video_path, args.analysis_interval)
        run_mode("traffic_sign", video_path, args.analysis_interval)
        run_mode("road_condition", video_path, args.analysis_interval)
        run_mode("width", video_path, args.analysis_interval)
        run_mode("daytime", video_path, args.analysis_interval)
        run_mode("crosswalk", video_path, args.analysis_interval)
        run_mode("accident", video_path, args.analysis_interval)
        run_mode("sidewalk", video_path, args.analysis_interval)

    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()