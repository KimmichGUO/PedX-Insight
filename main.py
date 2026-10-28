import argparse
import subprocess
import sys
import os
import math





import warnings

warnings.filterwarnings("ignore", category=UserWarning, module='pkg_resources')


# Modes whose failure is reported but does not fail the batch. Now EMPTY: 'sidewalk' and
# 'pose' were promoted 2026-07-24 after succeeding on all 11 batch cities ([E9] 86.8%
# non-empty polygon rate; [P12] real pose output, median keypoint confidence 0.73-0.96).
# Leaving them soft would let a real failure exit 0 and have run.py mark the video
# finished=TRUE — the exact silent gap that left Toronto/Cincinnati without [E9]/[P12].
# 'ag' left this set 2026-07-16 (reimplemented on YuNet + genderage.onnx).
OPTIONAL_MODES = set()

# Modes that only dispatch other modes as subprocesses.
ORCHESTRATOR_MODES = {"mul_all", "single_all", "pedestrian", "vehicle", "environment"}


def main():
    parser = argparse.ArgumentParser(description="Pedestrian Analysis Toolbox")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["id_img", "waiting", "speed", "ego", "scale", "clothing", "phone", "belongings", "ag",
                 "vehicle_type", "lane", "count_vehicle",
                 "weather", "traffic_sign", "width", "light", "road_condition", "daytime", "crosswalk", "accident",
                 "sidewalk", "scene_seg",
                 "risky", "cross_pede", "crosswalk_usage", "run_red", "crossing_vehicle_count", "on_lane", "nearby",
                 "pedestrian", "vehicle", "environment",
                 "pet", "vehicle_speed", "headway", "signal_timing", "micro_events", "groups", "pose",
                 "sum_video", "sum_pede", "personal_info", "env_info",
                 "localize",
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
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="City for --mode localize, e.g. 'Ulm, Germany' (inferred from mapping.csv if omitted)",
    )
    parser.add_argument(
        "--osm_python",
        type=str,
        default=None,
        help="Interpreter of the Monocular-OSM-Localization env (for --mode localize)",
    )
    parser.add_argument("--plan", action="store_true", help="Print the dependency plan without loading models")
    parser.add_argument("--resume-failed", action="store_true", help="Reuse verified stages from the same video/configuration/code")
    args = parser.parse_args()
    if not math.isfinite(args.analysis_interval) or args.analysis_interval <= 0:
        parser.error("--analysis_interval must be finite and positive")
    from modules.pipeline import build_plan, execute_pipeline
    if args.plan:
        for mode in ([args.mode] if args.mode in {"localize", "scene_seg"} else build_plan(args.mode)):
            print(mode)
        return
    if args.mode in ORCHESTRATOR_MODES:
        if args.mode == "mul_all":
            from pathlib import Path
            directory = Path(args.source_video_path)
            if not directory.is_dir():
                parser.error(f"not a video directory: {directory}")
            videos = sorted(path for path in directory.iterdir() if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"})
        else:
            videos = [args.source_video_path]
        failed = []
        for video in videos:
            manifest = execute_pipeline(video, args.mode, args.analysis_interval, args.weights_yolo,
                                        resume=args.resume_failed)
            if manifest["status"] not in {"complete", "partial"}:
                failed.append(str(video))
        if failed:
            print(f"[FAIL] incomplete videos: {failed}", flush=True)
            sys.exit(1)
        return
    from modules.pipeline import standalone_run
    with standalone_run(args.source_video_path, args.mode):
        _run_mode(args)


def _run_mode(args):

    from modules.provenance import utc_now
    started_utc = utc_now()

    if args.mode == "id_img":
        from new_track_id_with_imgs import ultralytics_pedestrian_tracking_with_imgsave
        ultralytics_pedestrian_tracking_with_imgsave(
            video_path=args.source_video_path,
            weights=args.weights_yolo,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "waiting":
        from modules.waiting_time_pede.waiting_time_pede import run_waiting_time_analysis
        run_waiting_time_analysis(
            video_path=args.source_video_path,
        )
    elif args.mode == "speed":
        from modules.speed.speed_estimation import run_speed_estimation
        run_speed_estimation(
            video_path=args.source_video_path,
        )
    elif args.mode == "ego":
        from modules.speed.ego_motion import run_ego_motion
        run_ego_motion(
            video_path=args.source_video_path,
        )
    elif args.mode == "scale":
        from modules.speed.scale_calibration import run_scale_calibration
        run_scale_calibration(
            video_path=args.source_video_path,
        )
    elif args.mode == "ag":
        from modules.age_gender.age_gender_detect import run_age_gender
        run_age_gender(
            video_path=args.source_video_path
        )
    elif args.mode == "clothing":
        from modules.clothing.clothing import run_clothing_detection
        run_clothing_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "phone":
        from modules.phone.phone import run_phone_detection
        run_phone_detection(
            video_path=args.source_video_path,
            weights=args.weights_yolo,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "belongings":
        from modules.belongings.belongings import run_belongings_detection
        run_belongings_detection(
            video_path=args.source_video_path,
            weights=args.weights_yolo,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "weather":
        from modules.weather.weather import run_weather_detection
        run_weather_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "daytime":
        from modules.daynight.daytime import run_daytime_detection
        run_daytime_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "light":
        from modules.traffic_light.traffic_light import run_traffic_light_detection
        run_traffic_light_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "traffic_sign":
        from modules.traffic_sign.traffic_sign import run_traffic_sign
        run_traffic_sign(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "sidewalk":
        # sys.executable, and propagate the exit code: it used to be discarded, so a failed
        # sidewalk run exited 0 even after 'sidewalk' was promoted out of OPTIONAL_MODES.
        cmd = [
            sys.executable, "modules/sidewalk/sidewalk_detect.py",
            "--video", args.source_video_path,
            "--analyze_interval_sec", str(args.analysis_interval)
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)
    elif args.mode == "scene_seg":
        # [E10] Mask2Former (Mapillary Vistas) scene segmentation. Standalone mode, NOT in
        # single_all: it is a candidate replacement for the crosswalk/sidewalk detectors and
        # stays out of the production pipeline until the JAAD benchmark (validation/) says so.
        from modules.scene_seg.scene_seg import run_scene_segmentation
        run_scene_segmentation(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval,
        )
    elif args.mode == "crosswalk":
        # Segmentation-based detector (the YOLO one was at chance on JAAD; see crosswalk_seg.py).
        from modules.crosswalk.crosswalk_seg import run_crosswalk_detection
        run_crosswalk_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "accident":
        from modules.accident.accident import run_accident_scene_detection
        run_accident_scene_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "road_condition":
        from modules.road_condition.road_condition import run_road_defect_detection
        run_road_defect_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "width":
        from modules.road_width.road_width import run_road_width_analysis
        run_road_width_analysis(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "count_vehicle":
        from modules.count_vehicle.count_vehicle import vehicle_count
        vehicle_count(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "crossing_vehicle_count":
        from modules.count_vehicle.count_vehicle_when_crossing import analyze_vehicle_during_crossing
        analyze_vehicle_during_crossing(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "vehicle_type":
        from modules.type_vehicle.type_vehicle import run_vehicle_frame_analysis
        run_vehicle_frame_analysis(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "lane":
        from modules.lane_detection.lane_detection import run_lane_detection
        run_lane_detection(
            video_path=args.source_video_path,
            analyze_interval_sec=args.analysis_interval
        )
    elif args.mode == "cross_pede":
        from modules.crossing_judge.crossing_5part import detect_crossing
        detect_crossing(
            video_path=args.source_video_path
        )
    elif args.mode == "run_red":
        from modules.run_redlight.run_redlight import determine_red_light_violation
        determine_red_light_violation(
            video_path=args.source_video_path,
        )
    elif args.mode == "crosswalk_usage":
        from modules.crosswalk_usage.crosswalk_usage import determine_crosswalk_usage
        determine_crosswalk_usage(
            video_path=args.source_video_path,
        )
    elif args.mode == "risky":
        from modules.risky_crossing.risky_crossing import detect_crossing_risk
        detect_crossing_risk(
            video_path=args.source_video_path,
        )
    elif args.mode == "on_lane":
        from modules.pede_on_lane.pede_on_lane import pedestrian_on_lane
        pedestrian_on_lane(
            video_path=args.source_video_path,
        )
    elif args.mode == "nearby":
        from modules.pede_around_count.pede_around import calculate_nearby_count
        calculate_nearby_count(
            video_path=args.source_video_path,
        )
    elif args.mode == "personal_info":
        from modules.crossed_pede_info.crossed_info import extract_pedestrian_info
        extract_pedestrian_info(
            video_path=args.source_video_path,
        )
    elif args.mode == "env_info":
        from modules.crossed_pede_info.env_info import merge_env_info
        merge_env_info(
            video_path=args.source_video_path,
        )
    elif args.mode == "sum_video":
        from modules.summary.video_info import generate_video_env_stats
        generate_video_env_stats(
            video_path=args.source_video_path,
            analysis_interval=args.analysis_interval,
        )
    elif args.mode == "sum_pede":
        from modules.summary.pede_info import summary_all_info
        summary_all_info(
            video_path=args.source_video_path,
        )
    elif args.mode == "pet":
        from modules.insights.pet_conflicts import run_pet_conflicts
        run_pet_conflicts(
            video_path=args.source_video_path,
        )
    elif args.mode == "vehicle_speed":
        from modules.insights.vehicle_speed import run_vehicle_speed
        run_vehicle_speed(
            video_path=args.source_video_path,
        )
    elif args.mode == "headway":
        from modules.insights.headway_stats import run_headway_stats
        run_headway_stats(
            video_path=args.source_video_path,
        )
    elif args.mode == "signal_timing":
        from modules.insights.signal_timing import run_signal_timing
        run_signal_timing(
            video_path=args.source_video_path,
        )
    elif args.mode == "micro_events":
        from modules.insights.micro_events import run_micro_events
        run_micro_events(
            video_path=args.source_video_path,
        )
    elif args.mode == "groups":
        from modules.insights.social_groups import run_social_groups
        run_social_groups(
            video_path=args.source_video_path,
        )
    elif args.mode == "pose":
        from modules.insights.pose_behavior import run_pose_behavior
        run_pose_behavior(
            video_path=args.source_video_path,
        )
    elif args.mode == "localize":
        from modules.localization.localize import run_localization
        run_localization(
            video_path=args.source_video_path,
            city=args.city,
            osm_python=args.osm_python,
        )

    else:
        print(f"Unknown mode: {args.mode}")

    # Provenance ([A0]): which code revision and weights produced this mode's outputs.
    # Orchestrator modes are skipped -- each sub-mode they spawn records itself. Reached
    # only when the mode returned normally (a raised exception exits before this line).
    if args.mode not in ORCHESTRATOR_MODES:
        from modules.provenance import record_mode_run
        record_mode_run(args.source_video_path, args.mode, started_utc, parameters=vars(args))



if __name__ == "__main__":
    main()
