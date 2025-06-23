import argparse
from modules.count_pedestrians import count_pedestrians
from modules.estimate_speed import estimate_speed
from modules.estimate_waiting_time import estimate_waiting_time
from modules.keypoint_tracking import track_keypoints
from modules.phone_usage_detection import detect_phone_usage

def main():
    parser = argparse.ArgumentParser(description="Pedestrian Analysis Toolbox")

    parser.add_argument('--video_path', type=str, required=True, help='Path to input video')
    parser.add_argument('--zone_config', type=str, default='zone.json', help='Zone config file')

    parser.add_argument('--count', action='store_true', help='Enable pedestrian counting')
    parser.add_argument('--speed', action='store_true', help='Enable speed estimation')
    parser.add_argument('--wait', action='store_true', help='Enable waiting time estimation')
    parser.add_argument('--keypoints', action='store_true', help='Enable keypoint tracking')
    parser.add_argument('--phone', action='store_true', help='Enable phone usage detection')

    args = parser.parse_args()

    if args.count:
        count_pedestrians(args.video_path, args.zone_config)

    if args.speed:
        estimate_speed(args.video_path, args.zone_config)

    if args.wait:
        estimate_waiting_time(args.video_path, args.zone_config)

    if args.keypoints:
        track_keypoints(args.video_path)

    if args.phone:
        detect_phone_usage(args.video_path)

if __name__ == '__main__':
    main()
