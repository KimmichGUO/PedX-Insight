# PedX-Insight: A Toolkit for Automated Analysis of Global Pedestrian Crossing Behavior

## Installation & Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies are pinned to their **newest** releases (July 2026), including the numpy 2.5.x and
OpenCV 5.x majors. Install CUDA builds of `torch`/`torchvision` from the appropriate PyTorch
index for your platform if you want GPU inference.

> ⚠️ **Age/gender (`--mode ag`) is currently disabled.** It relies on
> [paddlex](https://github.com/PaddlePaddle/PaddleX), which hard-caps `numpy<2.4` and pins
> `opencv-contrib-python==4.10.0.84` — incompatible with the newest numpy/OpenCV. paddlex was
> therefore removed from `requirements.txt`. `main.py` and every other mode still work; invoking
> `--mode ag` raises a clear, actionable error. To re-enable it, install paddlex in a **separate**
> environment (with `numpy<2.4` and `opencv-contrib-python==4.10.0.84`) and point that module at
> it, or reimplement the module on another model. See [`AGENTS.md`](AGENTS.md).

## Toolkit:
| Argument              | Description                                       | Required     | Default        |
| --------------------- |---------------------------------------------------|--------------|----------------|
| `--mode`              | Analysis mode                                     | Yes          | None           |
| `--source_video_path` | Path to the input video/dictionary                | Yes          | None           |
| `--analysis_interval` | Analysis interval in seconds (sampling frequency) | No (Optinal) | `1.0`          |
| `--weights_yolo`      | Path to YOLO weights file                         | No (Optinal) | `"yolo11n.pt"` |

### (1) Analyze multiple videos in a single folder using all the functions in the Toolkit.
```bash
python main.py --mode mul_all --source_video_path PATH/TO/DIR --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```
### (2) Analyze one video using all the functions in the Toolkit.
```bash
python main.py --mode single_all --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```

### 1. Basic Funtions (Pedestrian Analysis)
#### (1) Detect and Track Pedestrians
```bash
python main.py --mode id_img --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```
Result: [B1]tracked_pedestrians.csv (1 Hz, for the appearance/environment modules) **and**
[B2]dense_tracks.csv (dense foot-point trajectory, for the kinematic modules).
The tracker now runs **densely** (default `tracking_fps=15`) so ByteTrack association stays
valid for moving pedestrians, while [B1] is still down-sampled to `analysis_interval`. Set
`tracking_fps<=0` in `ultralytics_pedestrian_tracking_with_imgsave` to fall back to 1-rate tracking.

#### (1b) Pedestrian speed (measured)
```bash
python main.py --mode speed --source_video_path PATH/TO/VIDEO
```
Result: [S1]pedestrian_speed.csv
Measures each pedestrian's walking/net speed in **m/s** from their foot-point trajectory (prefers
[B2], falls back to [B1]). Pixel→metre scale uses a per-pedestrian height prior (bbox height ÷ the
city's `avg_height` from `mapping.csv`, fallback 1.70 m). Per-track columns include `walking_speed_mps`,
`net_speed_mps`, quality fields, and a `reliable` flag. This replaces the previous behaviour where
`crossing_speed` was a city-level constant imported from an external CSV, never measured from the video.
The per-video median flows into `[A1]video_info.csv` as `measured_avg_walking_speed_mps`.
> Note: within-video/relative speed is trustworthy; absolute cross-city m/s is approximate on
> uncalibrated monocular footage — gate on the `reliable` flag.

#### (1c) Camera ego-motion
```bash
python main.py --mode ego --source_video_path PATH/TO/VIDEO
```
Result: [B3]ego_motion.csv
Estimates the per-frame global background translation (Lucas-Kanade on background features, with the
pedestrian boxes from [B2] masked out) and accumulates it into a camera position. `speed` and `waiting`
subtract it when the camera is actually moving (handheld / dashcam / pan), so pedestrian motion is not
confounded with camera motion. Static-camera videos are left untouched.

#### (1d) Ground-plane scale calibration
```bash
python main.py --mode scale --source_video_path PATH/TO/VIDEO
```
Result: [S2]scale_calibration.csv
Recovers a real ground-plane metric scale from **crosswalk stripe periodicity**: a zebra crossing is a
periodic ground pattern of known real period (default 1.0 m = 0.5 m stripe + 0.5 m gap), so its period in
pixels gives pixels-per-metre at that image depth. Samples across depths are fit to `scale(y) = a*y + b`
(pixels-per-metre vs image row). [S1] prefers this over the height prior when the fit quality is `good`,
and falls back automatically otherwise.
> `stripe_period_m` is a country-dependent assumption — a wrong period scales every speed proportionally.
> The assumed period and a `quality` flag are recorded in [S2] for auditability.

**Scale priority used by [S1]:** stripe ground plane ([S2], ~2-5%) → per-pedestrian height prior
(`avg_height` from mapping.csv, ~10-20%) → lane-width cross-check ([V5]).

#### (2) Phone usage detection
```bash
python main.py --mode phone --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```  
Result: [P5]phone_usage.csv    

#### (3) Age and Gender
```bash
python main.py --mode ag --source_video_path PATH/TO/VIDEO
```
Result: [P6]age_gender.csv  
> ⚠️ Requires `paddlex`, which is **not** installed by default (see [Installation & Environment](#installation--environment)). This mode is disabled until paddlex is provided in a compatible environment.

#### (4) Clothing type analysis 
```bash
python main.py --mode clothing  --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```
Result: [P8]clothing.csv  

#### (5) Personal belongings 
```bash
python main.py --mode belongings  --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```
Result: [P9]pedestrian_belongings.csv   
### 2. Basic Funtions (Vehicle Analysis) 
#### (1) Vehicle Type
```bash
python main.py --mode vehicle_type  --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```  
Result: [V1]vehicle_type.csv

#### (2) Lane Detection
```bash
python main.py --mode lane --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```  
Result: [V5]lane_detection.csv  

#### (3) Different types of Vehicle Count 
```bash
python main.py --mode count_vehicle --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```  
Result: [V6]vehicle_count.csv

### 3. Basic Funtions (Environment Analysis) 
#### (1) Weather
```bash
python main.py --mode weather --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```  
Result: [E1]weather.csv  

#### (2) Traffic light
```bash
python main.py --mode light --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```  
Result: [E2]traffic_light.csv  

#### (3) Traffic sign 
```bash
python main.py --mode traffic_sign --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E3]traffic_sign.csv  

#### (4) Road Condition
```bash
python main.py --mode road_condition --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E4]road_condition.csv  

#### (5) Road Width
```bash
python main.py --mode width --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E5]road_width.csv  

#### (6) Day or Evening 
```bash
python main.py --mode daytime --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E6]daytime.csv  

#### (7) Crosswalk 
```bash
python main.py --mode crosswalk --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E7]crosswalk_detection.csv  

#### (8) Accident
```bash
python main.py --mode accident --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E8]accident_detection.csv  

#### (9) Sidewalk
```bash
python main.py --mode sidewalk --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```
Result: [E9]sidewalk_detection.csv  
Weights should be downloaded from https://drive.usercontent.google.com/download?id=1X1uKaGENEBZamF6tOfx9eKLTIQLsBN5h&export=download&authuser=0

### 4. Advanced Funtions
#### (1) Risky crossing analysis 
```bash
python main.py --mode risky --source_video_path PATH/TO/VIDEO
```
Result: [C1]risky_crossing.csv  
This function is used to detect whether pedestrians cross the street in a risky way based on the detection of the traffic light, the traffic sign, and the crosswalk.
#### (2) Determine whether a pedestrian has crossed the road 
```bash
python main.py --mode cross_pede --source_video_path PATH/TO/VIDEO
```
Result: [C3]crossing_judge.csv  
This function is used to Determine and record whether each pedestrian in the video has crossed the road.
#### (3) Determine whether a pedestrian has used the crosswalk when crossing
```bash
python main.py --mode crosswalk_usage --source_video_path PATH/TO/VIDEO
```
Result: [C4]crosswalk_usage.csv  
This function is used to analyse whether a pedestrian use the crosswalk or not besed on the fact that he/she has crossed the street.
#### (4) Detect red light runner
```bash
python main.py --mode run_red --source_video_path PATH/TO/VIDEO
```
Result: [C5]red_light_runner.csv  
This function is used to analyse whether a pedestrian run the red light or not besed on the fact that he/she has crossed the street.
#### (5) Vehicle Count when crossing
```bash
python main.py --mode crossing_vehicle_count --source_video_path PATH/TO/VIDEO
```
Result: [C6]crossing_ve_count.csv  
This function is used to analyse how many different types of vehicles there are when a pedestrian crosses the street.
#### (6) Extract crossed pedestrian information
```bash
python main.py --mode personal_info --source_video_path PATH/TO/VIDEO
```
Result: [C7]crossing_pe_info.csv  
This function is used to extract the information of pedestrians who have crossed the streets, including the gender, the clothing type, and the personal belongings.
#### (7) Pedestrian on lane
```bash
python main.py --mode on_lane --source_video_path PATH/TO/VIDEO
```
Result: [C8]pedestrian_on_lane.csv  
This function is used to analyse whether a pedestrian walks too close to a vehicle.
#### (8) Extract crossed environment information
```bash
python main.py --mode env_info --source_video_path PATH/TO/VIDEO
```
Result: [C9]crossing_env_info.csv  
This function is used to extract the information of environment when a pedestrian has crossed the streets, including weather, daytime, accident or not, road condition.
#### (9) Nearby pedestrian count
```bash
python main.py --mode nearby --source_video_path PATH/TO/VIDEO
```
Result: [C10]nearby_count.csv  
This function is used to count how many people are around pedestrians who are crossing the road.

### 5. Summary Functions
#### (1) Extract all video information
```bash
python main.py --mode sum_video --source_video_path PATH/TO/VIDEO
```
Result: [A1]video_info.csv  
This function is used to extract and summary the information of whole video.
#### (2) Extract all crossed pedestrians information
```bash
python main.py --mode sum_pede --source_video_path PATH/TO/VIDEO
```
Result: [A2]pedestrian_info.csv   
This function is used to extract and summary the information of all crossed pedestrians from whole video.

### 6. Novel Behavioral Insights
Six insight modules consume the CSVs above (no video needed once the producers ran; all are
part of `single_all`/`mul_all` and unit-tested under `tests/`):

| Mode | Output | What it measures |
|------|--------|------------------|
| `pet` | [I1]pet_conflicts.csv | **Post-encroachment time** pedestrian↔vehicle surrogate-safety conflicts (PET < 1.5 s = severe); pure time gap, robust to scale error |
| `vehicle_speed` | [V8]vehicle_speed.csv | Per-vehicle **metric speed profiles** (median/p85/max, at-crosswalk vs mid-block) with reliable flag |
| `headway` | [V11]headway_stats.csv | **Time-headway distribution** at the counting line: shifted-exponential fit, platoon fraction, flow |
| `signal_timing` | [P10]signal_timing.csv | **Phase-relative crossing starts**: anticipatory starts, startup latency, red-clearance exposure |
| `micro_events` | [P11]micro_events.csv | **Curb-dance hesitation**: aborted starts, mid-crossing freezes, evasive speed bursts |
| `groups` | [I2]/[I3] group CSVs | **Social groups & platooning**: co-moving clusters, leader/follower launch lags, group-vs-solo |

Supporting sidecars written by the producers: `[B0]video_meta.csv` (fps/width/height, survives video
deletion), `[V7]vehicle_tracks.csv` (dense per-frame vehicle trajectories), `[V10]line_crossing_events.csv`
(vehicle counting-line events; also reused by `crossing_vehicle_count` to skip its duplicate GPU pass).

## Dataset
https://github.com/Shaadalam9/pedestrians-in-youtube

## Run
```bash
python run.py --start_row 1 --start_step 1
```
| Argument       | Description                                                                 | Required     | Default |
|----------------|-----------------------------------------------------------------------------|--------------|---------|
| `--start_row`  | Specifies the row number to start processing from                           | No (Optional) | `1`     |
| `--start_step` | Specifies the processing step to start from (useful for resuming runs)      | No (Optional)         | `1`     |

**Possible values for `--start_step`:**

| Value | Meaning                        |
|-------|--------------------------------|
| `1`   | Download the video             |
| `2`   | Analyze the video and save results |
| `3`   | Delete the video               |

## Video Geolocation (Localization)

`--mode localize` estimates **where** a video was filmed (WGS84 latitude/longitude) from the
footage plus the city name, by wrapping the companion project
[Monocular-OSM-Localization](https://github.com/M-Colley/Monocular-OSM-Localization) — vendored
as a git submodule at `external/Monocular-OSM-Localization`, **pinned to its
[`0.1.0`](https://github.com/M-Colley/Monocular-OSM-Localization/releases/tag/0.1.0) release**.

> **Not a pip dependency.** Even at 0.1.0 the tool ships no package metadata (no `setup.py` /
> `pyproject.toml`, and it is not on PyPI), so it cannot be `pip install`ed. It stays vendored
> as a git submodule and is run as a **subprocess** using a configurable interpreter (resolved
> from `--osm_python`, then `$OSM_LOCALIZATION_PYTHON`, then the submodule's own `.venv`, then —
> new — PedX's own interpreter if the tool's deps are importable there).

0.1.0's pins (`numpy>=2.5`, `opencv-python>=4.13`, `osmnx`, …) are compatible with PedX's
environment, so the simplest setup is to install the tool's requirements **into PedX's own venv**
and let `localize` run with no separate interpreter — the closest thing to using it "as a
dependency". Pick one:

```bash
git submodule update --init external/Monocular-OSM-Localization   # checks out the 0.1.0 tag

# A) reuse PedX's venv (recommended; then --osm_python is not needed):
pip install -r external/Monocular-OSM-Localization/requirements.txt

# B) or give the tool a dedicated venv:
python -m venv external/Monocular-OSM-Localization/.venv
external/Monocular-OSM-Localization/.venv/Scripts/pip install -r external/Monocular-OSM-Localization/requirements.txt

# ffmpeg must also be on PATH in either case
```
Run:
```bash
python main.py --mode localize --source_video_path PATH/TO/VIDEO --city "Ulm, Germany"
# batch (localizes each video after analysis, before deletion):
python run.py --localize
# aggregate all per-video results for the Visualizer:
python get_all_video_locations.py     # -> summary_data/all_video_locations.csv
```
Result: `[L1]localization.csv` per video (columns `lat`, `lon`, `confidence_level`,
`confidence_spread_m`, `street_names`, `status`, `candidates`). `--city` is inferred from
`mapping.csv` when omitted. The aggregated `summary_data/all_video_locations.csv` is imported
into PedX-Visualizer with `node scripts/import-video-coordinates.js` (see that repo's
`VIDEO_COORDINATES_SETUP.md`) so the Globe shows each video at its REAL estimated position.

> Localization is opt-in (`--localize` for `run.py`; not part of `single_all` / `mul_all`):
> it needs the video file present, the Monocular-OSM-Localization environment, and network.

## Method for Adding New Modules

You can extend the toolbox by adding your own analysis module.  
Follow these steps:

### Step 1: Create the module
Build your function and save it in `./modules/NAME_OF_FUNCTION/`.  
For example, if you want to add a `emotion_estimation` function, you might create: `./modules/emotion/emotion_estimation.py`  

### Step 2: Import the function in `main.py`
Open `main.py` and add your function to the import section, e.g.:
```python
from modules.emotion.emotion_estimation import run_emotion_estimation
```
### Step 3: Add the mode to argparse

Find the parser.add_argument("--mode", ...) section in main.py.

Add your new mode to the choices list, for example:
```python
choices=["id_img", "waiting", ..., "emotion"]
```

### Step 4: Add the execution logic

In the if args.mode == ... block of main.py, add a new condition for your function:
```python
elif args.mode == "emotion":
    run_emotion_detection(
        video_path=args.source_video_path,
        analyze_interval_sec=args.analysis_interval
    )
```

### Step 5: Run your new module

You can now call your new function from the command line:
```bash
python main.py --mode emotion --source_video_path PATH/TO/VIDEO --analysis_interval 1.0
```

## Method for Analyzing the Final Results

``` bash
python get_all_pede_info.py
python get_all_video_info.py
python statistics_with_pdf_save.py
```