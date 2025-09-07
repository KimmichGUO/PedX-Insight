# PedX-Insight: A Toolkit for Automated Analysis of Global Pedestrian Crossing Behavior

## Toolkit:
| Argument              | Description                                       | Required     | Default        |
| --------------------- |---------------------------------------------------|--------------|----------------|
| `--mode`              | Analysis mode                                     | Yes          | None           |
| `--source_video_path` | Path to the input video/dictionary                | Yes          | None           |
| `--analysis_interval` | Analysis interval in seconds (sampling frequency) | No (Optinal) | `1.0`          |
| `--weights_yolo`      | Path to YOLO weights file                         | No (Optinal) | `"yolo11n.pt"` |

###(1) Analyze multiple videos in a single folder using all the functions in the Toolkit.
```bash
python main.py --mode mul_all --source_video_path PATH/TO/DIR --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```
###(2) Analyze one video using all the functions in the Toolkit.
```bash
python main.py --mode single_all --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```

### 1. Basic Funtions (Pedestrian Analysis)
#### (1) Detect and Track Pedestrians
```bash
python main.py --mode id_img --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```
Result: [B1]tracked_pedestrians.csv
#### (2) Phone usage detection
```bash
python main.py --mode phone --source_video_path PATH/TO/VIDEO --analysis_interval 1.0 --weights_yolo "yolo11n.pt" 
```  
Result: [P5]phone_usage.csv    

#### (3) Age and Gender
```bash
python main.py --mode ag --source_video_path PATH/TO/VIDEO
```
Result: [P6]age_gender_race.csv    

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

## Dataset
https://github.com/Shaadalam9/pedestrians-in-youtube

## Run
```bash
python run.py --start_row 1 --start_step 1
```
-start_row: Specifies the row number to start processing from, default=1.  
-start_step: Specifies the processing step to start from (useful for resuming interrupted runs), default=1.  
Possible values:

    1 : Download the video  
    2 : Analyze the video and save the results  
    3 : Delete the video  