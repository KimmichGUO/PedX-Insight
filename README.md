# MSc-Project

## Toolbox:
- Using all the functions in the toolbox.
- All videos used for detection are stored in a single folder.
```bash
python main.py --mode all --source_video_path 'your_folder'
```
### 1. Pedestrian Analysis
#### (0) Weights for Pedestrian Detection (Optional)
https://github.com/EdgeGalaxy/YoloPedestrian
#### (1) Count the number of pedestrians in the zone
```bash
python main.py --mode count --source_video_path 'your_video'
```
Result: [P1]pedestrian_count.csv  

#### (2) Pedestrian Speed Estimator
```bash
python main.py --mode speed_pede --source_video_path 'your_video'
```    
Result: [P2]pedestrian_speed.csv

#### (3) Waiting time in the zone
```bash
python main.py --mode Waiting --source_video_path 'your_video'
```  
Result: [P3]waiting_time.csv  

#### (4) Pedestrian tracking
```bash
python main.py --mode tracking_pede --source_video_path 'your_video'
```    
Result: [P4]pedestrian_tracking.csv  

#### (5) Phone usage detection
```bash
python main.py --mode phone --source_video_path 'your_video'
```  
Result: [P5]phone_usage.csv    

#### (6) Age
```bash
python main.py --mode face --source_video_path 'your_video'
```
Result: [P6]age_gender_race.csv    

#### (7) Gender
```bash
python main.py --mode gender --source_video_path 'your_video'
```
Result: [P7]pedestrian_gender.csv  

#### (8) Race
```bash
python main.py --mode face --source_video_path 'your_video'
```
Result: [P6]age_gender_race.csv  

#### (9) Clothing type analysis 
```bash
python main.py --mode clothing --source_video_path 'your_video'
```
Result: [P8]clothing.csv  

#### (10) Personal belongings 
```bash
python main.py --mode belongings --source_video_path 'your_video'
```
Result: [P9]pedestrian_belongings.csv   
### 2. Vehicle Analysis 
#### (1) Vehicle Type
```bash
python main.py --mode vehicle_type --source_video_path 'your_video'
```  
Result: [V1]vehicle_type.csv  

#### (2) Vehicle Speed 
```bash
python main.py --mode speed --source_video_path 'your_video'
```  
Result: [V2]vehicle_speed.csv  

#### (3) Distance between vehicles and vehicles 
```bash
python main.py --mode car_distance --source_video_path 'your_video'
```  
Result: [V3]distance_ve_ve.csv  

#### (4) Distance between vehicles and pedestrians 
```bash
python main.py --mode pede_distance --source_video_path 'your_video'
```  
Result: [V4]distance_ve_pe.csv  

#### (5) Lane Detection
```bash
python main.py --mode lane --source_video_path 'your_video'
```  
Result: [V5]lane_detection.csv  

#### (6) Different types of Vehicle Count 
```bash
python main.py --mode count_vehicle --source_video_path 'your_video'
```  
Result: [V6]vehicle_count.csv

### 3. Environment Analysis 
#### (1) Weather
```bash
python main.py --mode weather --source_video_path 'your_video'
```  
Result: [E1]weather.csv  

#### (2) Traffic light
```bash
python main.py --mode light --source_video_path pedestrian.mp4
```  
Result: [E2]traffic_light.csv  

#### (3) Traffic sign 
```bash
python main.py --mode traffic_sign --source_video_path 'your_video'
```
Result: [E3]traffic_sign.csv  

#### (4) Road Condition
```bash
python main.py --mode road_condition --source_video_path 'your_video'
```
Result: [E4]road_condition.csv  

#### (5) Road Width
```bash
python main.py --mode width --source_video_path 'your_video'
```
Result: [E5]road_width.csv  

#### (6) Day or Evening 
```bash
python main.py --mode daytime --source_video_path 'your_video'
```
Result: [E6]daytime.csv  

#### (7) Crosswalk 
```bash
python main.py --mode crosswalk --source_video_path 'your_video'
```
Result: [E7]crosswalk_detection.csv  

#### (8) Accident
```bash
python main.py --mode accident --source_video_path 'your_video'
```
Result: [E8]accident_detection.csv  

#### (9) Sidewalk
```bash
python main.py --mode sidewalk --source_video_path 'your_video'
```
Result: [E9]sidewalk_detection.csv
Weights should be downloaded from https://drive.usercontent.google.com/download?id=1X1uKaGENEBZamF6tOfx9eKLTIQLsBN5h&export=download&authuser=0

### 4. Combination Analysis on Pedestrians
#### (1) Risky crossing analysis 
```bash
python main.py --mode risky --source_video_path 'your_video'
```
Result: [C1]risky_crossing.csv  
This function is used to detect whether pedestrians cross the street in a risky way based on the detection of the traffic light, the traffic sign, and the crosswalk
#### (2) Trend in pedestrians speed when crossing
```bash
python main.py --mode acc --source_video_path 'your_video'
```
Result: [C2]pede_speed_trend.csv  
This function is used to analyze whether pedestrians accelerate, move at a constant speed, or decelerate while crossing the street.
#### (3) Determine whether a pedestrian has crossed the road 
```bash
python main.py --mode cross_pede --source_video_path 'your_video'
```
Result: [C3]crossing_judge.csv  
This function is used to Determine and record whether each pedestrian in the video has crossed the road.
#### (4) Determine whether a pedestrian has used the crosswalk when crossing
```bash
python main.py --mode crosswalk_usage --source_video_path 'your_video'
```
Result: [C4]crosswalk_usage.csv  
This function is used to analyse whether a pedestrian use the crosswalk or not besed on the fact that he/she has crossed the street.
#### (5) Detect red light runner
```bash
python main.py --mode run_red --source_video_path 'your_video'
```
Result: [C5]red_light_runner.csv  
This function is used to analyse whether a pedestrian run the red light or not besed on the fact that he/she has crossed the street.
#### (6) Vehicle Count when crossing
```bash
python main.py --mode crossing_vehicle_count --source_video_path 'your_video'
```
Result: [C6]crossing_ve_count.csv  
This function is used to analyse how many different types of vehicles there are when a pedestrian crosses the street.
#### (7) Extract crossed pedestrian information
```bash
python main.py --mode personal_info --source_video_path 'your_video'
```
Result: [C7]crossing_pe_info.csv  
This function is used to extract the information of pedestrians who have crossed the streets, including the gender, the clothing type, and the personal belongings.
#### (8) Pedestrian on lane
```bash
python main.py --mode on_lane --source_video_path 'your_video'
```
Result: [C8]pedestrian_on_lane.csv  
This function is used to analyse whether a pedestrian walks too close to a vehicle.
#### (9) Extract crossed environment information
```bash
python main.py --mode env_info --source_video_path 'your_video'
```
Result: [C9]crossing_env_info.csv  
This function is used to extract the information of environment when a pedestrian has crossed the streets, including weather, daytime, accident or not, road condition.
#### (10) Nearby pedestrian count
```bash
python main.py --mode nearby --source_video_path 'your_video'
```
Result: [C10]nearby_count.csv  
This function is used to count how many people are around pedestrians who are crossing the road.

### 5. Combination Analysis at the city level
#### (1) Extract all video information
```bash
python main.py --mode sum_video --source_video_path 'your_video'
```
Result: [A1]video_info.csv  
This function is used to extract and summary the information of whole video.
#### (2) Extract all crossed pedestrians information
```bash
python main.py --mode sum_pede --source_video_path 'your_video'
```
Result: [A2]pedestrian_info.csv 
This function is used to extract and summary the information of all crossed pedestrians from whole video.

## Dataset
https://github.com/Shaadalam9/pedestrians-in-youtube