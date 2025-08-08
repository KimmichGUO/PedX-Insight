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
https://github.com/roboflow/supervision/tree/develop/examples/count_people_in_zone
#### (2) Pedestrian Speed Estimator
```bash
python main.py --mode speed_pede --source_video_path 'your_video'
```    
Result: [P2]pedestrian_speed.csv  
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking

#### (3) Waiting time in the zone
```bash
python main.py --mode Waiting --source_video_path 'your_video'
```  
Result: [P3]waiting_time.csv  
https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone
#### (4) Pedestrian tracking
```bash
python main.py --mode tracking_pede --source_video_path 'your_video'
```    
Result: [P4]pedestrian_tracking.csv  
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking
#### (5) Phone usage detection
```bash
python main.py --mode phone --source_video_path 'your_video'
```  
Result: [P5]phone_usage.csv   
https://github.com/HasnainAhmedO7/Detection-of-Phone-Usage-with-Computer-Vision  

#### (6) Age
```bash
python main.py --mode face --source_video_path 'your_video'
```
Result: [P6]age_gender_race.csv    
https://github.com/serengil/deepface
#### (7) Gender
```bash
python main.py --mode gender --source_video_path 'your_video'
```
Result: [P7]pedestrian_gender.csv  
https://github.com/Sklyvan/Age-Gender-Prediction
#### (8) Race
```bash
python main.py --mode face --source_video_path 'your_video'
```
Result: [P6]age_gender_race.csv  
https://github.com/serengil/deepface
#### (9) Clothing type analysis 
```bash
python main.py --mode clothing --source_video_path 'your_video'
```
Result: [P8]clothing.csv  
https://github.com/ZerrnArsk/Fashion-Based-Person-Searcher
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
https://github.com/Srilakshmi2717/YOLO-Based-Real-Time-Vehicle-Detection-and-Classification
#### (2) Vehicle Speed 
```bash
python main.py --mode speed --source_video_path 'your_video'
```  
Result: [V2]vehicle_speed.csv  
https://github.com/WZS666/Yolov5_DeepSort_SpeedEstimate
#### (3) Distance between vehicles and vehicles 
```bash
python main.py --mode car_distance --source_video_path 'your_video'
```  
Result: [V3]distance_ve_ve.csv  
https://github.com/maheshpaulj/Lane_Detection
#### (4) Distance between vehicles and pedestrians 
```bash
python main.py --mode pede_distance --source_video_path 'your_video'
```  
Result: [V4]distance_ve_pe.csv  
https://github.com/maheshpaulj/Lane_Detection
#### (5) Lane Detection
```bash
python main.py --mode lane --source_video_path 'your_video'
```  
Result: [V5]lane_detection.csv  
https://github.com/maheshpaulj/Lane_Detection
#### (6) Different types of Vehicle Count 
```bash
python main.py --mode count_vehicle --source_video_path 'your_video'
```  
Result: [V6]vehicle_count.csv  
https://github.com/Srilakshmi2717/YOLO-Based-Real-Time-Vehicle-Detection-and-Classification

### 3. Environment Analysis 
#### (1) Weather
```bash
python main.py --mode weather --source_video_path 'your_video'
```  
Result: [E1]weather.csv  
https://github.com/nurcanyaz/yolov8WeatherClassification
#### (2) Traffic light
```bash
python main.py --mode light --source_video_path pedestrian.mp4
```  
Result: [E2]traffic_light.csv  
https://github.com/alasarerhan/Real-Time-Traffic-Light-and-Sign-Detection-with-YOLO11
#### (3) Traffic sign 
```bash
python main.py --mode traffic_sign --source_video_path 'your_video'
```
Result: [E3]traffic_sign.csv  
https://github.com/MDhamani/Traffic-Sign-Recognition-Using-YOLO
https://github.com/KL-lovesagiri/YOLOv8_GUI_For_Traffic_Sign_Detection
https://github.com/Kartik-Aggarwal/Real-Time-Traffic-Sign-Detection
#### (4) Road Condition
```bash
python main.py --mode road_condition --source_video_path 'your_video'
```
Result: [E4]road_condition.csv  
https://github.com/oracl4/RoadDamageDetection
#### (5) Road Width
```bash
python main.py --mode width --source_video_path 'your_video'
```
Result: [E5]road_width.csv  
https://github.com/saarthxxk/Real-Time-Road-Width-Detection
#### (6) Day or Evening 
```bash
python main.py --mode daytime --source_video_path 'your_video'
```
Result: [E6]daytime.csv  
https://github.com/KishieKube/CV_Day_Evening_detector
#### (7) Crosswalk 
```bash
python main.py --mode crosswalk --source_video_path 'your_video'
```
Result: [E7]crosswalk_detection.csv  
https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO
#### (8) Accident
```bash
python main.py --mode accident --source_video_path 'your_video'
```
Result: [E8]accident_detection.csv  
https://github.com/RoadEyeProject/RoadEye_model
#### (9) Sidewalk
```bash
python main.py --mode sidewalk --source_video_path 'your_video'
```
Result: [E9]sidewalk_detection.csv  
https://github.com/shaikhubaidahmed/Vehicle_Collision_Detection  
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
### 5. Combination Analysis for the city level
#### (1) Extract video info about environment
```bash
python main.py --mode env_info --source_video_path 'your_video'
```
Result: [A1]video_info.csv  
This function is used to extract and summary the information of whole video