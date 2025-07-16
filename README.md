# MSc-Project

## Toolbox:
- Using all the functions in the toolbox.
- All videos used for detection are stored in a single folder.
```bash
python main.py --mode all --source_video_path 'your_folder'
```
### 1. Pedestrian Analysis
#### (0) Pedestrian Detection
https://github.com/EdgeGalaxy/YoloPedestrian
#### (1) Count the number of pedestrians in the zone (Done)
```bash
python main.py --mode count --source_video_path 'your_video'
```
https://github.com/roboflow/supervision/tree/develop/examples/count_people_in_zone
#### (2) Pedestrian Speed Estimator (Done)
```bash
python main.py --mode speed_pede --source_video_path 'your_video'
```    
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking
#### (3) Waiting time in the zone (Done)
```bash
python main.py --mode Waiting --source_video_path 'your_video'
```  
https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone
#### (4) Pedestrian tracking (Done)
```bash
python main.py --mode tracking_pede --source_video_path 'your_video'
```    
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking
#### (5) Phone usage detection (Done)
```bash
python main.py --mode phone --source_video_path 'your_video'
```  
https://github.com/HasnainAhmedO7/Detection-of-Phone-Usage-with-Computer-Vision
#### (6) Age (Done)
```bash
python main.py --mode face --source_video_path 'your_video'
```
https://github.com/serengil/deepface
#### (7) Gender (Done)
```bash
python main.py --mode face --source_video_path 'your_video'
```
https://github.com/serengil/deepface
#### (8) Race (Done)
```bash
python main.py --mode face --source_video_path 'your_video'
```
https://github.com/serengil/deepface
#### (9) Clothing type analysis (Done)
```bash
python main.py --mode clothing --source_video_path 'your_video'
```
https://github.com/ZerrnArsk/Fashion-Based-Person-Searcher
#### (10) Personal belongings (Done)
```bash
python main.py --mode belongings --source_video_path 'your_video'
```
### 2. Vehicle Analysis 
#### (1) Vehicle Type (Done)
```bash
python main.py --mode type --source_video_path pedestrian.mp4
```  
https://github.com/Srilakshmi2717/YOLO-Based-Real-Time-Vehicle-Detection-and-Classification
#### (2) Vehicle Speed (Usage)
https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation
#### (3) Distance between vehicles and vehicles (Done)
```bash
python main.py --mode car_distance --source_video_path pedestrian.mp4
```  
https://github.com/maheshpaulj/Lane_Detection
#### (4) Distance between vehicles and pedestrians (Done)
```bash
python main.py --mode pede_distance --source_video_path pedestrian.mp4
```  
https://github.com/maheshpaulj/Lane_Detection
#### (5) Lane Detection (Done)
```bash
python main.py --mode lane --source_video_path pedestrian.mp4
```  
https://github.com/maheshpaulj/Lane_Detection
### 3. Environment Analysis 
#### (1) Weather (Done)
```bash
python main.py --mode weather --source_video_path 'your_video'
```  
https://github.com/nurcanyaz/yolov8WeatherClassification
#### (2) Traffic light (Done)
```bash
python main.py --mode light --source_video_path pedestrian.mp4
```  
https://github.com/alasarerhan/Real-Time-Traffic-Light-and-Sign-Detection-with-YOLO11
#### (3) Traffic sign (Done)
```bash
python main.py --mode traffic_sign --source_video_path 'your_video'
```
https://github.com/MDhamani/Traffic-Sign-Recognition-Using-YOLO
https://github.com/KL-lovesagiri/YOLOv8_GUI_For_Traffic_Sign_Detection
https://github.com/Kartik-Aggarwal/Real-Time-Traffic-Sign-Detection
#### (4) Road Condition (Done)
```bash
python main.py --mode road_defect --source_video_path 'your_video'
```
https://github.com/oracl4/RoadDamageDetection
#### (5) Road Width (Done)
```bash
python main.py --mode width --source_video_path 'your_video'
```
https://github.com/saarthxxk/Real-Time-Road-Width-Detection/tree/main
#### (6) Day or Evening (Done)
```bash
python main.py --mode daytime --source_video_path 'your_video'
```
https://github.com/KishieKube/CV_Day_Evening_detector/tree/main
#### (7) Crosswalk (Done)
```bash
python main.py --mode crosswalk --source_video_path 'your_video'
```
https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO/tree/main

