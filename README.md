# MSc-Project

## Toolbox:
- Using all the functions in the toolbox.
- All videos used for detection are stored in a single folder.
```bash
python main.py --mode all --source_video_path 'your_folder'
```
### 1. Pedestrian Analysis 
#### (1) Count the number of pedestrians in the zone (Done)
```bash
python main.py --mode count --source_video_path 'your_video'
```
https://github.com/roboflow/supervision/tree/develop/examples/count_people_in_zone
#### (2) Pedestrian Speed Estimator (Done)
```bash
python main.py --mode speed --source_video_path 'your_video'
```    
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking
#### (3) Waiting time in the zone (Done)
```bash
python main.py --mode Waiting --source_video_path 'your_video'
```  
https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone
#### (4) Pedestrian tracking (Done)
```bash
python main.py --mode tracking --source_video_path 'your_video'
```    
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking
#### (5) Pedestrian phone usage detection (Done)
```bash
python main.py --mode head --source_video_path 'your_video'
```  
https://github.com/HasnainAhmedO7/Detection-of-Phone-Usage-with-Computer-Vision
#### (6) Age and gender (Done)
```bash
python main.py --mode face --source_video_path 'your_video'
```
https://github.com/smahesh29/Gender-and-Age-Detection  
#### (7) Race (Done)
```bash
python main.py --mode face --source_video_path 'your_video'
```
https://github.com/serengil/deepface
#### (8) Head direction (Done)
```bash
python main.py --mode head --source_video_path 'your_video'
```  
https://github.com/HasnainAhmedO7/Detection-of-Phone-Usage-with-Computer-Vision
### 2. Vehicle Analysis 
#### (1) Traffic Analysis (Available)
```bash
python main.py --mode traffic --source_video_path pedestrian.mp4
```  
https://github.com/roboflow/supervision/tree/develop/examples/traffic_analysis  
Weights should be downloaded from https://drive.google.com/uc?id=1y-IfToCjRXa3ZdC1JpnKRopC7mcQW-5z  
#### (2) Vehicle Type (Available)
```bash
python main.py --mode type --source_video_path pedestrian.mp4
```  
https://github.com/Srilakshmi2717/YOLO-Based-Real-Time-Vehicle-Detection-and-Classification
#### (3) Vehicle Speed (Usage)
https://github.com/roboflow/supervision/tree/develop/examples/speed_estimation
#### (4) Distance between vehicles and vehicles (Usage)
https://github.com/maheshpaulj/Lane_Detection
#### (5) Distance between vehicles and pedestrians (Usage)
https://github.com/maheshpaulj/Lane_Detection
### 3. Environment Analysis 
#### (1) Weather (Available)
```bash
python main.py --mode weather --source_video_path pedestrian.mp4
```  
https://github.com/berkgulay/weather-prediction-from-image
#### (2) Traffic light (Available)
```bash
python main.py --mode light --source_video_path pedestrian.mp4
```  
https://github.com/alasarerhan/Real-Time-Traffic-Light-and-Sign-Detection-with-YOLO11
#### (3) All scenes detection (Available)
```bash
python main.py --mode total --source_video_path pedestrian.mp4
```  
https://github.com/alasarerhan/Real-Time-Traffic-Light-and-Sign-Detection-with-YOLO11
#### (4) Traffic sign (Usable)
https://github.com/Kartik-Aggarwal/Real-Time-Traffic-Sign-Detection/tree/main
#### (5) Road Width (Usable)
https://github.com/saarthxxk/Real-Time-Road-Width-Detection/tree/main
#### (6) Day or Evening (Available)
```bash
python main.py --mode daytime --source_video_path pedestrian.mp4
```
https://github.com/KishieKube/CV_Day_Evening_detector/tree/main
#### (7) Crosswalk (available)
```bash
python main.py --mode crosswalk --source_video_path pedestrian.mp4
```
https://github.com/xN1ckuz/Crosswalks-Detection-using-YOLO/tree/main

