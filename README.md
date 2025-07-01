# MSc-Project

## Toolbox:
### 1. Pedestrian Analysis 
#### (1) Count the number of pedestrians in the zone (available)
```bash
python main.py --mode count --source_video_path pedestrian.mp4
```
https://github.com/roboflow/supervision/tree/develop/examples/count_people_in_zone
#### (2) Pedestrian Speed Estimator
https://github.com/shriram1998/PedestrianSpeedEstimator
#### (3) Waiting time in the zone (available)
```bash
python main.py --mode Waiting --source_video_path pedestrian.mp4
```  
https://github.com/roboflow/supervision/tree/develop/examples/time_in_zone
#### (4) Pedestrian keypoint tracking (available)
```bash
python main.py --mode tracking --source_video_path pedestrian.mp4
```    
https://supervision.roboflow.com/develop/how_to/track_objects/#keypoint-tracking
#### (5) Pedestrian phone usage detection
https://github.com/HasnainAhmedO7/Detection-of-Phone-Usage-with-Computer-Vision
#### (6) Age and gender (available)
```bash
python main.py --mode agegender --source_video_path pedestrian.mp4
```
https://github.com/smahesh29/Gender-and-Age-Detection  
#### (7) Race (available)
```bash
python main.py --mode race --source_video_path pedestrian.mp4
```
https://github.com/serengil/deepface
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
#### (4) Traffic sign (Available)
```bash
python main.py --mode sign --source_video_path pedestrian.mp4
```
https://github.com/Kartik-Aggarwal/Real-Time-Traffic-Sign-Detection/tree/main
