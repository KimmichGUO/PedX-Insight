import os
import time
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch import nn
from collections import Counter
import csv


ClassificationModel = 'modules/gender/ResNet-18 Age 0.60 + Gender 93.pt'
runOn = "cuda:0" if torch.cuda.is_available() else "cpu"
transform = transforms.Compose([transforms.ToTensor()])


FaceDetector = torch.hub.load('ultralytics/yolov5',
                              'custom',
                              'modules/gender/Best.onnx',
                              _verbose=False)
FaceDetector.eval().to(runOn)

FaceClassifier = models.resnet18(pretrained=True)
FaceClassifier.fc = nn.Linear(512, 11)
FaceClassifier = nn.Sequential(FaceClassifier, nn.Sigmoid())
FaceClassifier.load_state_dict(torch.load(ClassificationModel, map_location='cpu'))
FaceClassifier.eval().to(runOn)

def extract_faces(img, detector, threshold=0.01):
    faces = []
    results = detector(img).pandas().xyxy[0]
    for det in results.values:
        xmin, ymin, xmax, ymax, conf = det[:5]
        if conf >= threshold:
            face = img.crop((xmin, ymin, xmax, ymax))
            faces.append(face)
    return faces

def preprocess(face_img):
    face_img = face_img.convert('RGB').resize((200, 200))
    return transform(face_img).unsqueeze(0)

def predict_gender(model, face_tensor):
    face_tensor = face_tensor.to(runOn)
    with torch.no_grad():
        output = model(face_tensor)[0]
        gender_output = output[-2:]
        pred = int(torch.argmax(gender_output))
        confidence = float(torch.max(gender_output))
        if confidence < 0.01:
            return "Unknown"
        return "Male" if pred == 0 else "Female"

def analyze_gender_majority(folder_path):
    all_genders = []

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            continue
        try:
            image_path = os.path.join(folder_path, filename)
            img = Image.open(image_path).convert('RGB').resize((200, 200))
            faces = extract_faces(img, FaceDetector)

            gender = "Unknown"
            for face in faces:
                tensor_face = preprocess(face)
                gender = predict_gender(FaceClassifier, tensor_face)
                if gender != "Unknown":
                    break

            all_genders.append(gender)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            all_genders.append("Unknown")

    detected = [g for g in all_genders if g != "Unknown"]

    if len(detected) == 0:
        return "Unknown"
    elif len(detected) == 1:
        return detected[0]
    elif len(detected) == 2:
        return detected[0] if detected[0] == detected[1] else random.choice(detected)
    else:
        freq = Counter(detected)
        return freq.most_common(1)[0][0]


def analyze_all_ids(pedestrian_img_dir, output_csv_path):
    gender_results = []

    for folder_name in sorted(os.listdir(pedestrian_img_dir)):
        id_folder = os.path.join(pedestrian_img_dir, folder_name)
        if os.path.isdir(id_folder):
            gender = analyze_gender_majority(id_folder)
            id_num = folder_name.split('_')[-1] if '_' in folder_name else folder_name
            gender_results.append((id_num, gender))

    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['id', 'gender'])
        writer.writerows(gender_results)

    print(f"\nGender detection results saved to {output_csv_path}")

def gender_analysis(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join("analysis_results", video_name)
    pedestrian_img_dir = os.path.join(output_dir, "pedestrian_img")
    output_csv_path = os.path.join(output_dir, "gender_pedestrians.csv")
    analyze_all_ids(pedestrian_img_dir, output_csv_path)
