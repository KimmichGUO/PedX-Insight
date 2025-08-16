import os
from collections import Counter
from paddlex import create_pipeline
import numpy as np
import csv

def predict_age_gender(image_path, pipeline):
    output_gen = pipeline.predict(image_path, cls_threshold=1e-10)
    output = list(output_gen)
    if not output or not output[0]['boxes']:
        return None, None

    first_box = output[0]['boxes'][0]
    labels = first_box['labels']
    scores = first_box['cls_scores']

    age_labels = ['AgeLess18(年龄小于18岁)', 'Age18-60(年龄在18-60岁之间)', 'AgeOver60(年龄大于60岁)']
    age_scores = [scores[labels.index(l)] if l in labels else 0 for l in age_labels]
    age = age_labels[np.argmax(age_scores)].split('(')[0]

    female_score = scores[labels.index('Female(女性)')] if 'Female(女性)' in labels else 0
    gender = 'female' if female_score > 0.1 else 'male'

    return age, gender


def run_age_gender(video_path):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    img_root = os.path.join('./analysis_results', video_name, 'pedestrian_img')

    pipeline = create_pipeline(pipeline="pedestrian_attribute_recognition")

    final_results = {}

    for person_id in os.listdir(img_root):
        person_folder = os.path.join(img_root, person_id)
        if not os.path.isdir(person_folder):
            continue

        ages, genders = [], []
        for img_file in os.listdir(person_folder):
            if not img_file.lower().endswith('.png'):
                continue
            img_path = os.path.join(person_folder, img_file)
            age, gender = predict_age_gender(img_path, pipeline)
            if age and gender:
                ages.append(age)
                genders.append(gender)

        if ages and genders:
            final_age = Counter(ages).most_common(1)[0][0]
            final_gender = Counter(genders).most_common(1)[0][0]
            # final_results[person_id] = {'age': final_age, 'gender': final_gender}
        else:
            final_age = None
            final_gender = None
        final_results[person_id] = {'age': final_age, 'gender': final_gender}
        print(f"Finished analyzing {person_id}: Age={final_age}, Gender={final_gender}")

    csv_path = os.path.join('./analysis_results', video_name, '[P6]age_gender.csv')
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'age', 'gender'])
        for person_id, res in final_results.items():
            clean_id = person_id.replace("id_", "")
            writer.writerow([clean_id, res['age'], res['gender']])

    print(f"Age and gender results saved to {csv_path}")
    return final_results