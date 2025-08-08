import os
import pandas as pd

def extract_pedestrian_info(video_path, gender_csv_path=None, clothing_csv_path=None, belongings_csv_path=None, crossing_csv_path=None, output_csv_path=None):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C7]crossing_pe_info.csv")
    if gender_csv_path is None:
        gender_csv_path = os.path.join(output_dir, "[P7]pedestrian_gender.csv")
    if clothing_csv_path is None:
        clothing_csv_path = os.path.join(output_dir, "[P8]clothing.csv")
    if belongings_csv_path is None:
        belongings_csv_path = os.path.join(output_dir, "[P9]pedestrian_belongings.csv")
    if crossing_csv_path is None:
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")

    gender_df = pd.read_csv(gender_csv_path)
    belongings_df = pd.read_csv(belongings_csv_path)
    clothing_df = pd.read_csv(clothing_csv_path)
    crossing_df = pd.read_csv(crossing_csv_path)

    crossing_df = crossing_df[crossing_df['crossed'] == True]

    belongings_cols = ['backpack', 'umbrella', 'handbag', 'suitcase']
    clothing_cols = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
                     'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers',
                     'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']

    output_rows = []
    for _, row in crossing_df.iterrows():
        pid = row['track_id']
        crossed = True

        gender_row = gender_df[gender_df['id'] == pid]
        gender = gender_row['gender'].values[0] if not gender_row.empty else 'Unknown'

        belongings_person = belongings_df[belongings_df['track_id'] == pid]
        belongings_result = {}
        if not belongings_person.empty:
            for item in belongings_cols:
                proportion = belongings_person[item].sum() / len(belongings_person)
                belongings_result[item] = 1 if proportion >= 0.2 else 0
        else:
            for item in belongings_cols:
                belongings_result[item] = 0

        clothing_person = clothing_df[clothing_df['track_id'] == pid]
        clothing_result = {}
        if not clothing_person.empty:
            for item in clothing_cols:
                proportion = clothing_person[item].sum() / len(clothing_person)
                clothing_result[item] = 1 if proportion >= 0.2 else 0
        else:
            for item in clothing_cols:
                clothing_result[item] = 0

        final_row = {
            'track_id': pid,
            'crossed': crossed,
            'gender': gender,
            **belongings_result,
            **clothing_result
        }
        output_rows.append(final_row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nPedestrians info results saved to：{output_csv_path}")

