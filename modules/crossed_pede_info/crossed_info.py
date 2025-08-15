import os
import pandas as pd

def extract_pedestrian_info(video_path, gender_csv_path=None, clothing_csv_path=None, belongings_csv_path=None,
                            crossing_csv_path=None, phone_usage_csv_path=None, output_csv_path=None,
                            risky_crossing_csv=None, run_redlight_csv=None, crosswalk_usage_csv=None,
                            waiting_csv_path=None, nearby_count_path=None
                            ):

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    if output_csv_path is None:
        output_dir = os.path.join("analysis_results", video_name)
        os.makedirs(output_dir, exist_ok=True)
        output_csv_path = os.path.join(output_dir, "[C7]crossing_pe_info.csv")
    if gender_csv_path is None:
        gender_csv_path = os.path.join(output_dir, "[P6]age_gender.csv")
    if clothing_csv_path is None:
        clothing_csv_path = os.path.join(output_dir, "[P8]clothing.csv")
    if belongings_csv_path is None:
        belongings_csv_path = os.path.join(output_dir, "[P9]pedestrian_belongings.csv")
    if crossing_csv_path is None:
        crossing_csv_path = os.path.join(output_dir, "[C3]crossing_judge.csv")
    if phone_usage_csv_path is None:
        phone_usage_csv_path = os.path.join(output_dir, "[P5]phone_usage.csv")
    if risky_crossing_csv is None:
        risky_crossing_csv = os.path.join(output_dir, "[C1]risky_crossing.csv")
    if run_redlight_csv is None:
        run_redlight_csv = os.path.join(output_dir, "[C5]red_light_runner.csv")
    if crosswalk_usage_csv is None:
        crosswalk_usage_csv = os.path.join(output_dir, "[C4]crosswalk_usage.csv")
    if waiting_csv_path is None:
        waiting_csv_path = os.path.join(output_dir, "[P3]waiting_time.csv")
    if nearby_count_path is None:
        nearby_count_path = os.path.join(output_dir, "[C10]nearby_count.csv")

    gender_df = pd.read_csv(gender_csv_path)
    belongings_df = pd.read_csv(belongings_csv_path)
    clothing_df = pd.read_csv(clothing_csv_path)
    crossing_df = pd.read_csv(crossing_csv_path)
    phone_usage_df = pd.read_csv(phone_usage_csv_path)
    risky_crossing_df = pd.read_csv(risky_crossing_csv)
    run_redlight_df = pd.read_csv(run_redlight_csv)
    crosswalk_usage_df = pd.read_csv(crosswalk_usage_csv)
    nearby_count_df = pd.read_csv(nearby_count_path)
    waiting_df = pd.read_csv(waiting_csv_path)

    crossing_df = crossing_df[crossing_df['crossed'] == True]

    belongings_cols = ['backpack', 'umbrella', 'handbag', 'suitcase']
    clothing_cols = ['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear',
                     'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers',
                     'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress']

    output_rows = []
    for _, row in crossing_df.iterrows():
        pid = row['track_id']
        crossed = True

        gender_row = gender_df[gender_df['person_id'] == pid]
        gender = gender_row['gender'].values[0] if not gender_row.empty else 'Unknown'
        age = gender_row['age'].values[0] if not gender_row.empty else 'Unknown'

        nearby_row = nearby_count_df[nearby_count_df['track_id'] == pid]
        nearby_begin = nearby_row['nearby_first_1_5'].values[0] if not nearby_row.empty else 0
        nearby_all = nearby_row['nearby_all'].values[0] if not nearby_row.empty else 0

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

        phone_person = phone_usage_df[phone_usage_df['track_id'] == pid]
        if not phone_person.empty:
            phone_ratio = phone_person['phone_using'].astype(str).str.lower().eq("true").sum() / len(phone_person)
            phone_using = 1 if phone_ratio >= 0.1 else 0
        else:
            phone_using = 0

        risky = risky_crossing_df[risky_crossing_df['track_id'] == pid]
        if not risky.empty:
            risky_ratio = risky['risk'].astype(str).str.lower().eq("risky").sum() / len(risky)
            risky_or_not = 1 if risky_ratio >= 0.2 else 0
        else:
            risky_or_not = 0

        # runred = run_redlight_df[run_redlight_df['track_id'] == pid]
        # if not runred.empty:
        #     runred_or_not = 1 if runred['ran_red_light']=="TRUE" else 0
        # else:
        #     runred_or_not = 0
        # cwuse = crosswalk_usage_df[crosswalk_usage_df['track_id'] == pid]
        # if not cwuse.empty:
        #     cwuse_or_not = 1 if cwuse['used_crosswalk']=="TRUE" else 0
        # else:
        #     cwuse_or_not = 0

        runred_row = run_redlight_df[run_redlight_df['track_id'] == pid]
        if not runred_row.empty:
            runred = 1 if runred_row['ran_red_light'].values[0] == "TRUE" else 0
        else:
            runred = 'Unknown'

        cwuse_row = crosswalk_usage_df[crosswalk_usage_df['track_id'] == pid]
        if not cwuse_row.empty:
            cwuse = 1 if cwuse_row['used_crosswalk'].values[0] == "TRUE" else 0
        else:
            cwuse = 'Unknown'

        waiting_time = waiting_df[waiting_df['track_id'] == pid]
        wt = waiting_time['waiting_time'].values[0] if not waiting_time.empty else 0



        final_row = {
            'track_id': pid,
            'crossed': crossed,
            'nearby_count_beginning':nearby_begin,
            'nearby_count_whole':nearby_all,
            'risky_crossing': risky_or_not,
            'run_red_light': runred,
            'crosswalk_use_or_not': cwuse,
            'waiting_time': wt,
            'gender': gender,
            'age': age,
            'phone_using': phone_using,
            **belongings_result,
            **clothing_result
        }
        output_rows.append(final_row)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\nPedestrians info results saved to：{output_csv_path}")