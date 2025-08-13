import pandas as pd
import ast

df = pd.read_csv('mapping.csv')

columns_needed = ['id', 'city', 'videos', 'time_of_day', 'start_time', 'end_time']
df = df[columns_needed]

new_rows = []

for _, row in df.iterrows():
    id_ = row['id']
    city = row['city']

    videos = row['videos'].strip('[]').split(',')
    videos = [v.strip() for v in videos]

    time_of_day_list = ast.literal_eval(row['time_of_day'])
    start_time_list = ast.literal_eval(row['start_time'])
    end_time_list = ast.literal_eval(row['end_time'])

    for i, video in enumerate(videos):
        start_time = start_time_list[i][0]
        end_time = end_time_list[i][0]

        if start_time > 100 or end_time > 5000:
            continue

        new_rows.append({
            'id': id_,
            'name': f"{city}{i+1}",
            'city': city,
            'video': video,
            'time_of_day': time_of_day_list[i][0],
            'start_time': start_time,
            'end_time': end_time
        })

new_df = pd.DataFrame(new_rows)

new_df.to_csv('mapping_ex.csv', index=False)
