import pandas as pd
import chardet

# Detect the file encoding
with open('shot_last_five.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']

print(f"Detected encoding: {file_encoding}")

# Read the CSV file with the detected encoding
df = pd.read_csv('shot_last_five.csv', 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})

df = df.drop(df[df['selected'] == False].index)
df = df.drop(columns=['selected'])
# df.to_csv('shot_only_five.csv', index=False, encoding='utf-8-sig')

# start grouping every rally together
# Sort by rally_id and shot_order (descending for last shots)
df_sorted = df.sort_values(by=['rally_id', 'shot_number'], ascending=[True, False])

# Group by rally_id and get the last five shots
grouped = df_sorted.groupby('rally_id').head(5)

# Define a function to extract shot attributes into separate columns
def extract_shots(group):
    shots = group.to_dict('records')
    row = {'rally_id': group['rally_id'].iloc[0]}
    for i, shot in enumerate(shots):
        row[f'shot_{i+1}_player'] = shot['shot_player']
        row[f'shot_{i+1}_aroundhead'] = shot['shot_aroundhead']
        row[f'shot_{i+1}_backhand'] = shot['shot_backhand']
        row[f'shot_{i+1}_hit_area'] = shot['shot_hit_area']
        row[f'shot_{i+1}_badminton_height'] = shot['shot_badminton_height']
        row[f'shot_{i+1}_player_location_x'] = shot['player_location_x']
        row[f'shot_{i+1}_player_location_y'] = shot['player_location_y']
        row[f'shot_{i+1}_opponent_location_x'] = shot['opponent_location_x']
        row[f'shot_{i+1}_opponent_location_y'] = shot['opponent_location_y']
        row[f'shot_{i+1}_切球'] = shot['shot_切球']
        row[f'shot_{i+1}_勾球'] = shot['shot_勾球']
        row[f'shot_{i+1}_小平球'] = shot['shot_小平球']
        row[f'shot_{i+1}_平球'] = shot['shot_平球']
        row[f'shot_{i+1}_後場抽平球'] = shot['shot_後場抽平球']
        row[f'shot_{i+1}_挑球'] = shot['shot_挑球']
        row[f'shot_{i+1}_推球'] = shot['shot_推球']
        row[f'shot_{i+1}_撲球'] = shot['shot_撲球']
        row[f'shot_{i+1}_擋小球'] = shot['shot_擋小球']
        row[f'shot_{i+1}_放小球'] = shot['shot_放小球']
        row[f'shot_{i+1}_未知球種'] = shot['shot_未知球種']
        row[f'shot_{i+1}_殺球'] = shot['shot_殺球']
        row[f'shot_{i+1}_發短球'] = shot['shot_發短球']
        row[f'shot_{i+1}_發長球'] = shot['shot_發長球']
        row[f'shot_{i+1}_過度切球'] = shot['shot_過度切球']
        row[f'shot_{i+1}_長球'] = shot['shot_長球']
        row[f'shot_{i+1}_防守回抽'] = shot['shot_防守回抽']
        row[f'shot_{i+1}_防守回挑'] = shot['shot_防守回挑']
        row[f'shot_{i+1}_點扣'] = shot['shot_點扣']
    return pd.DataFrame([row])

# Apply the function to each group
extracted_df = grouped.groupby('rally_id').apply(extract_shots).reset_index(drop=True)

# Display the result
print(extracted_df)
extracted_df.to_csv('shot_grouped.csv', index=False, encoding='utf-8-sig')