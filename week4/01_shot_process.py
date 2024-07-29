import pandas as pd
import chardet

# Detect the file encoding
with open('shot_grouped.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']
print(f"Detected encoding: {file_encoding}")
# Read the CSV file with the detected encoding
shot_unproccessed = pd.read_csv('shot_grouped.csv', 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})
shot = shot_unproccessed

# combine the shot types
for i in range(5):
    shot[f'shot{i+1}_網前小球'] = shot[f'shot_{i+1}_擋小球'] ^ shot[f'shot_{i+1}_勾球'] ^ shot[f'shot_{i+1}_小平球'] ^ shot[f'shot_{i+1}_放小球']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_放小球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_擋小球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_勾球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_小平球').columns, axis=1)
    shot[f'shot{i+1}_推撲球'] = shot[f'shot_{i+1}_推球'] ^ shot[f'shot_{i+1}_撲球']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_推球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_撲球').columns, axis=1)
    shot[f'shot{i+1}_挑球'] = shot[f'shot_{i+1}_挑球'] ^ shot[f'shot_{i+1}_防守回挑']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_挑球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_防守回挑').columns, axis=1)
    shot[f'shot{i+1}_平球'] = shot[f'shot_{i+1}_防守回抽'] ^ shot[f'shot_{i+1}_平球'] ^ shot[f'shot_{i+1}_後場抽平球']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_防守回抽').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_平球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_後場抽平球').columns, axis=1)
    shot[f'shot{i+1}_切球'] = shot[f'shot_{i+1}_切球'] ^ shot[f'shot_{i+1}_過度切球']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_切球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_過度切球').columns, axis=1)
    shot[f'shot{i+1}_殺球'] = shot[f'shot_{i+1}_殺球'] ^ shot[f'shot_{i+1}_點扣']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_殺球').columns, axis=1)
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_點扣').columns, axis=1)
    shot[f'shot{i+1}_長球'] = shot[f'shot_{i+1}_長球']
    shot = shot.drop(shot.filter(like= f'shot_{i+1}_長球').columns, axis=1)

# encode one-hot for hit_area
for i in range(5):
    shot = pd.get_dummies(shot, columns=[f'shot_{i+1}_hit_area'], prefix=f"shot_{i+1}_hit_area")
    for j in range(10, 17):
        try:
            shot = shot.drop(shot.filter(like= f'shot_{i+1}_hit_area_{j}').columns, axis=1)
        except KeyError:
            pass
shot = shot.astype(int)

shot.to_csv('01_shot.csv', index=False, encoding='utf-8-sig')