import pandas as pd
import chardet

# Detect the file encoding
with open('rally_0108.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']

print(f"Detected encoding: {file_encoding}")

# Read the CSV file with the detected encoding
df = pd.read_csv('rally_0108.csv', 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})

df_filted = df[df['rally_id'] >= 1352]
df_filted.to_csv('rally_from1352.csv', index=False, encoding='utf-8-sig')
count = df_filted['rally_score_reason'].value_counts()
print(count)