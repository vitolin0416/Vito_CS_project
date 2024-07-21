import pandas as pd
import chardet

# Detect the file encoding
with open('rally_from1352.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']

print(f"Detected encoding: {file_encoding}")

# Read the CSV file with the detected encoding
df = pd.read_csv('rally_from1352.csv', 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})

df_filtered = df[['rally_id', 'rally_score_reason']]
# df_filtered.to_csv('rally_filtered.csv', index=False, encoding='utf-8-sig')
def set_winning(row):
    if row['rally_score_reason'] in ['對手出界', '對手未過網', '對手掛網']:
        return 0
    if row['rally_score_reason'] in ['對手落點判斷失誤', '落地致勝']:
        return 1
    else:
        return -1
df_filtered['winning'] = df.apply(set_winning, axis=1)
df_filtered = df_filtered.drop(columns=['rally_score_reason'])
print(df_filtered)
df_filtered.to_csv('rally_winning.csv', index=False, encoding='utf-8-sig')

df_filtered = df[['rally_id', 'rally_score_reason']]
def set_out(row):
    if row['rally_score_reason'] in ['對手出界']:
        return 1
    if row['rally_score_reason'] in ['對手落點判斷失誤', '落地致勝', '對手未過網', '對手掛網']:
        return 0
    else:
        return -1
df_filtered['out'] = df_filtered.apply(set_out, axis=1)
df_filtered = df_filtered.drop(columns=['rally_score_reason'])
print(df_filtered)
df_filtered.to_csv('rally_out.csv', index=False, encoding='utf-8-sig')

df_filtered = df[['rally_id', 'rally_score_reason']]
def set_in(row):
    if row['rally_score_reason'] in ['落地致勝']:
        return 1
    if row['rally_score_reason'] in ['對手落點判斷失誤', '對手出界', '對手未過網', '對手掛網']:
        return 0
    else:
        return -1
df_filtered['in'] = df_filtered.apply(set_in, axis=1)
df_filtered = df_filtered.drop(columns=['rally_score_reason'])
print(df_filtered)
df_filtered.to_csv('rally_in.csv', index=False, encoding='utf-8-sig')