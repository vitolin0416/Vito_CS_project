import pandas as pd
import chardet

# Detect the file encoding
with open('shot_drop_unwanted.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']

print(f"Detected encoding: {file_encoding}")

# Read the CSV file with the detected encoding
df = pd.read_csv('shot_drop_unwanted.csv', 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})

# Filter rows
df_filtered = df[df['shot_id'] >= 11373]
df_filtered['shot_hit_area'] = df_filtered['shot_hit_area'].fillna(0)

# Reset index
df_filtered = df_filtered.reset_index(drop=True)

# Save to new CSV file with UTF-8-BOM encoding
# df_filtered.to_csv('shot_with_num_position.csv', index=False, encoding='utf-8-sig')
# print("File saved successfully. You should now be able to open it directly in Excel.")

shot_type_counts = df_filtered['shot_type'].value_counts()

# Display the counts
print("\nUnique shot types and their counts:")
print(shot_type_counts)

# Display the total number of unique shot types
print(f"\nTotal number of unique shot types: {len(shot_type_counts)}")

# Perform one-hot encoding
df_encoded = pd.get_dummies(df_filtered, columns=['shot_type'], prefix='shot')
# Check the new shape
print(f"New dataframe shape: {df_encoded.shape}")

# Display the first few rows and new columns
print(df_encoded[['shot_id'] + [col for col in df_encoded.columns if col.startswith('shot_')]].head())

# df_encoded.to_csv('shot_encoded.csv', index=False, encoding='utf-8-sig')

# df = df.dropna(axis=1)
# df_droped_unwanted = df.drop(columns=['frame_num', 'time', 'shot_hit_position_x', 'shot_hit_position_y', 'shot_return_position_x', 'shot_return_position_y'])
# df_droped_unwanted = df_droped_unwanted.astype(int)
# df_droped_unwanted.to_csv('shot_drop_unwanted.csv', index=False, encoding='utf-8-sig')

df_grouped = df_encoded.groupby('rally_id')
filtered_group = df_grouped.filter(lambda x: len(x) > 5)
selected_group = filtered_group.groupby('rally_id').tail(5)
df['selected'] = df.index.isin(selected_group.index)
df.to_csv('shot_last_five.csv', index=False, encoding='utf-8-sig')
counts = df['selected'].value_counts()
print('\nhow many selected: ')
print(counts)