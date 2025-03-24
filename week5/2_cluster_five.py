import pandas as pd

# df = pd.read_csv('clustered_shot_balltype.csv',encoding='utf-8')
df = pd.read_csv('clustered_shot_balltype.csv',encoding='utf-8')

# select rows that is at the last five of that rally
df_grouped = df.groupby('rally_id')
filtered_group = df_grouped.filter(lambda x: len(x) > 5)
selected_group = filtered_group.groupby('rally_id').tail(5)
df['selected'] = df.index.isin(selected_group.index)
counts = df['selected'].value_counts()
print('\nhow many selected: ')
print(counts)
# drop all the other rows that is not seletcted
df = df.drop(df[df['selected'] == False].index)
df = df.drop(columns=['selected'])
# df.to_csv('cluster_shot_last_five.csv', index=False, encoding='utf-8-sig')

# start grouping every rally together
# Sort by rally_id and shot_order (descending for last shots)
df_sorted = df.sort_values(by=['rally_id', 'shot_id'], ascending=[True, False])

# Group by rally_id and get the last five shots
grouped = df_sorted.groupby('rally_id').head(5)

# Define a function to extract shot attributes into separate columns
def extract_shots(group):
    shots = group.to_dict('records')
    row = {'rally_id': group['rally_id'].iloc[0],
           'set_id': group['set_id'].iloc[0],
           'match_id': group['match_id'].iloc[0]
           }
    for i, shot in enumerate(shots):
        row[f'shot_{i+1}_player'] = shot['player']
        row[f'shot_{i+1}_hit_player_x'] = shot['hit_player_x']
        row[f'shot_{i+1}_hit_player_y'] = shot['hit_player_y']
        row[f'shot_{i+1}_partner_x'] = shot['partner_x']
        row[f'shot_{i+1}_partner_y'] = shot['partner_y']
        row[f'shot_{i+1}_切球'] = shot['ball_type_切球']
        row[f'shot_{i+1}_平球'] = shot['ball_type_平球']
        row[f'shot_{i+1}_推撲球'] = shot['ball_type_推撲球']
        row[f'shot_{i+1}_網前小球'] = shot['ball_type_網前小球']
        row[f'shot_{i+1}_殺球'] = shot['ball_type_殺球']
        row[f'shot_{i+1}_發短球'] = shot['ball_type_發短球']
        row[f'shot_{i+1}_發長球'] = shot['ball_type_發長球']
        row[f'shot_{i+1}_長球'] = shot['ball_type_長球']
        row[f'shot_{i+1}_cluster'] = shot['cluster']
    return pd.DataFrame([row])

# Apply the function to each group
extracted_df = grouped.groupby('rally_id').apply(extract_shots).reset_index(drop=True)

# do one hot encoding on cluster
df = extracted_df
for i in range(1, 6):
    df = pd.get_dummies(df, columns=[f'shot_{i}_cluster'])
df = df.astype(int)

# Display the result
print(df)
df.to_csv('cluster_shot_grouped_positions.csv', index=False, encoding='utf-8-sig')

# add the winning column
grouped = df
rally = pd.read_csv('rally.csv',encoding='utf-8')
rally = rally[['rally_id', 'fault_player']]
merge = pd.merge(grouped, rally, on="rally_id", how='left')

merge['winning'] = merge['fault_player'] != merge['shot_1_player']
merge['winning'] = merge['winning'].astype(int)

columns_to_drop = merge.filter(regex=r'^shot_\d+_player$').columns
columns_to_obtain = [col for col in merge.columns if col not in columns_to_drop]
final_df = merge[columns_to_obtain]
print(final_df)
final_df.to_csv('cluster_final_BALLTYPE.csv', index=False, encoding='utf-8-sig')