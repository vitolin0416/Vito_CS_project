# 先把分群做出來 (要先把惠晴整理的方式轉成所有球員都適用)
# 把 shot 和 rally 合併 (有必要的話要加入match確認球員)
# 合併分群和shot
# 變成 5 shot 形式，加入輸贏

import pandas as pd
# Load the dataset
data = pd.read_csv('convert_shot.csv',encoding='utf-8')
match = pd.read_csv('match.csv', encoding='utf-8')
set = pd.read_csv('set.csv', encoding='utf-8')
rally = pd.read_csv('rally.csv', encoding='utf-8')
match = match[["match_id", "win_A", "win_B", "lose_C", "lose_D"]]
set = set[["set_id", "match_id"]]
rally = rally[["rally_id", "set_id"]]
data = pd.merge(data, rally, on='rally_id')
data = pd.merge(data, set, on="set_id")
data = pd.merge(data, match, on="match_id")
data.to_csv('mod_processed_shot.csv', index=False)
# Filter players
players_filtered = data
# players_filtered = data[~((data['rally_id'] >= 73) & (data['rally_id'] <= 143))]

# Initialize the list to collect rows
rows_list = []
permanent_row_list = []

# Initialize previous shot variable
prev = ""
prevRallyId = players_filtered.iloc[0]['rally_id']

# Iterate over the filtered DataFrame rows
for _, row in players_filtered.iterrows():
    # put the valid shots into the big row list
    if prevRallyId != row['rally_id']:
        permanent_row_list.extend(rows_list)
        rows_list = []
        prevRallyId = row['rally_id']

    if (row['ball_type'] == '擋小球') or (row['ball_type'] == '勾球') or (row['ball_type'] == '放小球') or (row['ball_type'] == '小平球') > 0:
        row['ball_type'] = '網前小球'
    elif (row['ball_type'] == '防守回挑'):
        row['ball_type'] = '挑球'
    elif (row['ball_type'] == '防守回抽') or (row['ball_type'] == '後場抽平球') > 0:
        row['ball_type'] = '平球'
    elif (row['ball_type'] == '過度切球'):
        row['ball_type'] = '切球'
    elif (row['ball_type'] == '推球') or (row['ball_type'] == '撲球') > 0:
        row['ball_type'] = '推撲球'
    if row['shot_num'] != 1:
        prev = row["ball_type"]
    newrow = {
        "hit_player_x": None,
        "hit_player_y": None,
        "partner_x": None,
        "partner_y": None,
        "ball_type": row["ball_type"],
        "player": row["player"], 
        "shot_id": row["shot_id"],
        "rally_id": row["rally_id"],
        "set_id": row["set_id"],
        "match_id": row["match_id"]
        # "挑球": 0,
        # "殺球": 0,
        # "平球": 0,
        # "網前小球": 0,
        # "切球": 0,
        # "推撲球": 0,
        # "長球": 0
    }
    if row['player'] == row['win_A']:
        newrow["hit_player_x"] = row['player_A_x']
        newrow["hit_player_y"] = row['player_A_y']
        newrow["partner_x"] = row['player_B_x']
        newrow["partner_y"] = row['player_B_y']
    elif row['player'] == row['win_B']:
        newrow["hit_player_x"] = row['player_B_x']
        newrow["hit_player_y"] = row['player_B_y']
        newrow["partner_x"] = row['player_A_x']
        newrow["partner_y"] = row['player_A_y']
    elif row['player'] == row['lose_C']:
        newrow["hit_player_x"] = row['player_C_x']
        newrow["hit_player_y"] = row['player_C_y']
        newrow["partner_x"] = row['player_D_x']
        newrow["partner_y"] = row['player_D_y']
    elif row['player'] == row['lose_D']:
        newrow["hit_player_x"] = row['player_D_x']
        newrow["hit_player_y"] = row['player_D_y']
        newrow["partner_x"] = row['player_C_x']
        newrow["partner_y"] = row['player_C_y']
    if newrow["hit_player_y"] > 67:
        newrow["hit_player_x"] = 61 - newrow["hit_player_x"]
        newrow["hit_player_y"] = 134 - newrow["hit_player_y"]
        newrow["partner_x"] = 61 - newrow["partner_x"]
        newrow["partner_y"] = 134 - newrow["partner_y"]
        
    # Ensure only valid ball types are added
    newrow["ball_type"] = row["ball_type"]
    if newrow["hit_player_x"] < 0 or newrow['hit_player_y'] < 0 or newrow['partner_x'] < 0 or newrow['partner_y'] < 0:
        rows_list = []
    elif newrow["hit_player_x"] > 61 or newrow['hit_player_y'] > 134 or newrow['partner_x'] > 61 or newrow['partner_y'] > 134:
        rows_list = []
    else:
        rows_list.append(newrow)
# in sure the last row is in the list
permanent_row_list.extend(rows_list)

# Create a new DataFrame from the list of rows
new_df = pd.DataFrame(permanent_row_list)

# Define the desired column order
column_order = [
    "player", "shot_id", "rally_id", "set_id", "match_id", 
    "hit_player_x", "hit_player_y", "partner_x", "partner_y",
    "ball_type"
]

# Reorder the columns
new_df = new_df[column_order]
new_df = pd.get_dummies(new_df, columns=['ball_type'])
ballTypes = [
    "ball_type_切球", "ball_type_平球", "ball_type_挑球", "ball_type_推撲球", "ball_type_殺球",
    'ball_type_發短球', "ball_type_發長球", "ball_type_網前小球", "ball_type_長球"
]
new_df[ballTypes] = new_df[ballTypes].astype(int)


# Save the processed data
new_df.to_csv('processed_shot.csv', index=False)


import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
# set traditional Chinese display
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
data = pd.read_csv('processed_shot.csv')

# selsct features
features = data[[
    "hit_player_x", "hit_player_y", "partner_x", "partner_y"
    , "ball_type_切球", "ball_type_平球", "ball_type_挑球", "ball_type_推撲球", "ball_type_殺球",
    'ball_type_發短球', "ball_type_發長球", "ball_type_網前小球", "ball_type_長球"
    ]]

# standardize features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# use elbow method to find the optimal k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    sse.append(kmeans.inertia_)

# plot the elbow method
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), sse, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.title('Elbow Method For Optimal k')
plt.savefig('elbow_plot_balltype.png')
plt.show()

# # using silhouette analysis to determine the best k
# range_n_clusters = list(range(2, 11))
# silhouette_avg = []
# for num_clusters in range_n_clusters:
#     kmeans = KMeans(n_clusters=num_clusters, random_state=42)
#     kmeans.fit(features_scaled)
#     cluster_labels = kmeans.labels_
#     silhouette_avg.append(silhouette_score(features_scaled, cluster_labels))
# optimal_k = range_n_clusters[silhouette_avg.index(max(silhouette_avg))]

# print("---------------------------\n")
# print(f"The optimal number of clusters is: {optimal_k}")
# print("\n---------------------------")

# clustering
n = 9  # Best n from silhouette analysis
kmeans = KMeans(n_clusters=n)
data['cluster'] = kmeans.fit_predict(features)
clusterCount = data['cluster'].value_counts()
print(clusterCount)

# plot the clustering result
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

data_indices = data.index

for cluster in range(0, n):
    plt.figure(figsize=(9.15, 10.05))
    
    cluster_data = data[data['cluster'] == cluster]
    if len(cluster_data) > 100:
        sample_cluster_data = cluster_data.sample(n=100, random_state=1)
    else:
        sample_cluster_data = cluster_data
    
    plt.scatter(sample_cluster_data['hit_player_x'], sample_cluster_data['hit_player_y'], color='r', label=f'Cluster {cluster} - Hit player')
    plt.plot([sample_cluster_data['hit_player_x'], sample_cluster_data['partner_x']], [sample_cluster_data['hit_player_y'], sample_cluster_data['partner_y']], color='y', label=f'Cluster {cluster} - Partner line') 
    plt.scatter(sample_cluster_data['partner_x'], sample_cluster_data['partner_y'], color='b', label=f'Cluster {cluster} - Partner')
    plt.plot([sample_cluster_data['hit_player_x'].mean(), sample_cluster_data['partner_x'].mean()], [sample_cluster_data['hit_player_y'].mean(), 
        sample_cluster_data['partner_y'].mean()], color='g', label=f'Cluster {cluster} - Mean line', linewidth=5)
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.xlim(0, 61) 
    plt.ylim(0, 67) 
    # plt.xlim(-150, 150) 
    # plt.ylim(-150, 150) 
    plt.savefig(f'cluster_{cluster}_positions_balltypes.jpg')
    plt.close()

# plot all clustering results in one figure
plt.figure(figsize=(9.15, 10.05))
for cluster in range(n):
    cluster_data = data[data['cluster'] == cluster]
    plt.scatter(cluster_data['hit_player_x'], cluster_data['hit_player_y'], color=colors[cluster % len(colors)], label=f'Cluster {cluster} - Hit player')
    plt.scatter(cluster_data['partner_x'], cluster_data['partner_y'], color=colors[cluster % len(colors)], marker='x', label=f'Cluster {cluster} - Partner')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.xlim(0, 61) 
plt.ylim(0, 67)
# plt.legend()
plt.savefig(f'clusters_positions_balltype.jpg')
plt.show()

data.to_csv('clustered_shot.csv', index=False, encoding='utf-8')