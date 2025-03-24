import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = 'clustered_shot.csv'
df = pd.read_csv(file_path)

# Inspect the data
print(df.head())

# Define the ball type columns
ball_type_columns = [col for col in df.columns if col.startswith('ball_type_')]

# Group by 'clusters' and sum the ball type columns
cluster_counts = df.groupby('cluster')[ball_type_columns].sum()

print(cluster_counts)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']

for cluster in cluster_counts.index:
    # Get the data for the current cluster
    cluster_data = cluster_counts.loc[cluster, ball_type_columns]
    
    # Sort the ball types in descending order and select the top four
    top_ball_types = cluster_data.sort_values(ascending=False).head(4)
    
    # Plot the data
    plt.figure(figsize=(10, 6))
    top_ball_types.plot(kind='barh', fontsize=20)
    plt.title(f'Ball Types in Cluster {cluster}')
    plt.xlabel('Count')
    plt.ylabel('Ball Type')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest count on top
    plt.show()