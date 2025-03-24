# calculate the total number of shots
# divide it by total set number
# get the average number of shots per set

import pandas as pd

print("cauculate th average number of shots per matches, using rally data only")
print("use 'rally_shot_count' to calculate the total number of shots")
print("use unique 'match_id' to calculate the total number of matches")

original_rally = pd.read_csv('rally_0108.csv')
original_set = pd.read_csv('set_0108.csv')

columns_to_keep = ["rally_id", "set_id", "rally_number", "rally_shot_count"]
rally_data = original_rally[columns_to_keep]
rally_data = rally_data.dropna(subset=columns_to_keep)

columns_to_keep = ["set_id", "match_id"]
set_data = original_set[columns_to_keep]
set_data = set_data.dropna(subset=columns_to_keep)

rally_data = pd.merge(rally_data, set_data, on="set_id", how="inner")

total_shots = rally_data["rally_shot_count"].sum()
total_matches = rally_data["match_id"].nunique()

average_shots_per_set = total_shots / total_matches if total_matches > 0 else 0

print(f"Total number of shots: {total_shots}")
print(f"Total number of unique matches: {total_matches}")
print(f"Average number of shots per match: {average_shots_per_set:.2f}")
print("---------------------------------")

# lets change to another approach
print("cauculate th average number of shots per matches, combine shots data and rally data")
print("count how many rows in shot data to calculate the total number of shots")
print("use unique 'match_id' to calculate the total number of matches")

original_shots = pd.read_csv('convert_shot.csv')
original_rally = pd.read_csv('rally_0108.csv')
original_set = pd.read_csv('set_0108.csv')

columns_to_keep = ["shot_id", "rally_id"]
shot_data = original_shots[columns_to_keep]
shot_data = shot_data.dropna(subset=columns_to_keep)

columns_to_keep = ["rally_id", "set_id"]
rally_data = original_rally[columns_to_keep]
rally_data = rally_data.dropna(subset=columns_to_keep)

columns_to_keep = ["set_id", "match_id"]
set_data = original_set[columns_to_keep]
set_data = set_data.dropna(subset=columns_to_keep)

merged_data = pd.merge(shot_data, rally_data, on="rally_id", how="inner")
merged_data = pd.merge(merged_data, set_data, on="set_id", how="inner")

total_shots = merged_data["shot_id"].nunique()
total_matches = merged_data["match_id"].nunique()

average_shots_per_set = total_shots / total_matches if total_matches > 0 else 0

print(f"Total number of shots: {total_shots}")
print(f"Total number of unique matches: {total_matches}")
print(f"Average number of shots per match: {average_shots_per_set:.2f}")



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# this section is to drop unwanted columns and rows with NaN values

# convert_shot 是惠晴給我的
original_shots = pd.read_csv('convert_shot.csv')
original_rally = pd.read_csv('rally_0108.csv')
original_set = pd.read_csv('set_0108.csv')

# print("Original shots data columns: ")
# print(original_shots.columns)
# print("---------------------------------")

# nan_count = original_shots.isna().sum()
# print("Number of NaN values in each column: ")
# print(nan_count)
# print("---------------------------------")

# drop nan values in shots data
print("drop nan values in shots data")
columns_to_check = ["shot_id", "shot_type",
                    "rally_id", "shot_number", "frame_num", "end_frame_num" ]
drop_nan_shots = original_shots.dropna(subset=columns_to_check)
# print("Number of NaN values in each column after dropping NaN values: ")
# print(drop_nan_shots.isna().sum())
# print("data after dropping NaN values: ", len(drop_nan_shots))

# drop columns unrelated in shots data
print("drop columns unrelated in shots data")
columns_to_drop = original_shots.columns.difference(columns_to_check)
drop_nan_and_unrelated_shots = drop_nan_shots.drop(columns=columns_to_drop)
print("---------------------------------")

# drop nan values in rally data
print("drop nan values in rally data")
columns_to_check = ["rally_id", "set_id", "rally_number", "rally_current_winner_score",
                    "rally_current_loser_score", "rally_shot_count"]
drop_nan_rally = original_rally.dropna(subset=columns_to_check)
print("data after dropping NaN values: drop_nan_rally")

# drop columns unrelated in rally data
print("drop columns unrelated in rally data")
columns_to_drop = drop_nan_rally.columns.difference(columns_to_check)
drop_nan_and_unrelated_rally = drop_nan_rally.drop(columns=columns_to_drop)

# drop nan values in set data
print("drop nan values in set data")
columns_to_check = ["set_id", "match_id", "set_number"]
drop_nan_set = original_set.dropna(subset=columns_to_check)

# drop columns unrelated in set data
print("drop columns unrelated in set data")
columns_to_drop = drop_nan_set.columns.difference(columns_to_check)
drop_nan_and_unrelated_set = drop_nan_set.drop(columns=columns_to_drop)

print("merge shots and rally and set data")
shots_merged = pd.merge(drop_nan_and_unrelated_shots, drop_nan_and_unrelated_rally, on="rally_id", how="inner")
shots_merged = pd.merge(shots_merged, drop_nan_and_unrelated_set, on="set_id", how="inner")
print("length of data after merging: ", len(shots_merged))

# print("columns of data after merging: ")
# print(shots_with_set_id.columns)
# print("---------------------------------")

# print("output data to csv file")
# shots_with_set_id.to_csv('shots_with_set_id.csv', index=False)
# print("file name: shots_with_set_id.csv")
# print("done")
# print("---------------------------------")

# print("print how many 0 in end_frame_num")
# print(shots_with_set_id["end_frame_num"].value_counts())
least_unique_numbers = shots_merged["end_frame_num"].dropna().unique()  # Get unique values
least_unique_numbers.sort()  # Sort them in ascending order

# Select the first 10 smallest values
top_10_least_numbers = least_unique_numbers[:10]
print("10 smallest unique numbers in 'end_frame_num':")
print(top_10_least_numbers)

# print("print the first 20 rows of the data")
# print(shots_with_set_id.head(30))


print("columns of data after merging: ")
print(shots_merged.columns)




import pandas as pd

mapping = {
    '過度切球': '切球',
    '防守回抽': '平球',
    '後場抽平球': '平球',
    '防守回挑': '挑球',
    '推球': '推撲球',
    '撲球': '推撲球',
    '擋小球': '網前小球',
    '勾球': '網前小球',
    '放小球': '網前小球',
    '小平球': '網前小球',
    '點扣': '殺球'
}
shots_merged['shot_type'] = shots_merged['shot_type'].replace(mapping)

def analyze_balltypes_detailed(df):
    if 'shot_type' not in df.columns:
        return "Error: 'shot_type_' column not found in the DataFrame"
    
    # Get value counts
    shot_type__counts = df['shot_type'].value_counts()
    total_rows = len(df)
    
    print(f"Analysis of ball types in the DataFrame")
    print(f"Total rows in DataFrame: {total_rows}")
    print(f"Total unique ball types: {len(shot_type__counts)}")
    print("\nBreakdown by ball type:")
    print("-" * 40)
    
    for shot_type, count in shot_type__counts.items():
        percentage = (count / total_rows) * 100
        print(f"Ball Type: {shot_type}")
        print(f"Count: {count} rows")
        print("-" * 40)
    
    return shot_type__counts.to_dict()

# Example usage:
result = analyze_balltypes_detailed(shots_merged)
print(result)




def process_shots_by_match(df):
    # Create new columns to store results
    df['shot_duration'] = 0
    df['match_part'] = 0  # 1 for first part (1-284), 2 for second part (285-568), 3 for third part (>=569)
    df['match_shot_counter'] = 0
    
    # Track shots per match and rally
    match_shot_counts = {}  # Dictionary to track shots per match
    current_rally_id = None
    rally_shot_counter = 0
    
    # Iterate through each row
    for index, row in df.iterrows():
        # 1. Calculate shot duration
        if row['end_frame_num'] <= row['frame_num']:
            df.at[index, 'shot_duration'] = -1
        else:
            df.at[index, 'shot_duration'] = row['end_frame_num'] - row['frame_num']
            
        # 2. Check which part of match the shot belongs to
        match_id = row['match_id']
        
        # Initialize match counter if not exists
        if match_id not in match_shot_counts:
            match_shot_counts[match_id] = 0
            
        # Rally tracking
        if current_rally_id != row['rally_id']:
            # When rally changes, verify shot count
            if current_rally_id is not None:
                # Get the rally_shot_count for the previous rally
                prev_rally_data = df[df['rally_id'] == current_rally_id].iloc[0]
                expected_count = prev_rally_data['rally_shot_count']
                
                # If actual count doesn't match expected, adjust match count
                if rally_shot_counter != expected_count:
                    match_shot_counts[match_id] -= rally_shot_counter  # Remove incorrect count
                    match_shot_counts[match_id] += expected_count     # Add correct count
            
            # Reset for new rally
            current_rally_id = row['rally_id']
            rally_shot_counter = 0
            
        # Increment counters
        rally_shot_counter += 1
        match_shot_counts[match_id] += 1
        
        # Determine which part of the match (1-3)
        shot_count = match_shot_counts[match_id]
        if shot_count <= 284:
            df.at[index, 'match_part'] = 1
        elif shot_count <= 568:
            df.at[index, 'match_part'] = 2
        else:
            df.at[index, 'match_part'] = 3
            
        df.at[index, 'match_shot_counter'] = match_shot_counts[match_id]
    
    # Final rally verification (for the last rally in the dataframe)
    if current_rally_id is not None:
        last_rally_data = df[df['rally_id'] == current_rally_id].iloc[0]
        expected_count = last_rally_data['rally_shot_count']
        if rally_shot_counter != expected_count:
            match_id = last_rally_data['match_id']
            match_shot_counts[match_id] -= rally_shot_counter
            match_shot_counts[match_id] += expected_count
            
            # Update match_part for all shots in this last rally
            shot_count = match_shot_counts[match_id]
            part = 1 if shot_count <= 284 else (2 if shot_count <= 568 else 3)
            for idx in df[df['rally_id'] == current_rally_id].index:
                df.at[idx, 'match_part'] = part
    
    return df

# Example usage:
shots_processed = process_shots_by_match(shots_merged)
# Save the processed data to a new csv file
shots_processed.to_csv('shots_processed_by_match.csv', index=False)
print("How many rows in the processed data: ", len(shots_processed))
print("How many shots are in each part of the match:")
print("First part (1-284): ", len(shots_processed[shots_processed['match_part'] == 1]))
print("Second part (285-568): ", len(shots_processed[shots_processed['match_part'] == 2]))
print("Third part (>=569): ", len(shots_processed[shots_processed['match_part'] == 3]))
print(shots_processed.columns)




def analyze_shot_type_by_match_parts(df):
    # Check required columns
    if 'shot_type' not in df.columns or 'match_part' not in df.columns:
        return "Error: Required columns 'shot_type' or 'match_part' not found"
    
    # Split into three parts of match
    first_part = df[df['match_part'] == 1]
    second_part = df[df['match_part'] == 2]
    third_part = df[df['match_part'] == 3]
    
    # Get counts
    first_part_counts = first_part['shot_type'].value_counts()
    second_part_counts = second_part['shot_type'].value_counts()
    third_part_counts = third_part['shot_type'].value_counts()
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'First_Part': first_part_counts,
        'Second_Part': second_part_counts,
        'Third_Part': third_part_counts
    }).fillna(0).astype(int)
    
    # Add percentages
    first_part_total = len(first_part)
    second_part_total = len(second_part)
    third_part_total = len(third_part)
    
    summary['First_Part_%'] = (summary['First_Part'] / first_part_total * 100).round(2)
    summary['Second_Part_%'] = (summary['Second_Part'] / second_part_total * 100).round(2)
    summary['Third_Part_%'] = (summary['Third_Part'] / third_part_total * 100).round(2)
    
    # Add percentage difference (Third Part % - First Part %)
    summary['Percentage_Diff'] = (summary['Third_Part_%'] - summary['First_Part_%']).round(2)
    
    # Print results
    print(f"Shot Type Distribution Analysis by Match Parts")
    print(f"Total shots - First Part (1-284): {first_part_total}, Second Part (285-568): {second_part_total}, Third Part (>=569): {third_part_total}")
    print("\nSummary Table:")
    print(summary)
    
    # Print observations for significant changes
    print("\nObservations (Significant Percentage Changes between First and Third Part):")
    for shot_type in summary.index:
        diff = summary.loc[shot_type, 'Percentage_Diff']
        if abs(diff) > 5:  # Highlight changes > 5%
            direction = "decreased" if diff < 0 else "increased"
            print(f"- {shot_type}: Usage {direction} by {abs(diff):.2f}% from first part to third part of match")
    
    return summary

# Example usage:
result = analyze_shot_type_by_match_parts(shots_processed)



def analyze_shot_type_durations_by_match_parts(df):
    # Define the IQR multiplier (change this value to adjust outlier threshold)
    IQR_MULTIPLIER = 3  # You can change this to 1.5, 3, etc., as needed
    
    # Check required columns
    required_cols = ['shot_type', 'match_part', 'shot_duration']
    if not all(col in df.columns for col in required_cols):
        return "Error: Required columns not found"
    
    # Split into three parts of match
    first_part = df[df['match_part'] == 1]
    second_part = df[df['match_part'] == 2]
    third_part = df[df['match_part'] == 3]
    
    # Exclude invalid durations (-1)
    first_part_valid = first_part[first_part['shot_duration'] != -1]
    second_part_valid = second_part[second_part['shot_duration'] != -1]
    third_part_valid = third_part[third_part['shot_duration'] != -1]
    
    # Function to remove outliers and calculate bounds
    def process_group(group):
        Q1 = group['shot_duration'].quantile(0.25)
        Q3 = group['shot_duration'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - IQR_MULTIPLIER * IQR
        upper_bound = Q3 + IQR_MULTIPLIER * IQR
        # Remove outliers
        cleaned = group[(group['shot_duration'] >= lower_bound) & (group['shot_duration'] <= upper_bound)]
        # Calculate stats on cleaned data
        stats = {
            'mean': cleaned['shot_duration'].mean(),
            'q25': cleaned['shot_duration'].quantile(0.25),
            'median': cleaned['shot_duration'].median(),
            'q75': cleaned['shot_duration'].quantile(0.75),
            'iqr': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        return pd.Series(stats)
    
    # Process data
    first_part_stats = first_part_valid.groupby('shot_type').apply(process_group).round(2)
    second_part_stats = second_part_valid.groupby('shot_type').apply(process_group).round(2)
    third_part_stats = third_part_valid.groupby('shot_type').apply(process_group).round(2)
    
    # Count shots after cleaning
    def clean_group(group, multiplier=IQR_MULTIPLIER):
        return group[(group['shot_duration'] >= group['shot_duration'].quantile(0.25) - multiplier * (group['shot_duration'].quantile(0.75) - group['shot_duration'].quantile(0.25))) &
                    (group['shot_duration'] <= group['shot_duration'].quantile(0.75) + multiplier * (group['shot_duration'].quantile(0.75) - group['shot_duration'].quantile(0.25)))]
    
    first_part_clean = first_part_valid.groupby('shot_type').apply(clean_group).reset_index(drop=True)
    second_part_clean = second_part_valid.groupby('shot_type').apply(clean_group).reset_index(drop=True)
    third_part_clean = third_part_valid.groupby('shot_type').apply(clean_group).reset_index(drop=True)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'First_Part_Avg': first_part_stats['mean'],
        'First_Part_Q25': first_part_stats['q25'],
        'First_Part_Median': first_part_stats['median'],
        'First_Part_Q75': first_part_stats['q75'],
        'First_Part_IQR': first_part_stats['iqr'],
        'First_Part_Lower_Bound': first_part_stats['lower_bound'],
        'First_Part_Upper_Bound': first_part_stats['upper_bound'],
        'Second_Part_Avg': second_part_stats['mean'],
        'Second_Part_Q25': second_part_stats['q25'],
        'Second_Part_Median': second_part_stats['median'],
        'Second_Part_Q75': second_part_stats['q75'],
        'Second_Part_IQR': second_part_stats['iqr'],
        'Second_Part_Lower_Bound': second_part_stats['lower_bound'],
        'Second_Part_Upper_Bound': second_part_stats['upper_bound'],
        'Third_Part_Avg': third_part_stats['mean'],
        'Third_Part_Q25': third_part_stats['q25'],
        'Third_Part_Median': third_part_stats['median'],
        'Third_Part_Q75': third_part_stats['q75'],
        'Third_Part_IQR': third_part_stats['iqr'],
        'Third_Part_Lower_Bound': third_part_stats['lower_bound'],
        'Third_Part_Upper_Bound': third_part_stats['upper_bound']
    }).fillna(0)
    
    # Print results
    print(f"Shot Type Duration Analysis by Match Parts (Outliers Removed with {IQR_MULTIPLIER}*IQR Rule)")
    print(f"Total shots before cleaning:")
    print(f"First Part (1-284): {len(first_part_valid)}")
    print(f"Second Part (285-568): {len(second_part_valid)}")
    print(f"Third Part (>=569): {len(third_part_valid)}")
    print(f"\nTotal shots after cleaning:")
    print(f"First Part (1-284): {len(first_part_clean)}")
    print(f"Second Part (285-568): {len(second_part_clean)}")
    print(f"Third Part (>=569): {len(third_part_clean)}")
    print("(Durations in frames, -1 values and outliers excluded)")
    print("\nSummary:")
    
    # Print stats for each shot type
    for shot_type in summary.index:
        print(f"\nShot Type: {shot_type}")
        print(f"First Part - Avg: {summary.loc[shot_type, 'First_Part_Avg']}, "
              f"Q25: {summary.loc[shot_type, 'First_Part_Q25']}, "
              f"Median: {summary.loc[shot_type, 'First_Part_Median']}, "
              f"Q75: {summary.loc[shot_type, 'First_Part_Q75']}, "
              f"IQR: {summary.loc[shot_type, 'First_Part_IQR']}")
        print(f"Second Part - Avg: {summary.loc[shot_type, 'Second_Part_Avg']}, "
              f"Q25: {summary.loc[shot_type, 'Second_Part_Q25']}, "
              f"Median: {summary.loc[shot_type, 'Second_Part_Median']}, "
              f"Q75: {summary.loc[shot_type, 'Second_Part_Q75']}, "
              f"IQR: {summary.loc[shot_type, 'Second_Part_IQR']}")
        print(f"Third Part - Avg: {summary.loc[shot_type, 'Third_Part_Avg']}, "
              f"Q25: {summary.loc[shot_type, 'Third_Part_Q25']}, "
              f"Median: {summary.loc[shot_type, 'Third_Part_Median']}, "
              f"Q75: {summary.loc[shot_type, 'Third_Part_Q75']}, "
              f"IQR: {summary.loc[shot_type, 'Third_Part_IQR']}")
    
    # Print observations for average duration changes
    print("\nObservations (Average Duration Changes between First and Third Part):")
    for shot_type in summary.index:
        first_avg = summary.loc[shot_type, 'First_Part_Avg']
        third_avg = summary.loc[shot_type, 'Third_Part_Avg']
        if first_avg > 0 and third_avg > 0:  # Only compare if both exist
            diff = third_avg - first_avg
            if abs(diff) > 5:  # Highlight significant changes (>5 frames)
                direction = "increased" if diff > 0 else "decreased"
                print(f"- {shot_type}: Average duration {direction} by {abs(diff):.2f} frames from first part to third part of match")
    
    return summary

# Example usage:
result = analyze_shot_type_durations_by_match_parts(shots_processed)



def plot_average_durations_by_part(df):
    # First get the processed statistics
    stats_summary = analyze_shot_type_durations_by_match_parts(df)
    
    # Set style
    plt.style.use('seaborn')
    
    # Set font for Chinese characters
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']  # Try these fonts
    plt.rcParams['axes.unicode_minus'] = False  # Correctly display minus sign
    
    # Get shot types (excluding '未知球種')
    shot_types = [st for st in stats_summary.index if st != '未知球種']
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))  # Reduced from (20, 15)
    
    # Define colors for each part
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    labels = ['First Part\n(1-284)', 'Second Part\n(285-568)', 'Third Part\n(≥569)']
    
    # Set bar width and positions
    bar_width = 0.25  # Back to original narrow width
    x = np.arange(len(labels))  # the label locations
    
    # Create a subplot for each shot type
    for idx, shot_type in enumerate(shot_types, 1):
        ax = fig.add_subplot(3, 3, idx)
        
        # Get the pre-calculated averages
        averages = [
            stats_summary.loc[shot_type, 'First_Part_Avg'],
            stats_summary.loc[shot_type, 'Second_Part_Avg'],
            stats_summary.loc[shot_type, 'Third_Part_Avg']
        ]
        
        # Create bar plot with specified width
        bars = ax.bar(x, averages, bar_width, color=colors)
        
        # Add title and labels with larger font size for better readability
        ax.set_title(f'{shot_type}\nAverage Duration by Match Part', fontsize=14, pad=10)
        ax.set_ylabel('Average Duration (frames)', fontsize=12)
        
        # Set x-axis ticks and labels
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        
        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set reasonable y-axis limits with some padding
        ymax = max(averages) * 1.15  # Add 15% padding above highest bar
        ax.set_ylim(0, ymax)
        
        # Set x-axis limits to reduce spacing
        ax.set_xlim(-0.5, len(labels) - 0.5)
    
    # Adjust layout
    plt.tight_layout(h_pad=2, w_pad=2)  # Reduced padding between subplots
    plt.suptitle('Average Shot Duration by Match Part for Each Shot Type\n(Outliers Removed)', 
                fontsize=16, y=1.02)  # Slightly reduced title font size
    
    # Save the plot with higher DPI
    plt.savefig('average_shot_durations.png', dpi=300, bbox_inches='tight')
    plt.close()

# Example usage:
plot_average_durations_by_part(shots_processed)
print("Bar plot has been saved as 'average_shot_durations.png'")