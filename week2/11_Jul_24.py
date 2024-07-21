import pandas as pd
import sklearn

set = pd.read_csv('cleaned_set_0108.csv')
player = pd.read_csv('player_0108.csv')

set_subset = set[['set_id', 'match_id', 'set_winner', 'set_loser']]
player_subset = player[["player_id", "player_height"]]

merged_winner = pd.merge(set_subset, player_subset, left_on='set_winner', right_on='player_id', how='inner')
merged_winner = merged_winner.rename(columns={'player_height': 'winner_height'})
merged_loser = pd.merge(merged_winner, player_subset, left_on="set_loser", right_on='player_id', how='inner')
merged_loser = merged_loser.rename(columns={'player_height': 'loser_height'})
merged = merged_loser[merged_loser['winner_height'] != 0]
merged = merged[merged['loser_height'] != 0]

merged['height_difference'] = merged['winner_height'] - merged['loser_height']
print(merged)
sum = merged['height_difference'].sum()
print(f'sum: {sum}')

merged.to_csv('processed.csv', index=False)
