import pandas as pd
import chardet

# change these with "out", "in" or "winning" to see different result
rally_file_name = 'rally_out.csv'
base_on = 'out'

# Detect the file encoding
with open(rally_file_name, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']
print(f"Detected encoding: {file_encoding}")
# Read the CSV file with the detected encoding
rally = pd.read_csv(rally_file_name, 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})
# Detect the file encoding
with open('shot_grouped.csv', 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']
print(f"Detected encoding: {file_encoding}")
# Read the CSV file with the detected encoding
shot = pd.read_csv('shot_grouped.csv', 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})

shot = shot.drop(columns=['shot_1_player', 'shot_2_player', 'shot_3_player', 'shot_4_player', 'shot_5_player'])
shot = shot.drop(shot.filter(like='player_location').columns, axis=1)
shot = shot.drop(shot.filter(like='opponent_location').columns, axis=1)

final_df = pd.merge(shot, rally, on='rally_id', how='left')
final_df = final_df[final_df[base_on] != -1]
print(final_df)
# final_df.to_csv('combined.csv', index=False, encoding='utf-8-sig')

feature_columns = [col for col in final_df.columns if col not in ['rally_id', base_on]]
x = final_df[feature_columns]
y = final_df[base_on]

# start training
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=26)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=26)
rf_model.fit(X_train, y_train)

# Make predictions and evaluate
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

import matplotlib.pyplot as plt
# Get feature importances
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({'feature': x.columns, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(feature_importance['feature'], feature_importance['importance'])
plt.xticks(rotation=90)
plt.title('Feature Importance')
plt.tight_layout()
plt.show()

# Print top 10 most important features
print(feature_importance.head(50))