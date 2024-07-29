import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

shot_file_name = '01_shot.csv'
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
with open(shot_file_name, 'rb') as file:
    raw_data = file.read()
    result = chardet.detect(raw_data)
    file_encoding = result['encoding']
print(f"Detected encoding: {file_encoding}")
# Read the CSV file with the detected encoding
shot = pd.read_csv(shot_file_name, 
                 encoding=file_encoding,
                 na_values=['', 'nan', 'NULL'],
                 keep_default_na=True,
                 dtype={'shot_id': int})

shot = shot.drop(columns=['shot_1_player', 'shot_2_player', 'shot_3_player', 'shot_4_player', 'shot_5_player'])
# shot = shot.drop(shot.filter(like='player_location').columns, axis=1)
# shot = shot.drop(shot.filter(like='opponent_location').columns, axis=1)

final_df = pd.merge(shot, rally, on='rally_id', how='left')
final_df = final_df[final_df[base_on] != -1]

print(final_df)
# final_df.to_csv('combined.csv', index=False, encoding='utf-8-sig')

feature_columns = [col for col in final_df.columns if col not in ['rally_id', base_on]]
x = final_df[feature_columns]
y = final_df[base_on]

# start training
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=26)
# Define the resampling methods
oversample = SMOTE()
undersample = RandomUnderSampler()

    # line 67-73 using RandomForest to train the model, and line 74-80 use Tabnet
    # please comment out the model which you don want to use.

# rf_model = RandomForestClassifier(n_estimators=100, random_state=26)
# pipeline = Pipeline(steps=[('o', oversample), ('u', undersample), ('model', rf_model)])
# pipeline.fit(X_train, y_train)
# # Make predictions and evaluate
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy}")
tabnet_model = TabNetClassifier()
pipeline = Pipeline(steps=[('o', oversample), ('u', undersample), ('model', tabnet_model)])
pipeline.fit(X_train.values, y_train.values)
# Make predictions and evaluate
y_pred = pipeline.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}")

# Get feature importances
rf_model = pipeline.named_steps['model']
importances = rf_model.feature_importances_
feature_importance = pd.DataFrame({'feature': x.columns, 'importance': importances})
feature_importance = feature_importance.sort_values('importance', ascending=False)

top_features = feature_importance.head(25)

font = FontProperties(fname='C:\Windows\Fonts\msjh.ttc')
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.yticks(fontproperties=font)
plt.xticks(fontproperties=font)
plt.gca().invert_yaxis()
plt.xlabel('importance')
plt.title(f'Top Feature Importance for \'{base_on}\'', fontproperties=font)
plt.tight_layout()
plt.show()

# Print top 10 most important features
# Resetting the index to add the ranking
feature_importance = feature_importance.reset_index(drop=True)
# Adding a rank column, which is the index + 1 (to start ranking from 1)
print()
print(feature_importance.head(50))
print("\nclassification report: ")
print(classification_report(y_test, y_pred))