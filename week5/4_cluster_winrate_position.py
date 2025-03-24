import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from pytorch_tabnet.tab_model import TabNetClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

shot_file_name = 'cluster_final_POSITIONS.csv'
base_on = 'winning'

shot = pd.read_csv(shot_file_name, encoding='utf-8')
# Assuming final_df is your DataFrame and you want to exclude 'column_to_exclude' from scaling
columns_to_exclude = ['winning']
columns_to_scale = [col for col in shot.columns if col not in columns_to_exclude]
# Separating the columns
df_to_scale = shot[columns_to_scale]
df_not_to_scale = shot[columns_to_exclude]
# Scaling the selected columns
scaler = StandardScaler()
scaled_df = scaler.fit_transform(df_to_scale)
scaled_df = pd.DataFrame(scaled_df, columns=columns_to_scale)
# Concatenating the scaled columns with the unscaled columns
final_df = pd.concat([scaled_df, df_not_to_scale.reset_index(drop=True)], axis=1)
# Reorder columns to match the original DataFrame order
final_df = final_df[final_df.columns]
# drop those rows contains NaN
final_df = final_df.dropna()
print(final_df)
# final_df.to_csv('combined.csv', index=False, encoding='utf-8-sig')

feature_columns = [col for col in final_df.columns if col not in ['rally_id', 'set_id', 'match_id', 'fault_player', base_on]]
x = final_df[feature_columns]
y = final_df[base_on]

    # line 67-73 using RandomForest to train the model, and line 74-80 use Tabnet
    # please comment out the model which you dont want to use.

# rf_model = RandomForestClassifier(n_estimators=100, random_state=26)
# rfe = RFE(estimator=rf_model, n_features_to_select=50)  # Selecting top 50 features
# x_reduced = rfe.fit_transform(x, y)
# # start training
# X_train, X_test, y_train, y_test = train_test_split(x_reduced, y, test_size=0.2, random_state=26)
# # Define the resampling methods
# oversample = SMOTE()
# undersample = RandomUnderSampler()
# pipeline = Pipeline(steps=[('o', oversample), ('u', undersample), ('model', rf_model)])
# pipeline.fit(X_train, y_train)
# # Make predictions and evaluate
# y_pred = pipeline.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"\nModel Accuracy: {accuracy}")

tabnet_model = TabNetClassifier()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=26)
# Define the resampling methods
oversample = SMOTE()
undersample = RandomUnderSampler()
pipeline = Pipeline(steps=[('o', oversample), ('u', undersample), ('model', tabnet_model)])
pipeline.fit(X_train.values, y_train.values)
# Make predictions and evaluate
y_pred = pipeline.predict(X_test.values)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy}")

# Get feature importances
rf_model = pipeline.named_steps['model']
importances = rf_model.feature_importances_

# selected_features = x.columns[rfe.support_]
# feature_importance = pd.DataFrame({'feature': selected_features, 'importance': importances})

feature_importance = pd.DataFrame({'feature': x.columns, 'importance': importances})

feature_importance = feature_importance.sort_values('importance', ascending=False)

top_features = feature_importance.head(20)

font = FontProperties(fname='C:\Windows\Fonts\msjh.ttc')
# Plot feature importances
plt.figure(figsize=(10, 6))
plt.barh(top_features['feature'], top_features['importance'])
plt.yticks(fontproperties=font)
plt.xticks(fontproperties=font)
plt.gca().invert_yaxis()
plt.xlabel('importance')
plt.title(f'Top Feature Importance for \'{base_on}\'', fontproperties=font)
plt.text(0.98, 0.98, f'Accuracy: {accuracy:.2f}', fontsize=15, ha='right', va='top', transform=plt.gca().transAxes)
plt.tight_layout()
plt.show()

# Print top 10 most important features
# Resetting the index to add the ranking
feature_importance = feature_importance.reset_index(drop=True)
# Adding a rank column, which is the index + 1 (to start ranking from 1)
print()
print(feature_importance.head(20))
print(f"\nLeast Importance:\n")
print(feature_importance.tail(10))
print("\nclassification report: ")
print(classification_report(y_test, y_pred))