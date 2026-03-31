import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. Download and Load Dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))


# 2. Data Minimization (PCA-Only Dataset)
df.drop(columns=['Amount', 'Time'], inplace=True)


X = df.drop('Class', axis=1)
y = df['Class']


# 3. Setting up Cross-Validator (30 total iterations)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)


# 4. Define Pipelines
pos_weight = (len(y) - sum(y)) / sum(y)


rf_pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('rf', RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42))
])


xgb_pipeline = Pipeline([
  ('scaler', StandardScaler()),
  ('xgb', XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42))
])


# 5. Running Cross-Validation for Multiple Metrics
def get_scores(pipeline, name):
   print(f"Running Cross-Validation for {name}...")
   recall = cross_val_score(pipeline, X, y, scoring='recall', cv=cv, n_jobs=-1)
   precision = cross_val_score(pipeline, X, y, scoring='precision', cv=cv, n_jobs=-1)
   f1 = cross_val_score(pipeline, X, y, scoring='f1', cv=cv, n_jobs=-1)
   return recall, precision, f1


rf_recall, rf_precision, rf_f1 = get_scores(rf_pipeline, "Random Forest")
xgb_recall, xgb_precision, xgb_f1 = get_scores(xgb_pipeline, "XGBoost")


# 6. Save Recall Scores (Matching your specific format for Part B)
pd.DataFrame({'Iteration': np.arange(1, 31), 'PCA_Only_Dataset_Recall': rf_recall})\
   .to_csv('rf_fraud_detection_results_secure.csv', index=False)


pd.DataFrame({'Iteration': np.arange(1, 31), 'PCA_Only_Dataset_Recall': xgb_recall})\
   .to_csv('xgb_fraud_detection_results_secure.csv', index=False)


# 7. Visualization: Individual Boxplots for each Classifier
sns.set_context("paper", font_scale=3.0)
sns.set_style("whitegrid")


# Figure 1: Random Forest
plt.figure(figsize=(8, 5))
sns.boxplot(x=rf_recall, color="skyblue")
# Optional: Add swarmplot to show the 30 individual points like before
sns.swarmplot(x=rf_recall, color=".25", alpha=0.5)
plt.title('Distribution of Recall Scores - PCA Only DataSet')
plt.xlabel('Recall Score')
plt.xlim(0, 1.05)
plt.tight_layout()
plt.show()


# Figure 2: XGBoost
plt.figure(figsize=(8, 5))
sns.boxplot(x=xgb_recall, color="skyblue")
sns.swarmplot(x=xgb_recall, color=".25", alpha=0.5)
plt.title('Distribution of Recall Scores - PCA Only DataSet')
plt.xlabel('Recall Score')
plt.xlim(0, 1.05)
plt.tight_layout()
plt.show()


# 8. Detailed Comparison Output
print("\n================ FINAL RESULTS (30 ITERATIONS) ================")
print(f"{'Metric':<15} | {'Random Forest':<15} | {'XGBoost':<15}")
print("-" * 50)
print(f"{'Mean Recall':<15} | {rf_recall.mean():.4f}          | {xgb_recall.mean():.4f}")
print(f"{'Mean Precision':<15} | {rf_precision.mean():.4f}          | {xgb_precision.mean():.4f}")
print(f"{'Mean F1-Score':<15} | {rf_f1.mean():.4f}          | {xgb_f1.mean():.4f}")
print("===============================================================")
