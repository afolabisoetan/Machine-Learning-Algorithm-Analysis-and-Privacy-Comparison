import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.svm import LinearSVC

# Download and Load Dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)
df = pd.read_csv(full_path)

# Data Setup
y = df['Class']
x = df.drop('Class', axis=1)

# Split for initial testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear SVC Classifier Test
print("=======================================================================================")
print("Linear SVC Classifier Test (Full Set)")

# defining the cross-validation strategy here
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Pipeline with Linear SVC
# dual=False is recommended when n_samples > n_features (which is true for your dataset)
pipeline_svc = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', LinearSVC(class_weight='balanced', dual=False, max_iter=2000))
])

# Running cross-validation
svc_scores = cross_val_score(pipeline_svc, x, y, scoring='recall', cv=cv, n_jobs=-1)

# saves score values into a CSV (can be commented out when the CSV is already made)
#results_df = pd.DataFrame({'Iteration': np.arange(1, len(svc_scores)+1), 'Full_Dataset_Recall': svc_scores})
#results_df.to_csv('Linear_SVC_Recall_Values_Full_Set.csv', index=False)

# Final report on 80/20 split
pipeline_svc.fit(x_train, y_train)
y_pred = pipeline_svc.predict(x_test)

print("--- Linear SVC Classification Report (Full Dataset)---")
print(classification_report(y_test, y_pred))
print(f"Mean Cross Validation Recall: {svc_scores.mean():.4f}")

# Visualizing results
sns.set_context("paper", font_scale=3.0)
plt.figure(figsize=(8, 5))
sns.boxplot(x=svc_scores)
plt.title('Recall Scores - Linear SVC (Full Dataset)')
plt.xlabel('Recall Score')
plt.show()