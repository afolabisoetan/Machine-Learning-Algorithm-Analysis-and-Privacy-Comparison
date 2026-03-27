import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Download and Load Dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
files = os.listdir(path)
csv_file = [f for f in files if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)
df = pd.read_csv(full_path)

# removes the human-readable columns
df.drop(columns=['Amount'], inplace=True)
df.drop(columns=['Time'], inplace=True)

# Data Setup
y = df['Class']
x = df.drop('Class', axis=1)

# Split for initial testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Logistic Regression Classifier Test
print("=======================================================================================")
print("Logistic Regression Classifier Test (PCA Only Set)")

# Setting up cross-validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Pipeline with Logistic Regression
# Note: solver='lbfgs' is default, max_iter increased for convergence
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000))
])

# Running cross-validation
secure_scores = cross_val_score(pipeline, x, y, scoring='recall', cv=cv, n_jobs=-1)

# saves score values into a CSV (can be commented out when the CSV is already made)
#results_df = pd.DataFrame({'Iteration': np.arange(1, len(secure_scores)+1), 'PCA_Only_Dataset_Recall': secure_scores})
#results_df.to_csv('LR_Recall_Values_PCA_Only_Set.csv', index=False)

# Final report on 80/20 split
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print("--- Logistic Regression Classification Report (PCA Only Dataset)---")
print(classification_report(y_test, y_pred))
print(f"Mean Cross Validation Recall: {secure_scores.mean():.4f}")

# Visualizing results
sns.set_context("paper", font_scale=3.0)
plt.figure(figsize=(8, 5))
sns.boxplot(x=secure_scores)
plt.title('Recall Scores - Logistic Regression (PCA Only Dataset)')
plt.xlabel('Recall Score')
plt.show()