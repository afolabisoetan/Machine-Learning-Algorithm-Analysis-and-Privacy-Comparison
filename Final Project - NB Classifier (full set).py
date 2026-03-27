import kagglehub
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import recall_score, precision_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score

# Download the dataset
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to folder:", path)

# lists the files in that folder to find the CSV name
files = os.listdir(path)
print("Files in folder:", files)
print()
print()

# joins the folder path with the filename (creditcard.csv)
# retrieves the first file that ends in .csv
csv_file = [f for f in files if f.endswith('.csv')][0]
full_path = os.path.join(path, csv_file)

# reads the actual file
df = pd.read_csv(full_path)

# Naive Bayes classifier test (Full set)
print("=======================================================================================")
print("Naive Bayes Classifier Test (Full Set)")

y = df['Class']
x = df.drop('Class', axis=1)
cols = x.columns

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# initialize the Scaler (StandardScaler is still good practice, though NB is robust)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# initialize and train Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(x_train_scaled, y_train)

y_pred = nb_model.predict(x_test_scaled)

# calculate metrics for the initial split
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Initial Split | Recall: {rec:.4f} | F1: {f1:.4f}")

# sets up cross-validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# using a Pipeline to prevent data leakage during cross-validation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nb', GaussianNB())
])

# Running cross-validation for the full scores distribution
full_scores = cross_val_score(pipeline, x, y, scoring='recall', cv=cv, n_jobs=-1)

# saves score values into a CSV
results_df = pd.DataFrame({ 'Iteration': np.arange(1, len(full_scores)+1), 'Full_Dataset_Recall': full_scores})
results_df.to_csv('nb_fraud_detection_results.csv', index=False)

# Training on the 80/20 split for the final classification report
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print("--- Naive Bayes Classification Report (Full Dataset)---")
print(classification_report(y_test, y_pred))
print(f"Mean Cross Validation Recall: {full_scores.mean():.4f}")

print()
print()
print("=======================================================================================")

print()
print()
print("=======================================================================================")

# Boxplot of the 30 recall scores to visualize stability
sns.set_context("paper", font_scale=3.0)
plt.figure(figsize=(8, 5))
sns.boxplot(x=full_scores)
plt.title('Distribution of Recall Scores - Naive Bayes (Full Dataset)')
plt.xlabel('Recall Score')
plt.show()