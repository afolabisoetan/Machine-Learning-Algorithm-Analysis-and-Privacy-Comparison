import kagglehub
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
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

# KNN classifier test (Full set)
print("=======================================================================================")
print("KNN Classifier Test (Full Set)")

y = df['Class']
x = df.drop('Class', axis=1)
cols = x.columns



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# initialize the Scaler and scale the feature values so that PCA components are not drowned out by Amount and Class
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# initialize and train KNN classifier (weights = 'distance' helps balance the imbalanced data)

from sklearn.metrics import recall_score, precision_score, f1_score

# Lists to store results
k_values = []
recalls = []
f1_scores = []

# Range: K should usually be an odd number to avoid ties (2 classes: 'Fraud' and 'Not fraud')
for i in range(3, 15, 2):
    knn = KNeighborsClassifier(n_neighbors=i, weights='distance')
    knn.fit(x_train_scaled, y_train)

    y_pred = knn.predict(x_test_scaled)

    # calculate metrics
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    k_values.append(i)
    recalls.append(rec)
    f1_scores.append(f1)

    print(f"k={i} | Recall: {rec:.4f} | F1: {f1:.4f}")

# Plotting the results to find the "elbow"
plt.figure(figsize=(10, 6))
plt.plot(k_values, recalls, label='Recall', marker='o')
plt.plot(k_values, f1_scores, label='F1-Score', marker='s')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Score')
plt.title('Selecting K for Fraud Detection')
plt.legend()
plt.show()


# sets up cross-validator
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# testing shows k = 7 is the optimum k value therefore run the classification_report with k = 7
knnOptimum = KNeighborsClassifier(n_neighbors=7, weights='distance')

# initialize the Scaler and scale the feature values so that PCA components are not drowned out by Amount and Class
# using a Pipeline to prevent data leakage during cross-validation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', knnOptimum)
])

# Running cross-validation for the full scores distribution
full_scores = cross_val_score(pipeline, x, y, scoring='recall', cv=cv, n_jobs=-1)

# saves score values into a CSV
results_df = pd.DataFrame({ 'Iteration': np.arange(1, len(full_scores)+1), 'Full_Dataset_Recall': full_scores})
results_df.to_csv('knn_fraud_detection_results.csv', index=False)

# Training on the 80/20 split for the final classification report
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

print("--- Optimal KNN Classification Report (Full Dataset)---")
print(classification_report(y_test, y_pred))
print(f"Mean Cross Validation Recall: {full_scores.mean():.4f}")

print()
print()
print("=======================================================================================")

print()
print()
print("=======================================================================================")

# Boxplot of the 30 recall scores to visualize stability
plt.figure(figsize=(8, 5))
sns.boxplot(x=full_scores)
plt.title('Distribution of Recall Scores - Full Dataset')
plt.xlabel('Recall Score')
plt.show()
