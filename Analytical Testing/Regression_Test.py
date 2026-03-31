from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
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

# X = PCA features
X = df[['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
        'V11','V12','V13','V14','V15','V16','V17','V18',
        'V19','V20','V21','V22','V23','V24','V25','V26','V27','V28']]

# -------- Predict Time --------
y_time = df['Time']

X_train, X_test, y_train, y_test = train_test_split(X, y_time, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Time R2:", r2_score(y_test, y_pred))
print("Time MSE:", mean_squared_error(y_test, y_pred))


# Scatter plot
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)

plt.xlabel("Actual Time")
plt.ylabel("Predicted Time")
plt.title("Reconstruction of Time from PCA Features")

# Ideal line (perfect prediction)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()

# -------- Predict Amount --------
y_amount = df['Amount']

X_train, X_test, y_train, y_test = train_test_split(X, y_amount, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Amount R2:", r2_score(y_test, y_pred))
print("Amount MSE:", mean_squared_error(y_test, y_pred))

import matplotlib.pyplot as plt

# Scatter plot
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.3)

plt.xlabel("Actual Amount")
plt.ylabel("Predicted Amount")
plt.title("Reconstruction of Amount from PCA Features")


# Ideal line (perfect prediction)
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val])

plt.show()


