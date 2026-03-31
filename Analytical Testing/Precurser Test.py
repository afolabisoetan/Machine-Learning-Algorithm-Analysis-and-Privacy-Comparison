import kagglehub
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
csv_file = [f for f in os.listdir(path) if f.endswith('.csv')][0]
df = pd.read_csv(os.path.join(path, csv_file))


y = df['Class']
X_human = df[['Amount', 'Time']]
pos_weight = (len(y) - sum(y)) / sum(y)
models = {
    'KNN': KNeighborsClassifier(n_neighbors=7, weights='distance'),
    'NB': GaussianNB(),
    'LR': LogisticRegression(),
    'Linear_SVC': LinearSVC(),
    'RF': RandomForestClassifier(n_estimators=100, n_jobs=1, random_state=42),
    'xgb': XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42),
}


cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)


results = {}

print("--- Human-Readable Feature Security Audit ---")
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    scores = cross_val_score(pipeline, X_human, y, scoring='recall', cv=cv, n_jobs=-1)
    results[name] = np.mean(scores)
    print(f"{name} Mean Recall: {results[name]:.4f}")



