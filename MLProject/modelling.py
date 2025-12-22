import mlflow
import mlflow.sklearn
import pandas as pd
import urllib.request
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# 1. Load Data
# Download Wine Quality Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
urllib.request.urlretrieve(url, "winequality-red.csv")

df = pd.read_csv("winequality-red.csv", sep=";")
df['quality_class'] = pd.cut(df['quality'], bins=[0, 5, 6, 10], labels=[0, 1, 2])

X = df.drop(['quality', 'quality_class'], axis=1)
y = df['quality_class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

if not os.environ.get("GITHUB_ACTIONS"):
    # Kalau TIDAK di GitHub Actions (artinya di laptop), pakai localhost
    mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Basic_Submission_Wine")

# 3. Training & Autolog
with mlflow.start_run():
    mlflow.sklearn.autolog() 

    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)

    mlflow.set_tag("developer", "Fatih_Fawwaz")

    # Register Model 
    mlflow.sklearn.log_model(model, "model", registered_model_name="WineQualityModelBasic")

print("Training Selesai. Cek Dashboard MLflow.")