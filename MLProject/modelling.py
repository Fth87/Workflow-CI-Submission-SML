import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Get parameters from command line if provided
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# 1. Load Data
# Load Breast Cancer Wisconsin Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup MLflow - ketika dijalankan dengan mlflow run, experiment sudah di-handle
# Hanya set experiment jika tidak ada MLFLOW_RUN_ID (artinya direct python execution)
if not os.environ.get("MLFLOW_RUN_ID"):
    mlflow.set_experiment("Basic_Submission_Cancer")
    mlflow.start_run()

# 3. Training & Autolog
mlflow.sklearn.autolog() 

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

mlflow.set_tag("developer", "Fatih_Fawwaz")
mlflow.log_param("n_estimators", n_estimators)
mlflow.log_param("max_depth", max_depth)

# Register Model 
mlflow.sklearn.log_model(model, "model", registered_model_name="CancerModelBasic")


if not os.environ.get("MLFLOW_RUN_ID"):
    mlflow.end_run()

print("Training Selesai. Cek Dashboard MLflow.")