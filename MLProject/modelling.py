import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# 1. Load Data
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Set Experiment
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Basic_Submission_Iris")

# 3. Training & Autolog
with mlflow.start_run():
    mlflow.sklearn.autolog() 

    model = RandomForestClassifier(n_estimators=50, max_depth=5)
    model.fit(X_train, y_train)

    mlflow.set_tag("developer", "Fatih_Fawwaz")

    # Register Model 
    mlflow.sklearn.log_model(model, "model", registered_model_name="IrisModelBasic")

print("Training Selesai. Cek Dashboard MLflow.")