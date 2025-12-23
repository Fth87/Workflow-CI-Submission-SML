import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os
import sys

# Get parameters from command line if provided
n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5

# Load data hasil preprocessing dari CSV
# Data sudah dinormalisasi menggunakan StandardScaler
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, 'cancer_preprocessing.csv')
df = pd.read_csv(data_path)

# Split features dan target
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Autolog akan otomatis mencatat semua parameter, metrik, dan model
mlflow.sklearn.autolog() 

model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
model.fit(X_train, y_train)

# Evaluasi model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.4f}")
print("Training Selesai. Cek Dashboard MLflow.")