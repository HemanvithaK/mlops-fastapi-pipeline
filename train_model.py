# mlops/train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()
model = RandomForestClassifier()
model.fit(data.data, data.target)
joblib.dump(model, "model.pkl")
