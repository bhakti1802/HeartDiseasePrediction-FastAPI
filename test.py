import pandas as pd
import pickle

model = pickle.load(open("heart_model.pkl", "rb"))

input_data = pd.DataFrame([{
    "age": 25,
    "sex": 1,
    "cp": 0,
    "trestbps": 110,
    "chol": 180,
    "fbs": 0,
    "restecg": 1,
    "thalach": 190,
    "exang": 0,
    "oldpeak": 0.0,
    "slope": 2,
    "ca": 0,
    "thal": 2
}])

proba = model.predict_proba(input_data)[0]

print("Probability [No Disease, Disease]:", proba)