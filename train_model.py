import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("Loading dataset...")

df = pd.read_csv("heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Creating model...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

print("Training model...")

model.fit(X_train, y_train)

print("MODEL TRAINED SUCCESSFULLY")

print("Calculating accuracy...")

train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

print("Saving model...")

with open("heart_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("DONE")