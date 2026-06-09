from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

import numpy as np
import pandas as pd
import pickle

from database import get_connection

# Initialize app
app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")

# Load trained model
model = pickle.load(open("heart_model.pkl", "rb"))


# ---------------- HOME PAGE ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


# ---------------- PREDICT (WEB FORM) ----------------
@app.post("/predict", response_class=HTMLResponse)
async def predict(
    request: Request,
    age: int = Form(...),
    sex: int = Form(...),
    cp: int = Form(...),
    trestbps: int = Form(...),
    chol: int = Form(...),
    fbs: int = Form(...),
    restecg: int = Form(...),
    thalach: int = Form(...),
    exang: int = Form(...),
    oldpeak: float = Form(...),
    slope: int = Form(...),
    ca: int = Form(...),
    thal: int = Form(...)
):

    # ---------------- SAFE INPUT HANDLING ----------------
    input_data = pd.DataFrame([{
    "age": age,
    "sex": sex,
    "cp": cp,
    "trestbps": trestbps,
    "chol": chol,
    "fbs": fbs,
    "restecg": restecg,
    "thalach": thalach,
    "exang": exang,
    "oldpeak": oldpeak,
    "slope": slope,
    "ca": ca,
    "thal": thal
     }])

    prediction = model.predict(input_data)[0]

    # Convert to model input
    features = np.array([[
         age, sex, cp, trestbps, chol,
         fbs, restecg, thalach, exang,
         oldpeak, slope, ca, thal
    ]])

    # Get probability instead of direct prediction
    proba = model.predict_proba(features)[0]

    # probability of class 1 (Heart Disease)
    risk_score = proba[1]

    # threshold decision
    threshold = 0.65  # you can tune this (0.5 - 0.7 is common)

    if risk_score >= threshold:
       result = "Heart Disease"
    else:
       result = "No Heart Disease"

    # ---------------- DATABASE SAVE ----------------
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO prediction
            (age, sex, cp, trestbps, chol, prediction)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (age, sex, cp, trestbps, chol, result)
        )

        conn.commit()
        cursor.close()
        conn.close()

    except Exception as e:
        print("Database Error:", e)

    # ---------------- RESULT PAGE ----------------
    return templates.TemplateResponse(
    "result.html",
    {
        "request": request,
        "prediction": result,
        "risk": round(risk_score * 100, 2)
    }
)


# ---------------- API ENDPOINT ----------------
@app.post("/api/predict")
async def predict_api(data: dict):

    input_data = pd.DataFrame([data])
    features = input_data.values

    prediction = model.predict(features)[0]

    result = "Heart Disease" if prediction == 1 else "No Heart Disease"

    try:
        conn = get_connection()
        cur = conn.cursor()

        cur.execute(
            """
            INSERT INTO prediction
            (age, sex, cp, trestbps, chol, prediction)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (
                data["age"],
                data["sex"],
                data["cp"],
                data["trestbps"],
                data["chol"],
                result
            )
        )

        conn.commit()
        cur.close()
        conn.close()

    except Exception as e:
        print("DB Error:", e)

    return {"prediction": result}