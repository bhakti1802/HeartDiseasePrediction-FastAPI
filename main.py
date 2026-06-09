from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel

import numpy as np
import pickle

from database import get_connection

app = FastAPI()

templates = Jinja2Templates(directory="templates")

model = pickle.load(open("heart_model.pkl", "rb"))


class HeartInput(BaseModel):

    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )


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

    features = np.array([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]])

    prediction = model.predict(features)[0]

    result = (
        "Heart Disease"
        if prediction == 1
        else "No Heart Disease"
    )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO prediction
        (age,sex,cp,trestbps,chol,prediction)
        VALUES(%s,%s,%s,%s,%s,%s)
        """,
        (
            age,
            sex,
            cp,
            trestbps,
            chol,
            result
        )
    )

    conn.commit()

    cursor.close()
    conn.close()

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prediction": result
        }
    )


@app.post("/api/predict")
async def predict_api(data: HeartInput):

    features = np.array([[
        data.age,
        data.sex,
        data.cp,
        data.trestbps,
        data.chol,
        data.fbs,
        data.restecg,
        data.thalach,
        data.exang,
        data.oldpeak,
        data.slope,
        data.ca,
        data.thal
    ]])

    prediction = model.predict(features)[0]

    result = (
        "Heart Disease"
        if prediction == 1
        else "No Heart Disease"
    )

    conn = get_connection()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO prediction
        (age,sex,cp,trestbps,chol,prediction)
        VALUES(%s,%s,%s,%s,%s,%s)
        """,
        (
            data.age,
            data.sex,
            data.cp,
            data.trestbps,
            data.chol,
            result
        )
    )

    conn.commit()

    cur.close()
    conn.close()

    return {
        "prediction": result
    }