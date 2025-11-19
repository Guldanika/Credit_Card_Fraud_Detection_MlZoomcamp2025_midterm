from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

# Загружаем модель и порог (обязательно в try/except, чтобы видеть ошибки)
try:
    model = joblib.load("model.pkl")
    threshold = joblib.load("threshold.pkl")
    print(f"Модель и threshold ({threshold:.6f}) успешно загружены!")
except Exception as e:
    raise RuntimeError(f"Не удалось загрузить модель или threshold: {e}")

app = FastAPI(title="Credit Card Fraud Detection API - ML Zoomcamp 2025")

class Transaction(BaseModel):
    Time: float = 0.0  # ← ЭТО ОБЯЗАТЕЛЬНО! Даже если 0 — модель ждёт эту колонку
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

@app.get("/")
def home():
    return {"message": "API работает! Иди в /docs и тестируй /predict"}

@app.post("/predict")
def predict(transaction: Transaction):
    try:
        # Критично: тот же порядок и все 30 колонок, как при обучении
        data = transaction.dict()
        df = pd.DataFrame([data])[
            ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9',
             'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19',
             'V20', 'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
        ]

        proba = model.predict_proba(df)[0][1]
        prediction = int(proba >= threshold)

        return {
            "prediction": prediction,                    # 1 = мошенничество
            "fraud_probability": round(float(proba), 6),
            "threshold_used": float(threshold),
            "status": "fraud" if prediction else "ok"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")