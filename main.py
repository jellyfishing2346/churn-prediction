from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
import os

app = FastAPI(title="Customer Churn Prediction API", description="Predict churn and get SHAP explanations.")

# Load model and scaler (use the most recent or default to customer_data)
MODEL_PATH = "model_customer_data.joblib"
SCALER_PATH = "scaler_customer_data.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class CustomerData(BaseModel):
    MonthlySpend: float
    TotalPurchaseFrequency: float
    Tenure: float
    CustomerType: str

@app.post("/predict")
def predict(data: CustomerData):
    # Prepare input
    df = pd.DataFrame([data.dict()])
    # Encode CustomerType
    df['CustomerType'] = 1 if df['CustomerType'].iloc[0].lower() == 'vip' else 0
    X_scaled = scaler.transform(df)
    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0][1]
    # SHAP explanation
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
    feature_importance = dict(zip(df.columns, shap_values[0]))
    return {
        "prediction": int(pred),
        "probability": float(proba),
        "shap_feature_importance": feature_importance
    }

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API. Use /predict endpoint with customer data."}
