# app.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib, pandas as pd, uvicorn
from io import BytesIO

app = FastAPI(title="Telecom Churn Prediction API", version="1.0.0")

# Allow your HTML origin (adjust port/host if needed)
origins = [
    "https://project-etxf.vercel.app",  # your frontend
    "http://127.0.0.1:5500",            # local testing (optional)
    "http://localhost:5500",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- existing code below ----
model_bundle = joblib.load("telecom_churn_xgb_strong.joblib")
pipeline = model_bundle["pipeline"]
threshold = model_bundle["threshold"]

FEATURE_COLUMNS = [
    "AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage",
    "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge",
    "OverageFee", "RoamMins"
]

class CustomerInput(BaseModel):
    AccountWeeks: float = Field(..., ge=0)
    ContractRenewal: int = Field(..., ge=0, le=1)
    DataPlan: int = Field(..., ge=0, le=1)
    DataUsage: float = Field(..., ge=0)
    CustServCalls: int = Field(..., ge=0)
    DayMins: float = Field(..., ge=0)
    DayCalls: int = Field(..., ge=0)
    MonthlyCharge: float = Field(..., ge=0)
    OverageFee: float = Field(..., ge=0)
    RoamMins: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerInput):
    data = [[
        customer.AccountWeeks, customer.ContractRenewal, customer.DataPlan,
        customer.DataUsage, customer.CustServCalls, customer.DayMins,
        customer.DayCalls, customer.MonthlyCharge, customer.OverageFee,
        customer.RoamMins
    ]]
    df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
    proba = pipeline.predict_proba(df)[:, 1][0]
    pred = int(proba >= threshold)
    risk_level = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
    return PredictionResponse(
        churn_probability=round(proba, 4),
        churn_prediction=pred,
        risk_level=risk_level
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
