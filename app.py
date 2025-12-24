# app.py
"""
Telecom Churn Prediction API
Production-grade ML system for predicting customer churn using XGBoost.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn
import logging
import os
from typing import Dict, List, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Telecom Churn Prediction API",
    version="1.0.0",
    description="Production ML system for customer churn prediction using XGBoost"
)

# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_home():
    return FileResponse("static/index.html")

# CORS configuration
origins = [
    "https://project-etxf-457vimp81-chaitanyaphuge17s-projects.vercel.app",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# FEATURE COLUMNS DEFINITION
# ============================================
FEATURE_COLUMNS = [
    "AccountWeeks", "ContractRenewal", "DataPlan", "DataUsage",
    "CustServCalls", "DayMins", "DayCalls", "MonthlyCharge",
    "OverageFee", "RoamMins"
]

# ============================================
# MODEL LOADING WITH ERROR HANDLING
# ============================================
MODEL_PATH = "telecom_churn_xgb_strong.joblib"
model_bundle = None
pipeline = None
threshold = 0.4
model_metadata = {}
feature_importances = {}

try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    
    logger.info(f"Loading model from {MODEL_PATH}")
    model_bundle = joblib.load(MODEL_PATH)
    pipeline = model_bundle.get("pipeline")
    threshold = model_bundle.get("threshold", 0.4)
    
    # Extract model metadata if available
    model_metadata = model_bundle.get("metadata", {})
    if not model_metadata:
        # Default metadata if not in bundle
        model_metadata = {
            "version": "1.0.0",
            "training_date": "2024-01-01",
            "algorithm": "XGBoost",
            "hyperparameters": {},
            "dataset_size": "Unknown"
        }
    
    # Extract feature importances from XGBoost model
    if pipeline is not None:
        try:
            # Get the XGBoost model from pipeline (assuming it's the last step)
            xgb_model = None
            if hasattr(pipeline, 'named_steps'):
                for name, step in pipeline.named_steps.items():
                    if hasattr(step, 'feature_importances_'):
                        xgb_model = step
                        break
            elif hasattr(pipeline, 'feature_importances_'):
                xgb_model = pipeline
            
            if xgb_model is not None:
                importances = xgb_model.feature_importances_
                feature_names = FEATURE_COLUMNS
                feature_importances = dict(zip(feature_names, importances.tolist()))
                # Normalize to 0-100 scale
                total = sum(feature_importances.values())
                if total > 0:
                    feature_importances = {k: (v/total)*100 for k, v in feature_importances.items()}
                logger.info("Feature importances extracted successfully")
        except Exception as e:
            logger.warning(f"Could not extract feature importances: {e}")
            # Fallback: equal importance
            feature_importances = {name: 100/len(FEATURE_COLUMNS) for name in FEATURE_COLUMNS}
    
    logger.info("Model loaded successfully")
    
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise RuntimeError(f"Model initialization failed: {e}")

# Input validation ranges (for frontend context)
FEATURE_RANGES = {
    "AccountWeeks": {"min": 0, "max": 200, "typical": "20-100"},
    "ContractRenewal": {"min": 0, "max": 1, "typical": "0 or 1"},
    "DataPlan": {"min": 0, "max": 1, "typical": "0 or 1"},
    "DataUsage": {"min": 0, "max": 50, "typical": "1-10 GB"},
    "CustServCalls": {"min": 0, "max": 20, "typical": "0-5"},
    "DayMins": {"min": 0, "max": 500, "typical": "100-300"},
    "DayCalls": {"min": 0, "max": 300, "typical": "50-150"},
    "MonthlyCharge": {"min": 0, "max": 200, "typical": "50-100"},
    "OverageFee": {"min": 0, "max": 100, "typical": "0-30"},
    "RoamMins": {"min": 0, "max": 50, "typical": "0-10"},
}

# ============================================
# PYDANTIC MODELS
# ============================================
class CustomerInput(BaseModel):
    AccountWeeks: float = Field(..., ge=0, description="Account age in weeks")
    ContractRenewal: int = Field(..., ge=0, le=1, description="Contract renewal status (0 or 1)")
    DataPlan: int = Field(..., ge=0, le=1, description="Has data plan (0 or 1)")
    DataUsage: float = Field(..., ge=0, description="Data usage in GB")
    CustServCalls: int = Field(..., ge=0, description="Number of customer service calls")
    DayMins: float = Field(..., ge=0, description="Daytime minutes")
    DayCalls: int = Field(..., ge=0, description="Number of daytime calls")
    MonthlyCharge: float = Field(..., ge=0, description="Monthly charge amount")
    OverageFee: float = Field(..., ge=0, description="Overage fee amount")
    RoamMins: float = Field(..., ge=0, description="Roaming minutes")

class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: str
    feature_importances: Optional[Dict[str, float]] = None

class ModelInfoResponse(BaseModel):
    version: str
    training_date: str
    algorithm: str
    hyperparameters: Dict
    dataset_size: str
    threshold: float
    feature_count: int
    feature_names: List[str]

class ModelMetricsResponse(BaseModel):
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    roc_auc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    train_accuracy: Optional[float] = None
    test_accuracy: Optional[float] = None

class FeatureImportanceResponse(BaseModel):
    importances: Dict[str, float]
    feature_names: List[str]

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get model metadata and information"""
    try:
        return ModelInfoResponse(
            version=model_metadata.get("version", "1.0.0"),
            training_date=model_metadata.get("training_date", "Unknown"),
            algorithm=model_metadata.get("algorithm", "XGBoost"),
            hyperparameters=model_metadata.get("hyperparameters", {}),
            dataset_size=model_metadata.get("dataset_size", "Unknown"),
            threshold=threshold,
            feature_count=len(FEATURE_COLUMNS),
            feature_names=FEATURE_COLUMNS
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/metrics", response_model=ModelMetricsResponse)
async def get_model_metrics():
    """Get model performance metrics"""
    try:
        # Try to get metrics from model bundle
        metrics = model_bundle.get("metrics", {})
        
        # If metrics not in bundle, return default structure (would be populated in production)
        return ModelMetricsResponse(
            accuracy=metrics.get("accuracy"),
            precision=metrics.get("precision"),
            recall=metrics.get("recall"),
            f1_score=metrics.get("f1_score"),
            roc_auc=metrics.get("roc_auc"),
            confusion_matrix=metrics.get("confusion_matrix"),
            train_accuracy=metrics.get("train_accuracy"),
            test_accuracy=metrics.get("test_accuracy")
        )
    except Exception as e:
        logger.error(f"Error getting model metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/features/importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """Get feature importance scores from the model"""
    try:
        if not feature_importances:
            raise HTTPException(status_code=503, detail="Feature importances not available")
        return FeatureImportanceResponse(
            importances=feature_importances,
            feature_names=FEATURE_COLUMNS
        )
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/features/ranges")
async def get_feature_ranges():
    """Get feature validation ranges for frontend"""
    return FEATURE_RANGES

@app.post("/predict", response_model=PredictionResponse)
async def predict_single(customer: CustomerInput):
    """Make a churn prediction for a single customer"""
    try:
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare input data
        data = [[
            customer.AccountWeeks, customer.ContractRenewal, customer.DataPlan,
            customer.DataUsage, customer.CustServCalls, customer.DayMins,
            customer.DayCalls, customer.MonthlyCharge, customer.OverageFee,
            customer.RoamMins
        ]]
        df = pd.DataFrame(data, columns=FEATURE_COLUMNS)
        
        # Make prediction
        proba = pipeline.predict_proba(df)[:, 1][0]
        pred = int(proba >= threshold)
        risk_level = "low" if proba < 0.3 else "medium" if proba < 0.6 else "high"
        
        # Log prediction (in production, this would go to a monitoring system)
        logger.info(f"Prediction made: prob={proba:.4f}, pred={pred}, risk={risk_level}")
        
        return PredictionResponse(
            churn_probability=round(proba, 4),
            churn_prediction=pred,
            risk_level=risk_level,
            feature_importances=feature_importances if feature_importances else None
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
