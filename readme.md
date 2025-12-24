# Telecom Customer Churn Prediction System

A production-grade machine learning system for predicting customer churn probability using XGBoost. This project demonstrates end-to-end ML engineering from model training to deployment with a professional analytics dashboard.

## Project Overview

**Problem**: Telecom companies lose customers to churn, resulting in revenue loss. Early identification of at-risk customers enables proactive retention strategies.

**Solution**: An XGBoost-based binary classification model that predicts churn probability from customer usage and billing data, deployed as a FastAPI backend with a modern React-like frontend dashboard.

**Target Audience**: ML engineers, data scientists, hiring managers evaluating ML internship portfolios.

## Architecture

```
┌─────────────────┐
│   Frontend      │  Static HTML/CSS/JS
│   (Dashboard)  │  ──────────────────
└────────┬────────┘
         │ HTTP/REST
         │
┌────────▼────────┐
│   FastAPI       │  Python Backend
│   Backend       │  ───────────────
│                 │  • Model Serving
│  /predict       │  • Model Metadata
│  /model/info    │  • Feature Importance
│  /model/metrics │  • Performance Metrics
└────────┬────────┘
         │
┌────────▼────────┐
│  XGBoost Model │  Pre-trained Model
│  (joblib)       │  ─────────────────
│                 │  • Pipeline
│                 │  • Threshold
│                 │  • Metadata
└─────────────────┘
```

##  Model Details

### Algorithm
- **XGBoost Classifier** (Gradient Boosting)
- Binary classification (Churn: Yes/No)
- Probability output (0-1 scale)

### Features (10 total)
1. `AccountWeeks` - Account age in weeks
2. `ContractRenewal` - Contract renewal status (0/1)
3. `DataPlan` - Has data plan (0/1)
4. `DataUsage` - Monthly data usage (GB)
5. `CustServCalls` - Customer service call count
6. `DayMins` - Daytime call minutes
7. `DayCalls` - Daytime call count
8. `MonthlyCharge` - Monthly subscription fee
9. `OverageFee` - Overage charges
10. `RoamMins` - Roaming minutes

### Model Performance
- Metrics available via `/model/metrics` endpoint
- Includes: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Confusion matrix visualization
- Train/test split performance

### Interpretability
- **Real Feature Importance**: XGBoost-derived feature importance scores
- **Key Model Signals**: Top contributing features with explanations
- **Business Insights**: Actionable recommendations based on risk level

## Quick Start

### Prerequisites
- Python 3.8+
- Model file: `telecom_churn_xgb_strong.joblib`

### Installation

1. **Clone/Download the repository**
   ```bash
   cd churn-predictor
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place model file**
   - Ensure `telecom_churn_xgb_strong.joblib` is in the project root
   - Model bundle should contain:
     - `pipeline`: Preprocessing + XGBoost model
     - `threshold`: Decision threshold (default: 0.4)
     - `metadata`: Optional model metadata (version, training date, etc.)
     - `metrics`: Optional performance metrics

4. **Start the server**
   ```bash
   uvicorn app:app --reload --port 8000
   ```

5. **Open in browser**
   ```
   http://localhost:8000
   ```

## API Endpoints

### Prediction
- **POST** `/predict`
  - Input: Customer feature values (JSON)
  - Output: Churn probability, prediction, risk level, feature importances

### Model Information
- **GET** `/model/info` - Model metadata (version, training date, hyperparameters)
- **GET** `/model/metrics` - Performance metrics (accuracy, precision, recall, etc.)
- **GET** `/model/features/importance` - Feature importance scores
- **GET** `/model/features/ranges` - Feature validation ranges

### Health Check
- **GET** `/health` - System health status

### Example Request
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "AccountWeeks": 50,
    "ContractRenewal": 1,
    "DataPlan": 1,
    "DataUsage": 3.5,
    "CustServCalls": 1,
    "DayMins": 200,
    "DayCalls": 100,
    "MonthlyCharge": 70,
    "OverageFee": 10,
    "RoamMins": 5
  }'
```

### Example Response
```json
{
  "churn_probability": 0.2345,
  "churn_prediction": 0,
  "risk_level": "low",
  "feature_importances": {
    "AccountWeeks": 15.2,
    "CustServCalls": 12.8,
    ...
  }
}
```

## Features

### Frontend Dashboard
- **Professional UI**: Clean, minimal, production-ready design
- **Model Overview**: Technical specifications and methodology
- **Performance Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC
- **Confusion Matrix**: Visual model evaluation
- **Real Feature Importance**: Model-derived importance scores
- **Key Model Signals**: Top contributing features with explanations
- **Business Insights**: Actionable recommendations by risk level
- **Input Validation**: Real-time validation with context
- **Responsive Design**: Mobile-friendly layout

### Backend Features
- **Error Handling**: Graceful failure handling with logging
- **Input Validation**: Pydantic models with range validation
- **Model Metadata**: Version tracking and provenance
- **Feature Importance**: XGBoost-derived importance extraction
- **Structured Logging**: Request and prediction logging
- **Health Checks**: System status monitoring

##  Model Evaluation

The system displays comprehensive model evaluation metrics:

- **Accuracy**: Overall prediction correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under ROC curve (discrimination ability)
- **Confusion Matrix**: Visual breakdown of predictions

##  Model Training 

While this repository focuses on deployment, typical training process:

1. **Data Collection**: Historical customer data with churn labels
2. **Feature Engineering**: Create/transform features
3. **Preprocessing**: Handle missing values, scaling, encoding
4. **Model Training**: XGBoost with hyperparameter tuning
5. **Evaluation**: Cross-validation, metrics calculation
6. **Threshold Selection**: Optimize threshold for business metrics
7. **Model Serialization**: Save pipeline + metadata to joblib

## Technology Stack

- **Backend**: FastAPI (Python)
- **ML Framework**: XGBoost, scikit-learn
- **Frontend**: Vanilla JavaScript, Chart.js
- **Styling**: CSS3 (Custom design system)
- **Deployment**: Uvicorn ASGI server

##  Project Structure

```
churn-predictor/
├── app.py                          # FastAPI backend
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── telecom_churn_xgb_strong.joblib # Pre-trained model
├── static/
│   └── index.html                  # Frontend dashboard
└── ML_INTERVIEW_REVIEW.md         # ML interviewer review
```

## Production Considerations

### Current Implementation
- Model serving with FastAPI
- Error handling and logging
- Input validation
- Model metadata tracking
- Feature importance extraction
- Performance metrics display

### Future Enhancements
- Model versioning system
- A/B testing framework
- Batch prediction endpoint
- Model monitoring and drift detection
- Automated retraining pipeline
- Database integration for predictions
- Authentication and rate limiting
- Docker containerization
- CI/CD pipeline

##  Business Impact

This system enables:
- **Proactive Retention**: Identify at-risk customers early
- **Resource Optimization**: Focus retention efforts on high-risk segments
- **Revenue Protection**: Reduce churn-related revenue loss
- **Data-Driven Decisions**: Evidence-based retention strategies

## Model Interpretability

The system provides multiple interpretability features:

1. **Feature Importance**: XGBoost-derived importance scores
2. **Key Signals**: Top contributing features for each prediction
3. **Business Context**: Explanations of feature impact
4. **Risk Segmentation**: Low/Medium/High risk categorization

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Chart.js Documentation: https://www.chartjs.org/

## License

This project is for portfolio/educational purposes.

##  Author

ML Internship Portfolio Project - Demonstrating production ML engineering skills.

---

**Note**: This is a portfolio project designed to showcase ML engineering capabilities. For production use, additional considerations (security, scalability, monitoring) should be implemented.
