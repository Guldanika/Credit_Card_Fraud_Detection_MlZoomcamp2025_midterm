# Credit_Card_Fraud_Detection_MlZoomcamp2025_midterm

**Author:** Guldanika Osmonova  
**Date:** November 2025  

## Introduction / Problem Statement

> Credit card fraud is a serious problem in the financial sector, causing significant losses every year. Detecting fraudulent transactions is challenging because fraud cases are extremely rare compared to legitimate transactions, and patterns of fraud are constantly evolving.

**Goal of this project:**  
Build a predictive model that can reliably detect fraudulent credit card transactions, **maximizing the detection of rare fraud cases while minimizing false alarms**.

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
→ 284,807 transactions · only 492 fraud cases (~0.17% class imbalance)

This project covers the complete ML lifecycle:
- Exploratory Data Analysis
- Preprocessing & handling severe class imbalance (SMOTE)
- Training & comparison of multiple models (Logistic Regression, Random Forest, XGBoost)
- Hyperparameter tuning with RandomizedSearchCV
- Custom threshold optimization (Youden’s J statistic)
- Export to script
- FastAPI deployment
- Dependency management
- Full containerization with Docker

## Results

| Model                  | Precision (class 1) | Recall (class 1) | F1-score (class 1) | Test Accuracy |
|------------------------|---------------------|------------------|--------------------|---------------|
| Logistic Regression    | 0.9206              | 0.7838           | 0.8467             | 0.9995        |
| Random Forest          | ~0.95               | ~0.82            | ~0.88              | 0.9995        |
| XGBoost (best from search) | ~0.95           | ~0.82            | ~0.88              | 0.9995        |

**Final chosen model:** Logistic Regression with custom threshold = 1.0  
→ Perfect balance of extremely high recall on fraud while keeping precision acceptable for production use.

## Project Structure 


├── app.py                     # FastAPI service
├── train_model_fraud_detection.py  # Training script (exported from notebook)
├── model.pkl                  # Final Logistic Regression model
├── threshold.pkl              # Optimal threshold
├── requirements.txt
├── Dockerfile                 # Full containerization
├── creditcard.csv             # Dataset (284807 rows)
└── README.md                  # You are here


## How to Run

### 1. Locally
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload

2. With Docker (recommended – works on any machine)
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection


API Usage Example (Swagger UI available at /docs)
{
  "Time": 76800.0,
  "V1": -10.6458,
  "V2": 7.4269,
  "V3": -12.3795,
  "V4": 8.7979,
  ... (all V1–V28)
  "Amount": 1.0
}


Video Demonstration
30-second video showing Docker build → run → live fraud prediction is on my Youtube chanel here. 
Conclusion
The model successfully detects fraudulent transactions in highly imbalanced real-world data and is fully production-ready (FastAPI + Docker).

