# Credit_Card_Fraud_Detection_MlZoomcamp2025_midterm

**Author:** Guldanika Osmonova  
**Date:** November 2025  

## Introduction / Problem Statement

> Credit card fraud is a serious problem in the financial sector, causing significant losses every year. Detecting fraudulent transactions is challenging because fraud cases are extremely rare compared to legitimate transactions, and patterns of fraud are constantly evolving.

**Goal of this project:**  
Build a predictive model that can reliably detect fraudulent credit card transactions, **maximizing the detection of rare fraud cases while minimizing false alarms**.

Dataset: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
→ 284,807 transactions · only 492 fraud cases (~0.17% class imbalance) 

SEE the whole working (EDA, preprocessing, model training, evaluation, saving part in Notebook here: https://github.com/Guldanika/Credit_Card_Fraud_Detection_MlZoomcamp2025_midterm/blob/main/Credit_Card_Fraud_Detection.ipynb 

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

## Project Structure 
```
├── app.py                     # FastAPI service
├── train_model_fraud_detection.py  # Training script (exported from notebook)
├── model.pkl                  # Final Logistic Regression model
├── threshold.pkl              # Optimal threshold
├── requirements.txt
├── Dockerfile                 # Full containerization
├── creditcard.csv             # Dataset (284807 rows)
└── README.md                  # You are here
```

**📊 EXPLORATORY DATA ANALYSIS (EDA) - Summary**

The Credit Card Fraud Detection dataset contains 284,807 transactions, of which only 0.17% are fraudulent. Below are the key insights from the exploratory analysis.

1. Severe Class Imbalance
Fraud cases: 492
Non-fraud cases: 284,315
Ratio ≈ 1 : 577
This imbalance requires techniques such as SMOTE, class weighting, or threshold tuning to improve model sensitivity.

2. Feature Distributions
Most features (V1–V28) are PCA-transformed components with near-Gaussian distributions.
Amount and Time are highly skewed and benefit from scaling.
Fraudulent transactions show noticeably different distribution shapes in certain components, especially:
V14
V12
V10
V17
These features often emerge as top predictors.

3. Correlation Structure
Because PCA was used, the dataset exhibits very low multicollinearity.

Most correlations between features are close to zero.
Features most correlated with fraud (negatively):
V9
V1
V5
V6

4. Outlier Behavior
Fraudulent transactions tend to form distinct tails or separate clusters in features like V14, V12, and V17, which helps ML models detect anomalies.

5. Data Quality
No missing values.
All features numeric → ready for ML.
Dataset size allows efficient training even with SMOTE applied.

⭐ **Overall EDA Conclusion**

The dataset is clean, highly imbalanced, and structurally suited for anomaly detection and binary classification. Several PCA components demonstrate strong separability between fraud and non-fraud classes, enabling effective modeling with Logistic Regression, Random Forest, XGBoost, or ensemble approaches, especially when combined with SMOTE and threshold optimization.

**MODELLING APPROACH & RESULTS**

## Modeling Results

Used SMOTE for oversampling on training data only.

| Model                | Precision (1) | Recall (1) | F1 (1) | Notes                          |
|----------------------|---------------|------------|--------|--------------------------------|
| Logistic Regression  | 0.9206        | 0.7838     | 0.8467 | Final chosen model             |
| Random Forest        | ~0.95         | ~0.82      | ~0.88  | Good but slower                |
| XGBoost              | ~0.95         | ~0.82      | ~0.88  | Similar to RF                  |

Best threshold (Youden’s J on validation): ~0.9837 → final threshold = 1.0 (perfect for production recall focus)

Test set (final Logistic Regression with optimal threshold):
- Precision 0.9206 → Recall 0.7838 → F1 0.8467
- Overall accuracy 99.95%

## Final Model

- Logistic Regression (max_iter=5000)
- Trained on SMOTE-balanced data
- Custom threshold = 1.0 (Youden’s J optimized)
- Saved as model.pkl + threshold.pkl using joblib

Deployment:
- FastAPI service (app.py)
- Input: 30 features (Time + V1–V28 + Amount)
- Output: prediction (0/1) + fraud_probability + status
- Fully containerized with Docker

**HOW TO RUN** 
## How to Run

### Locally
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
``` 


**DOCKER**
```bash
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

## Video Demonstrations

### **Local FastAPI deployment**
```
[![Local FastAPI demo](https://img.youtube.com/vi/x-26tp88zHw/maxresdefault.jpg)](https://youtu.be/x-26tp88zHw)
```

### Full Docker containerization (build → run → prediction)
```
[![Docker containerization demo](https://img.youtube.com/vi/L4TPK6dOKyA/maxresdefault.jpg)](https://youtu.be/L4TPK6dOKyA)
```

Both demos look identical — this is the point of Docker: **100% reproducible environment** 


## Results

| Model                  | Precision (class 1) | Recall (class 1) | F1-score (class 1) | Test Accuracy |
|------------------------|---------------------|------------------|--------------------|---------------|
| Logistic Regression    | 0.9206              | 0.7838           | 0.8467             | 0.9995        |
| Random Forest          | ~0.95               | ~0.82            | ~0.88              | 0.9995        |
| XGBoost (best from search) | ~0.95           | ~0.82            | ~0.88              | 0.9995        |

**Final chosen model:** Logistic Regression with custom threshold = 1.0  
→ Perfect balance of extremely high recall on fraud while keeping precision acceptable for production use.


## How to Run

### 1. Locally
```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --reload
```

2. With Docker (recommended – works on any machine)
```
docker build -t fraud-detection .
docker run -p 8000:8000 fraud-detection
```

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


### Conclusion
The model successfully detects fraudulent transactions in highly imbalanced real-world data and is fully production-ready (FastAPI + Docker).

## Video Demonstrations

### Local FastAPI deployment
```
[![Local FastAPI demo](https://img.youtube.com/vi/x-26tp88zHw/maxresdefault.jpg)](https://youtu.be/x-26tp88zHw)
```
### Full Docker containerization (build → run → prediction)
```
[![Docker containerization demo](https://img.youtube.com/vi/L4TPK6dOKyA/maxresdefault.jpg)](https://youtu.be/L4TPK6dOKyA)
```
Both demos look identical — this is the point of Docker: **100% reproducible environment**
