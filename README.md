# 🛡️ Financial Fraud Detection Model

This project implements a machine learning pipeline to identify fraudulent transactions with high precision.

## 📊 Performance Summary
After hyperparameter tuning, the Random Forest model achieved:
- **Precision:** 0.90 (Only 10% false alarm rate)
- **Recall:** 0.86 (Caught 86% of all fraud cases)
- **F1-Score:** 0.88

## 🛠️ Technical Implementation
1. **Feature Scaling:** Used `StandardScaler` to normalize numerical data.
2. **Model Tuning:** Optimized via `RandomizedSearchCV` (30 total fits).
3. **Best Parameters:** - `n_estimators`: 200
   - `max_depth`: 20
   - `bootstrap`: False

## 📈 Key Insights
The model's top predictors were visualized using Feature Importance. 
*(Optional: Insert your image here using: ![Feature Importance](./feature_importance.png))*

## 🚀 How to Run
1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt`
3. Open the notebook in `notebooks/` to view the full pipeline.
