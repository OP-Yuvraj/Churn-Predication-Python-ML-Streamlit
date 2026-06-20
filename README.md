# 📱 Mobile User Churn Prediction (Streamlit)

This project predicts whether a mobile app user is likely to churn based on their behavior and engagement features.

## 🔥 Project Highlights
- Large scale dataset (1M+ sample from 19M rows)
- Feature engineering using user-level aggregation
- Time-based churn labeling (inactive for last 30 days)
- Machine Learning model training & evaluation
- Streamlit app for interactive predictions

## ✅ Features Used
- Recency (days since last activity)
- Total interactions / frequency
- Active days
- User lifetime days
- Average rating given by user
- Rating variance (std)
- Category diversity
- Unique apps used

## 🧠 Model
- Logistic Regression (baseline)
- StandardScaler used for feature scaling

## 📊 Evaluation
Metrics used:
- Precision
- Recall
- F1-score
- ROC-AUC

## 🚀 Run Streamlit App
Install requirements:

```bash
pip install -r requirements.txt
