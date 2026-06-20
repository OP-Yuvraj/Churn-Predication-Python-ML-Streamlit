# 📱 Mobile User Churn Prediction (Streamlit)

## 📌 Project Overview

This project predicts whether a mobile app user is likely to churn based on user engagement, activity patterns, and behavioral features.

The goal is to identify inactive users early so businesses can improve retention strategies and reduce customer churn.

---

## 🔥 Project Highlights

* Large-scale dataset (1M+ sample from 19M rows)
* Feature engineering using user-level aggregation
* Time-based churn labeling (inactive for last 30 days)
* Machine Learning model training & evaluation
* Interactive Streamlit web application for predictions

---

## 📂 Dataset Information

* Original dataset size: **19 million+ rows**
* Sample used for training: **1 million+ rows**
* Data includes:

  * User interactions
  * Ratings
  * App categories
  * Activity timestamps

---

## ✅ Features Used

* Recency (days since last activity)
* Total interactions / frequency
* Active days
* User lifetime days
* Average rating given by user
* Rating variance (std)
* Category diversity
* Unique apps used

---

## 🧠 Machine Learning Model

* Logistic Regression (Baseline Model)
* StandardScaler for feature scaling

---

## 📊 Evaluation Metrics

Model performance evaluated using:

* Precision
* Recall
* F1-score
* ROC-AUC

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit

## 📁 Project Structure

```bash
## 📁 Project Structure


Churn-Prediction-Python-ML-Streamlit/
│
├── Notebook/
│   └── churn_analysis.ipynb
│
├── Script/
│   └── app.py
│
├── screenshots/
│   ├── home_page.png
│   └── prediction_page.png
│
├── README.md
├── requirements.txt
└── .gitignore

```

## 🚀 Run Streamlit App

### Install Dependencies

pip install -r requirements.txt

### Run Application

streamlit run Script/app.py

---

## 📷 Screenshots

![alt text](home_page.png)
![alt text](predicaton_page.png)

## 🎯 Future Improvements

* Try advanced ML models (XGBoost, Random Forest)
* Hyperparameter tuning
* Improve deployment pipeline
* Add dashboard visualizations

---

## 👨‍💻 Author

**Yuvraj Singh**
BCA Student | Aspiring Data Analyst / Data Scientist | yuvrajsinghrawat88@gmail.com

