import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------------------
# Page settings
# ---------------------------
st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📱 Mobile User Churn Prediction")
st.write("Predict whether a user is likely to churn based on behavior features.")

# ---------------------------
# Load model files
# ---------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, feature_cols = load_artifacts()

# ---------------------------
# Sidebar settings
# ---------------------------
st.sidebar.header("⚙️ Settings")
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.5, 0.01)

st.sidebar.info(
    "Threshold controls the churn prediction.\n"
    "Higher threshold → fewer churn predictions.\n"
    "Lower threshold → more churn predictions."
)

# ---------------------------
# Input mode selection
# ---------------------------
mode = st.radio(
    "Choose input method:",
    ["Manual Input (Single User)", "Upload CSV (Batch Prediction)"]
)

# ---------------------------
# Helper functions
# ---------------------------
def predict_single_user(input_df: pd.DataFrame):
    """
    input_df: dataframe with 1 row and columns = feature_cols
    """
    # Ensure correct order
    input_df = input_df[feature_cols]

    # Scale
    X_scaled = scaler.transform(input_df)

    # Probability
    prob = model.predict_proba(X_scaled)[:, 1][0]

    # Prediction using threshold
    pred = int(prob >= threshold)

    return prob, pred


def risk_level(prob):
    if prob >= 0.80:
        return "🔥 Very High Risk"
    elif prob >= 0.60:
        return "⚠️ High Risk"
    elif prob >= 0.40:
        return "🟡 Medium Risk"
    else:
        return "✅ Low Risk"


# ---------------------------
# Manual input mode
# ---------------------------
if mode == "Manual Input (Single User)":

    st.subheader("🧾 Enter user feature values")

    # Make form for better UI
    with st.form("manual_form"):
        cols = st.columns(3)

        user_input = {}

        # Create numeric inputs for all features
        for i, col in enumerate(feature_cols):
            with cols[i % 3]:
                user_input[col] = st.number_input(
                    label=col,
                    value=0.0,
                    step=1.0
                )

        submitted = st.form_submit_button("✅ Predict Churn")

    if submitted:
        input_df = pd.DataFrame([user_input])

        prob, pred = predict_single_user(input_df)

        st.markdown("---")
        st.subheader("📌 Prediction Result")

        st.metric("Churn Probability", f"{prob:.3f}")

        st.write("**Risk Level:**", risk_level(prob))

        if pred == 1:
            st.error("🚨 Prediction: User is likely to CHURN")
        else:
            st.success("✅ Prediction: User is NOT likely to churn")

        st.write("### Input Summary")
        st.dataframe(input_df)

# ---------------------------
# CSV Upload mode
# ---------------------------
else:
    st.subheader("📂 Upload CSV for Batch Prediction")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.write("✅ Uploaded Data Preview")
        st.dataframe(df.head())

        # Check missing columns
        missing_cols = [c for c in feature_cols if c not in df.columns]

        if missing_cols:
            st.error("❌ Your CSV is missing these required columns:")
            st.write(missing_cols)
        else:
            # Ensure correct order
            X = df[feature_cols].copy()

            # Scale
            X_scaled = scaler.transform(X)

            # Probabilities
            probs = model.predict_proba(X_scaled)[:, 1]
            preds = (probs >= threshold).astype(int)

            output = df.copy()
            output["churn_probability"] = probs
            output["churn_prediction"] = preds

            st.markdown("---")
            st.subheader("📌 Batch Prediction Results")
            st.dataframe(output.head(20))

            # Summary
            st.write("### 📊 Prediction Summary")
            st.write(output["churn_prediction"].value_counts())

            # Download button
            csv_data = output.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_data,
                file_name="churn_predictions.csv",
                mime="text/csv"
            )
