
import streamlit as st
import pandas as pd
import joblib

# Load model and feature names
model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# Page Title
st.title("🛡️ AI Risk Manager System")
st.write("Enter transaction details and analyze fraud risk.")

# Input Fields
inputs = []

for feature in feature_names:
    value = st.number_input(feature, value=0.0, format="%.6f")
    inputs.append(value)

# Predict Button
if st.button("Analyze Risk"):

    data = pd.DataFrame([inputs], columns=feature_names)

    prediction = model.predict(data)[0]
    probability = model.predict_proba(data)[0][1]

    risk_score = int(probability * 100)

    if risk_score < 20:
        risk = "LOW"
        recommendation = "Approve"

    elif risk_score < 60:
        risk = "MEDIUM"
        recommendation = "Manual Review"

    else:
        risk = "HIGH"
        recommendation = "Block Transaction"

    st.success("Analysis Complete")

    st.subheader("Results")

    st.write(
        "**Prediction:**",
        "Fraud" if prediction == 1 else "Not Fraud"
    )

    st.write(
        "**Fraud Probability:**",
        f"{probability*100:.2f}%"
    )

    st.write(
        "**Risk Score:**",
        risk_score
    )

    st.write(
        "**Risk Level:**",
        risk
    )

    st.write(
        "**Recommendation:**",
        recommendation
    )

    st.subheader("Risk Summary")

    result_df = pd.DataFrame({
        "Prediction": ["Fraud" if prediction == 1 else "Not Fraud"],
        "Fraud Probability (%)": [round(probability*100, 2)],
        "Risk Score": [risk_score],
        "Risk Level": [risk],
        "Recommendation": [recommendation]
    })

    st.dataframe(result_df)
