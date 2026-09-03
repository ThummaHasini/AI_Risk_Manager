import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model
model = joblib.load("fraud_model.pkl")

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ AI Risk Manager")
st.markdown("AI-powered Financial Fraud Risk Detection System")

# -------------------------------
# Risk Functions
# -------------------------------

def risk_level(score):
    if score < 20:
        return "LOW"
    elif score < 60:
        return "MEDIUM"
    else:
        return "HIGH"

def recommendation(score):
    if score < 20:
        return "Approve Transaction"
    elif score < 60:
        return "Manual Review"
    else:
        return "Block Transaction"

# -------------------------------
# Demo Transaction
# -------------------------------

st.subheader("Quick Demo")

if st.button("🚀 Run Demo"):

    demo_data = pd.DataFrame(
        np.random.randn(1, 30),
        columns=[
            "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
            "V10","V11","V12","V13","V14","V15","V16","V17","V18",
            "V19","V20","V21","V22","V23","V24","V25","V26",
            "V27","V28","Amount"
        ]
    )

    prob = model.predict_proba(demo_data)[0][1]
    pred = model.predict(demo_data)[0]

    score = int(prob * 100)

    st.success("Analysis Complete")

    st.metric("Fraud Probability", f"{prob*100:.2f}%")
    st.metric("Risk Score", score)

    st.write("### Prediction")
    st.write("Fraud" if pred == 1 else "Not Fraud")

    st.write("### Risk Level")
    st.write(risk_level(score))

    st.write("### Recommendation")
    st.write(recommendation(score))

st.divider()

# -------------------------------
# CSV Upload
# -------------------------------

st.subheader("📁 Upload Transaction CSV")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        scores = (probabilities * 100).astype(int)

        results = pd.DataFrame({
            "Prediction": predictions,
            "Fraud Probability (%)": np.round(probabilities * 100, 2),
            "Risk Score": scores
        })

        results["Risk Level"] = results["Risk Score"].apply(risk_level)
        results["Recommendation"] = results["Risk Score"].apply(recommendation)

        st.subheader("📊 Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Total", len(results))
        col2.metric("High Risk", (results["Risk Level"]=="HIGH").sum())
        col3.metric("Medium Risk", (results["Risk Level"]=="MEDIUM").sum())
        col4.metric("Low Risk", (results["Risk Level"]=="LOW").sum())

        st.subheader("📋 Results")

        st.dataframe(results, use_container_width=True)

        st.subheader("📈 Risk Distribution")

        st.bar_chart(results["Risk Level"].value_counts())

        csv = results.to_csv(index=False)

        st.download_button(
            "⬇️ Download Report",
            csv,
            "risk_analysis_report.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"Error processing file: {e}")
