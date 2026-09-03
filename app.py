import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

# ==========================
# LOAD MODEL
# ==========================
model = joblib.load("fraud_model.pkl")

# ==========================
# TITLE
# ==========================
st.title("🛡️ AI Risk Manager")
st.markdown("### AI-Powered Financial Fraud Risk Detection System")

# ==========================
# FUNCTIONS
# ==========================

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


required_cols = [
    "Time","V1","V2","V3","V4","V5","V6","V7","V8","V9",
    "V10","V11","V12","V13","V14","V15","V16","V17","V18",
    "V19","V20","V21","V22","V23","V24","V25","V26",
    "V27","V28","Amount"
]

# ==========================
# DEMO SECTION
# ==========================

st.subheader("🚀 Quick Demo")

st.write(
    "Run a sample transaction through the fraud detection model."
)

if st.button("Run Demo Analysis"):

    demo_data = pd.DataFrame(
        np.random.randn(1, 30),
        columns=required_cols
    )

    prediction = model.predict(demo_data)[0]
    probability = model.predict_proba(demo_data)[0][1]

    risk_score = int(probability * 100)

    risk = risk_level(risk_score)
    action = recommendation(risk_score)

    st.success("Analysis Completed")

    col1, col2 = st.columns(2)

    col1.metric(
        "Fraud Probability",
        f"{probability*100:.2f}%"
    )

    col2.metric(
        "Risk Score",
        risk_score
    )

    result_df = pd.DataFrame({
        "Prediction": [
            "Fraud" if prediction == 1 else "Not Fraud"
        ],
        "Risk Level": [risk],
        "Recommendation": [action]
    })

    st.dataframe(result_df, use_container_width=True)

st.divider()

# ==========================
# CSV UPLOAD
# ==========================

st.subheader("📁 Upload Transaction CSV")

uploaded_file = st.file_uploader(
    "Choose a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        missing = [
            col for col in required_cols
            if col not in data.columns
        ]

        if missing:
            st.error(
                f"Missing Required Columns: {missing}"
            )
            st.stop()

        data = data[required_cols]

        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        scores = (probabilities * 100).astype(int)

        results = pd.DataFrame({
            "Prediction": np.where(
                predictions == 1,
                "Fraud",
                "Not Fraud"
            ),
            "Fraud Probability (%)": np.round(
                probabilities * 100,
                2
            ),
            "Risk Score": scores
        })

        results["Risk Level"] = results[
            "Risk Score"
        ].apply(risk_level)

        results["Recommendation"] = results[
            "Risk Score"
        ].apply(recommendation)

        st.success("File Processed Successfully")

        st.subheader("📊 Summary")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Transactions",
            len(results)
        )

        col2.metric(
            "High Risk",
            (results["Risk Level"] == "HIGH").sum()
        )

        col3.metric(
            "Medium Risk",
            (results["Risk Level"] == "MEDIUM").sum()
        )

        col4.metric(
            "Low Risk",
            (results["Risk Level"] == "LOW").sum()
        )

        st.subheader("📋 Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        st.subheader("📈 Risk Distribution")

        st.bar_chart(
            results["Risk Level"].value_counts()
        )

        csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Report",
            data=csv,
            file_name="fraud_risk_report.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")

# ==========================
# FOOTER
# ==========================

st.markdown("---")
st.caption(
    "AI Risk Manager | Fraud Detection using Machine Learning"
)
