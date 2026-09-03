import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

# =========================
# LOAD FILES
# =========================

model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =========================
# FUNCTIONS
# =========================

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


# =========================
# UI HEADER
# =========================

st.title("🛡️ AI Risk Manager")
st.markdown("### AI-Powered Financial Fraud Risk Detection System")

st.write(
    """
Upload a transaction dataset and the system will:

✅ Detect fraud transactions  
✅ Generate fraud probability  
✅ Calculate risk score  
✅ Assign risk level  
✅ Recommend action
"""
)

st.divider()

# =========================
# FILE UPLOAD
# =========================

st.subheader("📂 Upload Transaction CSV")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

# =========================
# PROCESS FILE
# =========================

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        # ---------------------
        # COLUMN VALIDATION
        # ---------------------

        uploaded_columns = list(data.columns)
        expected_columns = list(feature_names)

        if uploaded_columns != expected_columns:

            st.error("❌ Invalid CSV Format")

            st.write("### Expected Columns")

            st.code(", ".join(expected_columns))

            st.write("### Uploaded Columns")

            st.code(", ".join(uploaded_columns))

            st.warning(
                f"""
Expected exactly {len(expected_columns)} columns.

Your file contains {len(uploaded_columns)} columns.

Please upload a CSV having the same structure used for model training.
"""
            )

        else:

            # ---------------------
            # PREDICTIONS
            # ---------------------

            predictions = model.predict(data)
            probabilities = model.predict_proba(data)[:, 1]

            scores = (probabilities * 100).astype(int)

            results = pd.DataFrame({
                "Prediction": [
                    "Fraud" if p == 1 else "Not Fraud"
                    for p in predictions
                ],
                "Fraud Probability (%)":
                    np.round(probabilities * 100, 2),
                "Risk Score":
                    scores
            })

            results["Risk Level"] = results["Risk Score"].apply(
                risk_level
            )

            results["Recommendation"] = results[
                "Risk Score"
            ].apply(recommendation)

            # ---------------------
            # DASHBOARD
            # ---------------------

            st.success("✅ Analysis Completed Successfully")

            total = len(results)

            high = (
                results["Risk Level"] == "HIGH"
            ).sum()

            medium = (
                results["Risk Level"] == "MEDIUM"
            ).sum()

            low = (
                results["Risk Level"] == "LOW"
            ).sum()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Transactions", total)
            col2.metric("High Risk", high)
            col3.metric("Medium Risk", medium)
            col4.metric("Low Risk", low)

            st.divider()

            st.subheader("📊 Risk Distribution")

            st.bar_chart(
                results["Risk Level"].value_counts()
            )

            st.divider()

            st.subheader("📋 Detailed Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            # ---------------------
            # DOWNLOAD REPORT
            # ---------------------

            csv = results.to_csv(index=False)

            st.download_button(
                label="⬇ Download Risk Report",
                data=csv,
                file_name="fraud_risk_report.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(f"Error processing file: {e}")

# =========================
# FOOTER
# =========================

st.divider()

st.caption(
    "AI Risk Manager | Financial Fraud Detection using Machine Learning"
)
