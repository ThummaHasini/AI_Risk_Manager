import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Risk Manager",
    page_icon="🛡️",
    layout="wide"
)

# =====================================
# LOAD MODEL & FEATURES
# =====================================

model = joblib.load("fraud_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# =====================================
# FUNCTIONS
# =====================================

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


# =====================================
# HEADER
# =====================================

st.title("🛡️ AI Risk Manager")
st.markdown("### AI-Powered Financial Fraud Risk Detection System")

st.write("""
Upload a transaction dataset to:

✅ Detect Fraud Transactions  
✅ Calculate Risk Scores  
✅ Classify Risk Levels  
✅ Generate Recommendations  
""")

st.divider()

# =====================================
# SAMPLE CSV DOWNLOAD
# =====================================

st.subheader("📥 Sample CSV")

sample_df = pd.DataFrame(columns=feature_names)

sample_csv = sample_df.to_csv(index=False)

st.download_button(
    label="Download Sample CSV",
    data=sample_csv,
    file_name="sample_transaction.csv",
    mime="text/csv"
)

st.info(
    "Expected columns: Time, V1, V2, ..., V28, Amount"
)

st.divider()

# =====================================
# FILE UPLOAD
# =====================================

st.subheader("📂 Upload Transaction CSV")

uploaded_file = st.file_uploader(
    "Choose CSV File",
    type=["csv"]
)

# =====================================
# PROCESS FILE
# =====================================

if uploaded_file is not None:

    try:

        data = pd.read_csv(uploaded_file)

        # Automatically remove Class column
        if "Class" in data.columns:
            data = data.drop("Class", axis=1)

        expected_columns = list(feature_names)

        # Check missing columns
        missing_cols = [
            col for col in expected_columns
            if col not in data.columns
        ]

        # Check extra columns
        extra_cols = [
            col for col in data.columns
            if col not in expected_columns
        ]

        if missing_cols or extra_cols:

            st.error("❌ Invalid CSV Format")

            st.write("### Expected Columns")
            st.code(", ".join(expected_columns))

            if missing_cols:
                st.write("### Missing Columns")
                st.code(", ".join(missing_cols))

            if extra_cols:
                st.write("### Extra Columns")
                st.code(", ".join(extra_cols))

            st.stop()

        # Arrange columns correctly
        data = data[expected_columns]

        # =========================
        # PREDICTIONS
        # =========================

        predictions = model.predict(data)
        probabilities = model.predict_proba(data)[:, 1]

        scores = (probabilities * 100).astype(int)

        results = pd.DataFrame({
            "Prediction": np.where(
                predictions == 1,
                "Fraud",
                "Not Fraud"
            ),
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

        st.success("✅ Analysis Completed Successfully")

        # =========================
        # SUMMARY
        # =========================

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

        # =========================
        # CHART
        # =========================

        st.subheader("📊 Risk Distribution")

        st.bar_chart(
            results["Risk Level"].value_counts()
        )

        st.divider()

        # =========================
        # TABLE
        # =========================

        st.subheader("📋 Detailed Results")

        st.dataframe(
            results,
            use_container_width=True
        )

        # =========================
        # DOWNLOAD REPORT
        # =========================

        report_csv = results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇ Download Risk Report",
            data=report_csv,
            file_name="fraud_risk_report.csv",
            mime="text/csv"
        )

    except Exception as e:

        st.error(
            f"Error processing file: {e}"
        )

# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "AI Risk Manager | Financial Fraud Detection using Machine Learning"
)
