# 🛡️ AI Risk Manager

AI-Powered Financial Fraud Detection System using Machine Learning.

## 📌 Overview

AI Risk Manager is a machine learning-based web application that detects potentially fraudulent financial transactions and classifies them into different risk categories.

The system analyzes transaction data, predicts fraud probability, generates risk scores, and provides actionable recommendations for risk management.

---

## 🚀 Live Demo

🔗 Streamlit App:
https://airiskmanager-hd.streamlit.app/

---

## ✨ Features

- Fraud Detection using Machine Learning
- Risk Score Calculation
- Risk Level Classification
  - Low Risk
  - Medium Risk
  - High Risk
- Automated Recommendations
- CSV Upload Support
- Downloadable Risk Reports
- Interactive Dashboard
- Risk Distribution Visualization

---

## 🧠 Machine Learning Model

The project uses a trained Random Forest Classifier on credit card transaction data.

### Input Features

- Time
- V1 – V28
- Amount

### Output

- Fraud / Not Fraud Prediction
- Fraud Probability
- Risk Score
- Risk Level
- Recommendation

---

## 📂 Project Structure

```text
AI_Risk_Manager/
│
├── app.py
├── fraud_model.pkl
├── feature_names.pkl
├── sample_transaction.csv
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

## 📊 Dataset

Credit Card Fraud Detection Dataset

Features:
- Time
- V1–V28 (PCA-transformed features)
- Amount

Target:
- Class
  - 0 = Legitimate Transaction
  - 1 = Fraudulent Transaction

---

## ▶️ How to Run Locally

### Clone Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI_Risk_Manager
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

---

## 📋 Usage

1. Download the sample CSV file.
2. Prepare transaction data using the same format.
3. Upload the CSV file.
4. View:
   - Fraud Predictions
   - Risk Scores
   - Risk Distribution
   - Recommendations
5. Download the generated risk report.

---

## 🛠 Challenges Faced

- Handling large transaction datasets
- Matching uploaded CSV format with model features
- Managing feature consistency during deployment
- Streamlit cloud deployment issues
- Risk scoring and classification logic

### Solutions

- Automatic column validation
- Automatic removal of unnecessary columns
- Feature alignment using saved feature names
- Optimized deployment workflow
- Dynamic risk scoring system

---

## 🔮 Future Enhancements

- Real-time transaction monitoring
- Deep Learning based fraud detection
- User authentication system
- Database integration
- API deployment
- Advanced analytics dashboard

---

## 👩‍💻 Author

Thumma Hasini

Final Year B.Tech (AI & ML)

---

## 📜 License

This project is developed for academic and educational purposes.
