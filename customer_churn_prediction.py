import os
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# # Debug: Show working directory
# st.text(f"Working directory: {os.getcwd()}")

# Load model, encoders, scaler, and feature order
try:
    with open("best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("encoder.pkl", "rb") as f:
        encoders = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("feature_order.pkl", "rb") as f:
        feature_order = pickle.load(f)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Streamlit UI
st.title("📊 Customer Churn Prediction")
st.write("Enter customer details to predict whether they will churn or stay.")

# User Inputs
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (in months)", 0, 72, 1)
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No"])
internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "None"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
online_backup = st.selectbox("Online Backup", ["Yes", "No"])
device_protection = st.selectbox("Device Protection", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
payment_method = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=0.1)
total_charges = st.number_input("Total Charges", min_value=0.0, step=0.1)

# Create input DataFrame
input_data = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})

# Preprocessing
def preprocess_input(data):
    try:
        for col, encoder in encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col])
        numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        data[numeric_cols] = scaler.transform(data[numeric_cols])
        return data
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

# Predict
if st.button("Predict Churn"):
    try:
        processed = preprocess_input(input_data.copy())
        if processed is not None:
            # Reorder columns to match model input
            processed = processed[feature_order]
            prediction = model.predict(processed)[0]
            probability = model.predict_proba(processed)[0][1]
            if prediction == 1:
                st.error(f"🔴 Prediction: Customer will **churn**.")
            else:
                st.success(f"🟢 Prediction: Customer will **stay**.")
            st.info(f"Probability of churn: {probability * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
