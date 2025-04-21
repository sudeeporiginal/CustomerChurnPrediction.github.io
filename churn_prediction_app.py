import streamlit as st
import joblib
import pandas as pd

# Load all 3 saved files
model = joblib.load('voting_classifier_model.pkl')
scaler = joblib.load('scaler.pkl')
model_columns = joblib.load('model_columns.pkl')

st.title("Customer Churn Prediction")

# Input form
st.header("Customer Details")

tenure = st.slider("Tenure (in months)", 0, 72, 10)
monthly_charges = st.slider("Monthly Charges", 0, 200, 50)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
phone_service = st.selectbox("Phone Service", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
total_charges = st.number_input("Total Charges", value=200.0)

# Build raw input
input_dict = {
    'Contract': contract,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'TechSupport': tech_support,
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'MultipleLines': multiple_lines,
    'StreamingMovies': streaming_movies,
    'PhoneService': phone_service,
    'Dependents': dependents,
    'TotalCharges': total_charges
}

input_df = pd.DataFrame([input_dict])
input_encoded = pd.get_dummies(input_df)

# Ensure all expected columns are present
for col in model_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Arrange columns in training order
input_encoded = input_encoded[model_columns]

# Scale
input_scaled = scaler.transform(input_encoded)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    st.subheader("Prediction:")
    st.write("🚨 Customer will **Churn**." if prediction == 1 else "✅ Customer will **Stay**.")
