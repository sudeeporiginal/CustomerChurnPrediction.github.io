# 📊 Customer Churn Prediction App

This is a Streamlit web application that predicts whether a customer will churn or stay, based on their service usage data. The model uses machine learning trained on telecom customer data.

---

## 🚀 Features

- Interactive UI built with Streamlit
- User-friendly input forms for customer details
- Displays churn prediction and probability
- Uses pre-trained machine learning model (`best_model.pkl`)
- Handles data preprocessing including label encoding and scaling

---

## 📁 Project Structure

Customer Churn Prediction/

├── customer\_churn\_prediction.py

├── README.md

├── best\_model.pkl

├── encoder.pkl

├── scaler.pkl

└── feature\_order.pkl


# **## 🛠️ Installation & Run**



## **1. Clone the repo**
bash
git clone [https://github.com/your-username/customer-churn-app.git](https://github.com/your-username/customer-churn-app.git)
cd customer-churn-app

## 2. Install dependencies
pip install -r requirements.txt

## 3. Add model files
The model file (best\_model.pkl) is not included in the repository due to size limits.
Download the model from this link: [Download best_model.pkl](https://drive.google.com/file/d/1d9WvTRVIKgowANL6pK-2FJeQvPcBsLeP/view?usp=sharing)
Place it in the root directory of the project (same folder as customer\_churn\_prediction.py)

## 4. Run the app
streamlit run customer_churn_prediction.py


### 🙌 Acknowledgements
Built using Streamlit
Machine Learning with scikit-learn and pandas


