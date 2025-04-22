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

📦Customer Churn Prediction/ ├── customer_churn_prediction.py # Main Streamlit app 
    ├── encoder.pkl # Encoded mappings for categorical variables 
    ├── scaler.pkl # StandardScaler used during training 
    ├── feature_order.pkl # Correct order of features for prediction 
    └── README.md # This file


---

## 🛠️ Installation & Run

### **1. Clone the repo**
```bash
git clone https://github.com/your-username/customer-churn-app.git
cd customer-churn-app

### **2. Install dependencies**


---
 ## 

### 2. Install dependencies
pip install -r requirements.txt

### 3. Add model files
The model file (best_model.pkl) is not included in the repository due to size limits.
Download the model from this link: Download best_model.pkl
Place it in the root directory of the project (same folder as customer_churn_prediction.py)

4. Run the app
streamlit run customer_churn_prediction.py

📌 Notes
If the app throws a feature mismatch error, make sure the input feature order matches feature_order.pkl.
All .pkl files must match what was used during training.

🙌 Acknowledgements
Built using Streamlit
Machine Learning with scikit-learn and pandas


