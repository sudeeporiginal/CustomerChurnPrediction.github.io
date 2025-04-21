Customer Churn Prediction - Machine Learning Project

📌 Project Overview

This project aims to build a machine learning model that predicts customer churn in a telecommunications company. By analyzing customer attributes such as tenure, monthly charges, contract type, and service usage, the model identifies whether a customer is likely to leave the service.

⚙️ Tools and Technologies Used
- Python
- Scikit-learn
- XGBoost
- Pandas & NumPy
- Streamlit
- Matplotlib & Seaborn
- Google Colab & VS Code
  
📊 Data Preprocessing
- Missing value treatment
- Encoding categorical variables
- Feature scaling using StandardScaler
- Addressing class imbalance using SMOTE
  
📈 Modeling Approach
- Trained multiple models: Logistic Regression, Random Forest, XGBoost
- Applied hyperparameter tuning using GridSearchCV
- Combined top models using a Voting Classifier ensemble
- Focused on maximizing recall and F1-score to capture more churn cases
  
📦 Files in this Repository
- `churn_prediction_app.py`: Streamlit code for the web app
- `voting_classifier_model.pkl`: Trained ensemble model
- `scaler.pkl`: Scaler used for feature normalization
- `model_columns.pkl`: Feature column order used during training
  
🚀 Running the App Locally
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Run the Streamlit app:
   ```bash
   streamlit run churn_prediction_app.py
   ```
   
