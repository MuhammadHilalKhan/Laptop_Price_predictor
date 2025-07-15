# import streamlit as st  
# import joblib  
# import numpy as np  

# # Load the trained model  
# model = joblib.load("Employee_model.pkl")  

# st.title("Employee Attrition Prediction")  
# st.write("Enter employee details to predict attrition")  

# # User input fields (adjust based on dataset features)  
# age = st.number_input("Age", min_value=18, max_value=65, step=1)  
# monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, step=500)  
# job_satisfaction = st.slider("Job Satisfaction (1-4)", 1, 4, 2)  
# work_life_balance = st.slider("Work-Life Balance (1-4)", 1, 4, 2)  
# total_working_years = st.number_input("Total Working Years", min_value=0, max_value=40, step=1)  

# # Convert input to numpy array  
# input_data = np.array([[age, monthly_income, job_satisfaction, work_life_balance, total_working_years]])  

# # Predict button  
# if st.button("Predict Attrition"):  
#     prediction = model.predict(input_data)  
#     result = "Will Leave (Attrition)" if prediction[0] == 1 else "Will Stay"  
#     st.write("Prediction:", result)  



import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load trained model, encoders, and scaler
model, label_encoders, scaler = joblib.load("Employee_model_fixed.pkl")

# Define expected features from dataset
expected_features = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction',
    'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction',
    'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18',
    'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
    'YearsWithCurrManager'
]

# Streamlit UI
st.title("Employee Attrition Prediction")
st.write("Enter employee details to predict the likelihood of attrition.")

# Input fields for numeric and categorical features
input_data = {}
for feature in expected_features:
    if feature in label_encoders:
        input_data[feature] = st.selectbox(f"{feature}", label_encoders[feature].classes_)
    else:
        input_data[feature] = st.number_input(f"{feature}", min_value=0.0, format='%f')

# Convert categorical inputs to numerical values
for feature in label_encoders:
    input_data[feature] = label_encoders[feature].transform([input_data[feature]])[0]

# Convert to DataFrame and scale
input_df = pd.DataFrame([input_data])
input_df[input_df.columns] = scaler.transform(input_df)

# Predict
if st.button("Predict Attrition"):
    prediction = model.predict(input_df)
    st.write("Predicted Attrition: ", "Yes" if prediction[0] == 1 else "No")
