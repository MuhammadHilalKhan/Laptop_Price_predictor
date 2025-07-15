# import streamlit as st
# import pickle
# import numpy as np

# # Define file paths
# CLF_MODEL_PATH = "D:/Courses/Streamlit/models/clf.pkl"
# OHE_EMBARKED_PATH = "D:/Courses/Streamlit/models/ohe_embarked.pkl"
# OHE_SEX_PATH = "D:/Courses/Streamlit/models/ohe_sex.pkl"

# # Load all models and encoders
# @st.cache_resource
# def load_model(file_path):
#     with open(file_path, "rb") as file:
#         return pickle.load(file)

# clf = load_model(CLF_MODEL_PATH)
# ohe_embarked = load_model(OHE_EMBARKED_PATH)
# ohe_sex = load_model(OHE_SEX_PATH)

# # App title
# st.title("Passenger Survival Prediction")

# # Input Fields
# st.header("Enter Passenger Details")

# # Sex
# sex_input = st.selectbox("Select Sex", ["male", "female"])

# # Age
# age_input = st.number_input("Enter Age", min_value=0, max_value=120, value=25)

# # Embarked
# embarked_input = st.selectbox("Select Embarkation Point", ["C", "Q", "S"])

# # Other numeric features (e.g., Fare)
# fare_input = st.number_input("Enter Fare Amount", min_value=0.0, step=1.0, value=50.0)

# # Process Inputs
# if st.button("Predict Survival"):
#     # Encode Sex
#     sex_encoded = ohe_sex.transform([[sex_input]]) #.toarray()
    
#     # Encode Embarked
#     embarked_encoded = ohe_embarked.transform([[embarked_input]])#.toarray()
    
#     # Combine all features
#     final_input = np.hstack([sex_encoded, embarked_encoded, [[age_input, fare_input]]])
    
#     # Make Prediction
#     prediction = clf.predict(final_input)
#     prediction_proba = clf.predict_proba(final_input)
    
#     # Display results
#     st.subheader("Prediction Results")
#     if prediction[0] == 1:
#         st.success("The passenger is predicted to survive.")
#     else:
#         st.error("The passenger is predicted not to survive.")
    
#     st.write("Prediction Probability:", prediction_proba)

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score
# import pickle

# # Load dataset
# data = pd.read_csv('random_dataset.csv')

# # Preprocessing: Remove duplicate values
# data = data.drop_duplicates()

# # Display dataset info and first few rows
# print(data.info())
# print(data.head())

# # Exploratory Data Analysis (EDA)
# plt.figure(figsize=(8, 6))
# plt.scatter(data['CGPA'], data['IQ'], c=data['Placements'], cmap='viridis')
# plt.xlabel("CGPA")
# plt.ylabel("IQ")
# plt.title("Scatter plot of CGPA vs IQ colored by Placements")
# plt.colorbar(label="Placements")
# plt.show()

# # Extract input (independent variables) and output (dependent variable)
# X = data.iloc[:, 0:2]  # Independent variables (CGPA, IQ)
# y = data.iloc[:, -1]   # Dependent variable (Placements)

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale features
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# # Train Logistic Regression model
# clf = LogisticRegression()
# clf.fit(X_train, y_train)

# # Predictions and accuracy
# y_pred = clf.predict(X_test)
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Model Accuracy: {accuracy:.2f}")

# # Save model to a file
# with open('model.pkl', 'wb') as model_file:
#     pickle.dump(clf, model_file)
# print("Model saved as 'model.pkl'")

import streamlit as st
import pickle
import numpy as np

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as model_file:
        return pickle.load(model_file)

model = load_model()

# App title and description
st.title("Logistic Regression Model: Placements Prediction")
st.write("""
This app predicts the placement outcome based on a candidate's **CGPA** and **IQ** using a pre-trained logistic regression model.
""")

# Input Fields
st.header("Enter Candidate Details")
cgpa = st.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
iq = st.number_input("Enter IQ", min_value=50, max_value=200, value=100, step=1)

# Predict Button
if st.button("Predict Placement"):
    # Prepare input data
    input_data = np.array([[cgpa, iq]])
    
    # Make prediction
    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)

    # Display results
    st.subheader("Prediction Results")
    if prediction[0] == 1:
        st.success("The candidate is likely to be placed.")
    else:
        st.error("The candidate is unlikely to be placed.")
    
    st.write("Prediction Probabilities:")
    st.write(f"Not Placed: {prediction_proba[0][0]:.2f}, Placed: {prediction_proba[0][1]:.2f}")
page_bg = f"""
<style>
.st-emotion-cache-bm2z3a {{
    display: flex;
    flex-direction: column;
    width: 100%;
    overflow: auto;
    -webkit-box-align: center;
    background-color: #398172;
    items: center;
}}
<style\>

"""
st.markdown(page_bg, unsafe_allow_html=True)