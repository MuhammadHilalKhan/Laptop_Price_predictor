import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

# Title & Description
st.title("❤️ Heart Disease Prediction App")
st.markdown("""
Welcome to the Heart Disease Prediction Web App!  
Please fill in the following patient information to predict the risk of heart disease.
""")

# Sidebar info
st.sidebar.title("🩺 About")
st.sidebar.info("""
This ML model uses **XGBoost** trained on the **UCI Heart Disease Dataset**  
to predict whether a patient is likely to develop heart disease.
""")

# Input Form
with st.form("prediction_form"):
    st.subheader("📝 Enter Patient Data:")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 29, 77, 50)
        sex = st.selectbox("Sex", ("Male", "Female"))
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
        restecg = st.selectbox("Resting ECG", [0, 1])

    with col2:
        thalach = st.number_input("Max Heart Rate Achieved", 70, 210, 150)
        exang = st.selectbox("Exercise Induced Angina", ("Yes", "No"))
        oldpeak = st.slider("ST Depression", 0.0, 6.0, 1.0)
        cp = st.selectbox("Chest Pain Type", [1, 2, 3])
        slope = st.selectbox("Slope of Peak Exercise ST Segment", [1, 2])
        thal = st.selectbox("Thalassemia", [1, 2])

    submitted = st.form_submit_button("Predict")

# Load model only when user submits
if submitted:
    try:
        # Map categorical values
        sex = 1 if sex == "Male" else 0
        fbs = 1 if fbs == "Yes" else 0
        exang = 1 if exang == "Yes" else 0

        # Prepare input
        input_data = np.array([[age, sex, trestbps, chol, fbs, restecg,
                                thalach, exang, oldpeak, cp, slope, thal]])

        # Load model and predict
        model = joblib.load("heart_disease_model.pkl")
        prediction = model.predict(input_data)[0]

        # Display result
        st.subheader("🔍 Prediction Result:")
        if prediction == 1:
            st.error("⚠️ High Risk of Heart Disease")
        else:
            st.success("✅ Low Risk of Heart Disease")

        st.balloons()
        
    except ModuleNotFoundError as e:
        st.error(f"❌ Error: {e}\n\nYou must install missing packages (e.g., `xgboost`) in this environment.")
