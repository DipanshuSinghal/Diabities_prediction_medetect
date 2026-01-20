import streamlit as st
import joblib
import numpy as np

# Load model
with open("classifier_compressed.joblib", "rb") as f:
    model = joblib.load(f)

st.title("🩺 Diabetes Prediction App")
st.write("Fill in patient details below:")

# ---- STRING INPUTS ----
gender_str = st.selectbox(
    "Gender",
    ["Female", "Male", "Other"]
)

smoking_str = st.selectbox(
    "Smoking History",
    ["Never", "Current", "Former", "Ever", "Not Current", "No Info"]
)

# ---- NUMERIC INPUTS ----
age = st.number_input("Age", min_value=1, max_value=120)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0)
HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0)
blood_glucose_level = st.number_input("Blood Glucose Level", min_value=50, max_value=400)

hypertension = st.selectbox("Hypertension", ["No", "Yes"])
heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])

# ---- MAP STRING → INTEGER ----
gender_map = {"Female": 0, "Male": 1, "Other": -1}
smoking_map = {
    "Never": 1,
    "Current": 0,
    "Former": 2,
    "Ever": 3,
    "Not Current": 4,
    "No Info": -1
}

gender = gender_map[gender_str]
smoking_history = smoking_map[smoking_str]
hypertension = 1 if hypertension == "Yes" else 0
heart_disease = 1 if heart_disease == "Yes" else 0

# ---- PREDICTION ----
if st.button("Predict Diabetes"):
    input_data = np.array([[
        gender,
        age,
        hypertension,
        heart_disease,
        smoking_history,
        bmi,
        HbA1c_level,
        blood_glucose_level
    ]])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High risk of diabetes (Probability: {probability:.2%})")
    else:
        st.success(f"✅ Low risk of diabetes (Probability: {probability:.2%})")
