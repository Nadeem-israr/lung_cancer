import streamlit as st
import numpy as np
import joblib
import pandas as pd

# Load trained model and scaler
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit UI
st.title("Lung Cancer Prediction App 🚑")
st.write("Enter patient details to predict the likelihood of lung cancer.")

# User Input Fields
age = st.number_input("Age", min_value=1, max_value=120, value=50)
gender = st.radio("Gender", [0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
smoking = st.radio("Smoking", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
finger_discoloration = st.radio("Finger Discoloration", [0, 1])
mental_stress = st.radio("Mental Stress", [0, 1])
pollution_exposure = st.radio("Exposure to Pollution", [0, 1])
long_term_illness = st.radio("Long-term Illness", [0, 1])
energy_level = st.slider("Energy Level", min_value=0.0, max_value=100.0, value=50.0)
immune_weakness = st.radio("Immune Weakness", [0, 1])
breathing_issue = st.radio("Breathing Issue", [0, 1])
alcohol_consumption = st.radio("Alcohol Consumption", [0, 1])
throat_discomfort = st.radio("Throat Discomfort", [0, 1])
oxygen_saturation = st.slider("Oxygen Saturation (%)", min_value=80.0, max_value=100.0, value=95.0)
chest_tightness = st.radio("Chest Tightness", [0, 1])
family_history = st.radio("Family History of Lung Cancer", [0, 1])
smoking_family_history = st.radio("Family Smoking History", [0, 1])
stress_immune = st.radio("Stress Impact on Immunity", [0, 1])

# Prepare input data
input_data = np.array([
    age, gender, smoking, finger_discoloration, mental_stress,
    pollution_exposure, long_term_illness, energy_level, immune_weakness,
    breathing_issue, alcohol_consumption, throat_discomfort, oxygen_saturation,
    chest_tightness, family_history, smoking_family_history, stress_immune
]).reshape(1, -1)

# Scale input data
input_scaled = scaler.transform(input_data)

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    result = "🚨 Lung Cancer Detected 🚨" if prediction == 1 else "✅ No Lung Cancer"
    
    st.subheader("Prediction Result:")
    st.markdown(f"<h2 style='color: red;'>{result}</h2>" if prediction == 1 else f"<h2 style='color: green;'>{result}</h2>", unsafe_allow_html=True)

    # Show additional medical advice based on prediction
    if prediction == 1:
        st.warning("Consult a doctor immediately for further evaluation.")
    else:
        st.success("Maintain a healthy lifestyle to reduce future risks.")
