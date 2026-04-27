import streamlit as st

# Set page title and favicon
st.set_page_config(page_title="KEMRI CVD Risk Predictor", page_icon="🏥")

# Custom CSS to improve the "Look & Feel"
st.markdown("""
    <style>
    .main {
        background-color: #F8F9FA;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #1E3E72;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Add a header with a simulated logo/title area
st.image("", width=150)
st.title("Cardiovascular Disease Risk Prediction Dashboard")
st.subheader("KEMRI | SLAS Research Program")

import streamlit as st
import joblib
import numpy as np

# 1. Load your pre-trained model
model = joblib.load('cvd_rf_model.pkl')

st.title("Cardiovascular Disease Risk Predictor")
st.write("Enter behavioral and clinical data to assess CVD risk.")

# 2. Layout with Columns for better UI
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 1, 100, 50)
    gender = st.selectbox("Gender", options=[(1, "Male"), (2, "Female")], format_func=lambda x: x[1])[0]
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)

with col2:
    ap_hi = st.slider("Systolic Blood Pressure", 80, 250, 120)
    ap_lo = st.slider("Diastolic Blood Pressure", 40, 150, 80)
    smoke = st.radio("Do you smoke?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    active = st.radio("Are you physically active?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

# 3. Calculation & Prediction Logic
bmi = weight / ((height / 100) ** 2)

# Features must match the exact order and names used during training
input_data = np.array([[age, gender, height, weight, ap_hi, ap_lo, 1, 1, smoke, 0, active, bmi]])

if st.button("Predict Risk"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]
    
    if prediction == 1:
        st.error(f"High Risk Detected! (Probability: {prob*100:.1f}%)")
    else:
        st.success(f"Low Risk Detected. (Probability: {prob*100:.1f}%)")
