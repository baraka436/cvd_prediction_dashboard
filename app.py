import streamlit as st
import joblib
import numpy as np

# 1. Page & Model Setup
st.set_page_config(page_title="KEMRI CVD Risk Predictor", page_icon="🏥")
model = joblib.load('cvd_prediction_risk.pkl') # Loaded your Mentor-guided RF model

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
st.image("kemri_logo.png", width=150)
st.title("Cardiovascular Disease Risk Prediction Dashboard")
st.subheader("KEMRI | SLAS Research Program")

st.title("KEMRI | SLAS CVD Risk Dashboard")
st.markdown("### Enter the required data below to access your personalized cardiovascular disease risk assessment")

<<<<<<< HEAD
=======
# 1. Load your pre-trained model
model = joblib.load('cvd_prediction_risk.pkl')

st.title("Cardiovascular Disease Risk Predictor")
st.write("Enter behavioral and clinical data to assess CVD risk.")

# 2. Layout with Columns for better UI
>>>>>>> 78c410912231dd3562cc87ef9c13eba9ad3d3ddb
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age (Years)", 1, 100, 45)
    age_days = age * 365
    gender = st.selectbox("Gender", options=[(1, "Male"), (2, "Female")], format_func=lambda x: x[1])[0]
    height = st.number_input("Height (cm)", 100, 250, 170)
    weight = st.number_input("Weight (kg)", 30, 200, 70)
    waist = st.number_input("Waist Circumference (cm)", 40, 200, 85)

with col2:
    ap_hi = st.slider("Systolic BP", 80, 250, 120)
    ap_lo = st.slider("Diastolic BP", 40, 150, 80)
    active = st.radio("Physically Active?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    
    # --- Conditional Smoking Logic ---
    smoke_status = st.radio("Do you smoke?", ["No", "Yes"])
    smoking_multiplier = 1.0 # Default base risk
    
    if smoke_status == "Yes":
        smoke_val = 1
        freq = st.select_slider("Frequency", options=["Occasional", "Daily", "Heavy"], value="Daily")
        # Multipliers based on clinical evidence of dose-response
        freq_map = {"Occasional": 1.1, "Daily": 1.5, "Heavy": 2.0}
        smoking_multiplier = freq_map[freq]
    else:
        smoke_val = 0
        shs = st.radio("Live with people who smoke? (Secondhand Exposure)", ["No", "Yes"])
        if shs == "Yes":
            smoking_multiplier = 1.3 # SHS increases risk by ~30%

# 2. Body Shape Index (ABSI) Calculation
# ABSI = WC / (BMI^(2/3) * height^(1/2))
bmi = weight / ((height / 100) ** 2)
absi = (waist / 100) / ((bmi**(2/3)) * ((height / 100)**(1/2))) #

# 3. Model Prediction Logic
# Features: [age, gender, height, weight, ap_hi, ap_lo, chol(1), gluc(1), smoke, alco(0), active, bmi]
input_data = np.array([[age_days, gender, height, weight, ap_hi, ap_lo, 1, 1, smoke_val, 0, active, bmi]])

if st.button("Calculate Integrated Risk"):
    # Base prediction from your trained Random Forest model
    base_prob = model.predict_proba(input_data)[0][1]
    
    # Adjust probability based on lifestyle frequency factors and ABSI
    # Note: ABSI > 0.08 is typically considered higher risk
    absi_adjustment = 1.1 if absi > 0.08 else 1.0
    final_prob = min(base_prob * smoking_multiplier * absi_adjustment, 1.0)
    
    st.divider()
    if final_prob > 0.5:
        st.error(f"High Risk Detected: {final_prob*100:.1f}%")
        st.write(f"**ABSI:** {absi:.4f} | **BMI:** {bmi:.1f}")
    else:
        st.success(f"Low Risk Detected: {final_prob*100:.1f}%")
st.warning("Recommendation: Please consult a healthcare professional for a comprehensive cardiovascular screening.")
    # --- Expanded Risk Explanation Logic ---
with st.expander("🔍 See why you received this score "):
        reasons = []
        
        # 1. Age & Gender (The Non-Modifiable Factors)
        if age > 50:
            reasons.append(f"⏳ **Age ({age} yrs):** Age is a primary driver. As we get older, blood vessels naturally lose elasticity, which increases the baseline probability in the model.")
        
        if gender == 1: # Male
            if age >= 45:
                reasons.append("♂️ **Gender:** Statistically,men often face earlier risks due to lower levels of protective hormones.")
        else: # Female
            if age >= 55:
                reasons.append("♀️ **Gender/Age:** Risk for women is often lower early on due to oestrogen ,but it typically aligns with men's risk after menopause.")

        # 2. Blood Pressure
        if ap_hi > 130 or ap_lo > 85:
            reasons.append(f"🩺 **Blood Pressure ({ap_hi}/{ap_lo}):** Your levels are in the hypertensive range. Think of your arteries as a hose pipe ,high pressure causes the walls to stretch and weaken over time .This silent strain creates micro-tears where cholesterol can easily build up ,narrowing your arteries and forcing your heart to work much harder.")

        # 3. Smoking & Frequency
        if smoke_status == "Yes":
            reasons.append(f"🚬 **Smoking ({freq}):** You identified as a {freq.lower()} smoker. Nicotine increases heart rate and carbon monoxide reduces oxygen in your blood, compounding your risk.")
        elif shs == "Yes":
            reasons.append("🏠 **Passive Smoking:** Living with smokers exposes you to toxins that cause inflammation, contributing to your risk probability.")

        # 4. Body Shape (ABSI) vs Weight
        if absi > 0.083:
            reasons.append(f"⚖️ **ABSI ({absi:.3f}):** Even if weight is normal, your waist-to-hip ratio suggests 'visceral fat' around organs, which is more dangerous for the heart than general weight.")
        elif bmi > 25:
            reasons.append(f"⚖️ **BMI ({bmi:.1f}):** Being overweight increases the workload on your heart muscle.")

        # Display the breakdown
        if reasons:
            st.write("### Factors influencing your score:")
            for r in reasons:
                st.write(r)
        else:
            st.write("✅ Your clinical markers are within healthy ranges .Keep maintaining your current lifestyle!")



