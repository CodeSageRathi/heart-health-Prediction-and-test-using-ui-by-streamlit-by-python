import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Set page configuration
st.set_page_config(page_title="Heart Health", page_icon="❤️", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
    h1, h2, h3 {
        color: #d63384;
        text-align: center;
    }
    .stButton>button {
        background-color: #d63384;
        color: white;
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: bold;
    }
    .stDownloadButton>button {
        background-color: #20c997;
        color: white;
        font-weight: bold;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>❤️ Heart Health Checker</h1>", unsafe_allow_html=True)

# Safe casting helper
def safe_cast(val, to_type, fallback=None):
    try:
        return to_type(val)
    except:
        return fallback

# Input Form
with st.form("heart_form"):
    st.subheader("🩺 Please enter your health details:")

    age = safe_cast(st.text_input("1. Age", placeholder="e.g., 45"), int)
    sex = st.radio("2. Gender", ["Female", "Male"])
    cp = st.selectbox("3. Chest pain level", ["No pain", "Mild pain", "Moderate pain", "Severe pain"])
    trestbps = safe_cast(st.text_input("4. Resting blood pressure (mmHg)", placeholder="e.g., 120"), int)
    chol = safe_cast(st.text_input("5. Cholesterol (mg/dL)", placeholder="e.g., 200"), int)
    fbs = st.radio("6. Fasting blood sugar > 120 mg/dL?", ["No", "Yes"])
    restecg = st.selectbox("7. ECG results", ["Normal", "Abnormal", "Possible hypertrophy"])
    thalach = safe_cast(st.text_input("8. Max heart rate achieved", placeholder="e.g., 150"), int)
    exang = st.radio("9. Exercise-induced angina?", ["No", "Yes"])
    oldpeak = safe_cast(st.text_input("10. ST depression (oldpeak)", placeholder="e.g., 1.4"), float)
    slope = st.selectbox("11. ST segment slope", ["Rising", "Flat", "Falling"])
    ca = safe_cast(st.text_input("12. Number of blocked vessels (0–3)", placeholder="e.g., 0"), int)
    thal = st.selectbox("13. Thalassemia type", ["Normal", "Fixed defect", "Reversible defect"])

    submitted = st.form_submit_button("✅ Predict")

# Predict after submit
if submitted:
    required_fields = [age, trestbps, chol, thalach, oldpeak, ca]
    if any(field is None for field in required_fields):
        st.error("🚫 Please fill all fields with valid numeric values.")
    else:
        try:
            # Convert categorical inputs to model-friendly numeric values
            sex_val = 0 if sex == "Female" else 1
            cp_val = ["No pain", "Mild pain", "Moderate pain", "Severe pain"].index(cp)
            fbs_val = 0 if fbs == "No" else 1
            restecg_val = ["Normal", "Abnormal", "Possible hypertrophy"].index(restecg)
            exang_val = 0 if exang == "No" else 1
            slope_val = ["Rising", "Flat", "Falling"].index(slope)
            thal_val = ["Normal", "Fixed defect", "Reversible defect"].index(thal)

            # Final input features (13 inputs)
            features = [age, sex_val, cp_val, trestbps, chol, fbs_val,
                        restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]

            # Load model and scaler
            scaler = joblib.load("scaler.pkl")
            model = joblib.load("heart_disease_model.pkl")
            X_scaled = scaler.transform([features])

            pred = model.predict(X_scaled)[0]
            prob = model.predict_proba(X_scaled)[0][1] * 100

            result = "✅ Heart Looks Healthy" if pred == 0 else "⚠️ At Risk of Heart Disease"
            recommendation = (
                "Please consult a cardiologist and consider lifestyle changes."
                if pred == 1 else
                "Keep up your healthy habits! Stay active and eat well."
            )

            # Display prediction results
            st.markdown("---")
            st.markdown(f"### Result: {result}")
            st.markdown(f"**Predicted Risk Probability:** {prob:.2f}%")

            if pred == 1:
                st.warning(f"💬 Recommendation: {recommendation}")
            else:
                st.success(f"💬 Recommendation: {recommendation}")

            # Show summary
            st.markdown("## 🖨️ Your Input Summary")
            st.text(f"""
Age: {age}
Gender: {sex}
Chest Pain: {cp}
Resting BP: {trestbps}
Cholesterol: {chol}
Fasting Sugar: {fbs}
ECG Result: {restecg}
Max Heart Rate: {thalach}
Exercise Angina: {exang}
ST Depression: {oldpeak}
ST Slope: {slope}
Blocked Vessels: {ca}
Thalassemia: {thal}
""")

            # Downloadable report
            report = f"""Heart Health Report
-------------------------
Result: {'At Risk' if pred == 1 else 'Healthy'}
Risk Probability: {prob:.2f}%
Recommendation: {recommendation}

Input Summary:
Age: {age}
Gender: {sex}
Chest Pain: {cp}
Resting BP: {trestbps}
Cholesterol: {chol}
Fasting Sugar: {fbs}
ECG: {restecg}
Max Heart Rate: {thalach}
Exercise Angina: {exang}
Oldpeak: {oldpeak}
ST Slope: {slope}
Blocked Vessels: {ca}
Thalassemia: {thal}
"""
            b64 = base64.b64encode(report.encode()).decode()
            st.download_button("📥 Download Report", data=report, file_name="heart_report.txt", mime="text/plain")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
