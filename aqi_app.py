import streamlit as st
import numpy as np
import joblib

# Load model
model = joblib.load("aqi_model.pkl")

st.title("🌫️ AQI Prediction App")
st.write("Predict Air Quality using pollutant levels")

# 🎛️ Input sliders
pm25 = st.slider("PM2.5", 0, 500, 50)
pm10 = st.slider("PM10", 0, 500, 80)
no2 = st.slider("NO2", 0, 200, 30)
co = st.slider("CO", 0.0, 5.0, 0.5)
so2 = st.slider("SO2", 0, 100, 10)

# 🔘 Button
if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, co, so2]])
    prediction = model.predict(input_data)[0]

    st.success(f"🌫️ Predicted AQI: {prediction:.2f}")

    # Simple explanation
    if prediction <= 50:
        st.info("😊 Good Air Quality")
    elif prediction <= 100:
        st.warning("😐 Moderate Air Quality")
    else:
        st.error("😷 Poor Air Quality")