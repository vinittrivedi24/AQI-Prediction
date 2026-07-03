import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor

X = [
    [50, 100, 30, 0.5, 10],
    [30, 60, 20, 0.3, 5],
    [100, 150, 60, 1.0, 20],
    [200, 300, 80, 2.0, 30]
]

y = [80, 40, 150, 250]

model = RandomForestRegressor()
model.fit(X, y)

st.title("🌫️ AQI Prediction App (No Dataset Needed)")

pm25 = st.slider("PM2.5", 0, 500, 50)
pm10 = st.slider("PM10", 0, 500, 80)
no2 = st.slider("NO2", 0, 200, 30)
co = st.slider("CO", 0.0, 5.0, 0.5)
so2 = st.slider("SO2", 0, 100, 10)

if st.button("Predict AQI"):
    input_data = np.array([[pm25, pm10, no2, co, so2]])
    prediction = model.predict(input_data)[0]

    st.success(f"🌫️ Predicted AQI: {prediction:.2f}")

    if prediction <= 50:
        st.info("Good Air Quality 😊")
    elif prediction <= 100:
        st.warning("Moderate Air Quality 😐")
    else:
        st.error("Poor Air Quality 😷")