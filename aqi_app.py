import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data.csv")

df.columns = df.columns.str.strip()
df = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'AQI']]
df = df.dropna()

X = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']]
y = df['AQI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model INSIDE app (no pickle needed)
model = RandomForestRegressor()
model.fit(X_train, y_train)

# UI
st.title("🌫️ AQI Prediction App")

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