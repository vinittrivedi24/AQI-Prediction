import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 📥 Load dataset
df = pd.read_csv("city_day.csv")

# 🧹 Clean columns
df.columns = df.columns.str.strip()

# Keep required columns
df = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2', 'AQI']]
df = df.dropna()

# 🎯 Features & Target
X = df[['PM2.5', 'PM10', 'NO2', 'CO', 'SO2']]
y = df['AQI']

# ✂️ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 🌲 Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔮 Prediction
y_pred = model.predict(X_test)

# 📊 Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# 💾 Save model
joblib.dump(model, "aqi_model.pkl")

# 📊 GRAPH 1: Actual vs Predicted
plt.figure()
plt.plot(y_test.values[:30], label="Actual AQI")
plt.plot(y_pred[:30], label="Predicted AQI")
plt.title("AQI Prediction Trend")
plt.legend()
plt.show()

# 🌍 GRAPH 2: City-wise pollution (if City column exists)
if "City" in df.columns:
    top_cities = df.groupby("City")["AQI"].mean().sort_values(ascending=False).head(10)

    plt.figure()
    sns.barplot(x=top_cities.values, y=top_cities.index)
    plt.title("Top 10 Polluted Cities")
    plt.xlabel("AQI")
    plt.ylabel("City")
    plt.show()