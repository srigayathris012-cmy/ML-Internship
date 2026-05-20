# app.py

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Title
st.title("🌧️ Rainfall Prediction App")
st.write("Predict whether it will rain based on windspeed")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Rainfall.csv")
    df.columns = df.columns.str.strip()  # clean column names
    return df

raindf = load_data()

# Handle missing values
raindf = raindf.dropna(subset=["windspeed", "rainfall"])

# Encode target
le = LabelEncoder()
raindf["rainfall_encoded"] = le.fit_transform(raindf["rainfall"])

# Features and target
X = raindf[["windspeed"]]
y = raindf["rainfall_encoded"]

# Train model
LR = LinearRegression()
LR.fit(X, y)

# User input
st.subheader("Enter Input")
windspeed = st.number_input(
    "Windspeed (km/hr)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.1
)

# Prediction
if st.button("Predict Rainfall"):
    input_df = pd.DataFrame({"windspeed": [windspeed]})
    numerical_prediction = LR.predict(input_df)[0]

    # Threshold
    binary_prediction = 1 if numerical_prediction >= 0.5 else 0
    final_prediction = le.inverse_transform([binary_prediction])[0]

    if final_prediction == "yes":
        st.success("🌧️ Prediction: Rainfall YES")
    else:
        st.info("☀️ Prediction: Rainfall NO")

# Show mapping
st.write("### Label Encoding Mapping")
mapping_df = pd.DataFrame({
    "Label": le.classes_,
    "Encoded Value": range(len(le.classes_))
})
st.table(mapping_df)
