import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# App title
st.title("💼 Salary Prediction App")

st.write("Predict Salary based on Years of Experience")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Salary Data.csv")
    return df

data = load_data()

# Handle missing values
data = data.dropna()

# Select input and output
X = data[['Years of Experience']]
y = data['Salary']

# Train model
model = LinearRegression()
model.fit(X, y)

# User input
experience = st.number_input(
    "Enter Years of Experience",
    min_value=0.0,
    max_value=30.0,
    step=0.5
)

# Prediction
if st.button("Predict Salary"):
    prediction = model.predict([[experience]])
    st.success(f"💰 Predicted Salary: ₹ {prediction[0]:,.2f}")

