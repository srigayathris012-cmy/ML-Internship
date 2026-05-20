import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Population Growth Prediction", layout="centered")

st.title("📈 Population Growth Prediction using Polynomial Regression")

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    return pd.read_csv("World Population Growth.csv")

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------- Clean Population Column ----------
data['Population'] = data['Population'].astype(str)
data['Population'] = data['Population'].str.replace(',', '')
data['Population'] = pd.to_numeric(data['Population'], errors='coerce')
data = data.dropna()

# ---------- Features & Target ----------
X = data[['Year']]
y = data['Population']

# ---------- Polynomial Degree ----------
degree = st.slider("Select Polynomial Degree", 1, 5, 3)

poly = PolynomialFeatures(degree=degree)
X_poly = poly.fit_transform(X)

# ---------- Train Model ----------
model = LinearRegression()
model.fit(X_poly, y)

# ---------- Prediction ----------
future_year = st.number_input(
    "Enter Future Year",
    min_value=int(X.min().values[0]) + 1,
    max_value=2100,
    step=1
)

future_year_poly = poly.transform([[future_year]])
prediction = model.predict(future_year_poly)

st.success(f"🌍 Predicted Population in {future_year}: {int(prediction[0]):,}")



