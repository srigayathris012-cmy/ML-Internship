import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction App")
st.write("Predict whether a person is diabetic or not")
st.write("👇 Enter details below")

# ---------- LOAD DATA ----------
try:
    dia = pd.read_excel("diabetes (1).xlsx")
except:
    st.error("❌ File 'diabetes (1).xlsx' not found")
    st.stop()

# ---------- MODEL ----------
X = dia[['age', 'mass', 'insu', 'plas']]
y = dia['class']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# ---------- INPUTS ----------
st.subheader("Patient Details")

age = st.number_input("Age", 1, 100, 25)
mass = st.number_input("BMI (mass)", 0.0, 70.0, 25.0)
insulin = st.number_input("Insulin Level", 0, 900, 80)
plasma = st.number_input("Plasma Glucose Level", 0, 300, 120)

# ---------- PREDICTION ----------
if st.button("Predict"):
    input_df = pd.DataFrame(
        [[age, mass, insulin, plasma]],
        columns=['age', 'mass', 'insu', 'plas']
    )

    result = model.predict(input_df)[0]

    if result == "tested_positive":
        st.error("❌ Tested POSITIVE for Diabetes")
    else:
        st.success("✅ Tested NEGATIVE for Diabetes")
