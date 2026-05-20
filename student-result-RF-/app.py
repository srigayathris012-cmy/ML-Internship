import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Exam Result Prediction", page_icon="📘")

st.title("📘 Exam Result Prediction")
st.write("Prediction using Random Forest Algorithm")

# Load dataset
data = pd.read_csv("results.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

# Encode Results column if it is text
if data['Results'].dtype == 'object':
    le = LabelEncoder()
    data['Results'] = le.fit_transform(data['Results'])
else:
    le = None

# Features & Target
X = data[['Hindi', 'English', 'Maths', 'History']]
y = data['Results']

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

st.markdown("---")
st.subheader("📝 Enter Student Marks")

hindi = st.number_input("Hindi Marks", 0, 100)
english = st.number_input("English Marks", 0, 100)
maths = st.number_input("Maths Marks", 0, 100)
history = st.number_input("History Marks", 0, 100)

if st.button("🔍 Predict Result"):
    input_data = np.array([[hindi, english, maths, history]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("🎉 Prediction: PASS")
    else:
        st.error("❌ Prediction: FAIL")

