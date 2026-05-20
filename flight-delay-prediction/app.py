import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Flight Delay Prediction", layout="centered")

# Title
st.title("✈️ Flight Delay Prediction App")
st.write("Predict whether a flight will be delayed or not")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_excel("flight delay.xlxs.xlsx")
    df.columns = df.columns.str.strip()
    return df

try:
    fly = load_data()
except:
    st.error("❌ Dataset file not found. Please upload the Excel file.")
    st.stop()

# Select features & target
X = fly[['DayOfWeek', 'Time', 'Length']]
y = fly['Delay']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)

st.subheader("📊 Model Accuracy")
st.write(f"Accuracy: **{accuracy * 100:.2f}%**")

st.divider()

# User input
st.subheader("🧾 Enter Flight Details")

day = st.number_input("Day of Week (1 = Monday, 7 = Sunday)", 1, 7, 3)
time = st.number_input("Time (HHMM format)", 0, 2359, 900)
length = st.number_input("Flight Length (minutes)", 1, 1000, 200)

# Prediction
if st.button("Predict Delay"):
    input_df = pd.DataFrame(
        [[day, time, length]],
        columns=['DayOfWeek', 'Time', 'Length']
    )

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.error("⏰ Result: The flight **WILL BE DELAYED**")
    else:
        st.success("✅ Result: The flight **WILL NOT BE DELAYED**")

# Footer
st.write("---")
st.caption("Machine Learning Project using Logistic Regression")

