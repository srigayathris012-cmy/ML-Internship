import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

st.set_page_config(page_title="Google Stock Prediction (RNN)", layout="centered")

st.title("📈 Google Stock Price Prediction using RNN")

st.write("This app predicts stock prices using a **Recurrent Neural Network (SimpleRNN)**.")

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv("GOOGL_2006-01-01_to_2018-01-01.csv")
    return df

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# ---------- Use Close Price ----------
close_price = data[['Close']]

# ---------- Scaling ----------
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_price)

# ---------- Create Time-Series ----------
def create_dataset(dataset, time_step=10):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:i+time_step, 0])
        y.append(dataset[i+time_step, 0])
    return np.array(X), np.array(y)

time_step = 10
X, y = create_dataset(scaled_data, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

# ---------- Build RNN Model ----------
model = Sequential()
model.add(SimpleRNN(50, activation='tanh', input_shape=(time_step, 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# ---------- Train Model ----------
with st.spinner("Training RNN model..."):
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

# ---------- Prediction ----------
predicted = model.predict(X, verbose=0)
predicted = scaler.inverse_transform(predicted)
actual = scaler.inverse_transform(y.reshape(-1,1))

# ---------- Plot ----------
st.subheader("📊 Actual vs Predicted Stock Price")

fig, ax = plt.subplots()
ax.plot(actual, label="Actual Price")
ax.plot(predicted, label="Predicted Price")
ax.set_xlabel("Time")
ax.set_ylabel("Stock Price")
ax.legend()

st.pyplot(fig)

# ---------- Future Prediction ----------
st.subheader("🔮 Predict Next Day Price")

last_10_days = scaled_data[-10:]
last_10_days = last_10_days.reshape(1, time_step, 1)

next_day_scaled = model.predict(last_10_days, verbose=0)
next_day_price = scaler.inverse_transform(next_day_scaled)

st.success(f"Predicted Next Day Closing Price: ${next_day_price[0][0]:.2f}")
