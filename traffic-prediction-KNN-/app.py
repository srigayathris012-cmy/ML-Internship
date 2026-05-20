import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# App Title
# -------------------------------
st.title("🚦 Traffic Signal Prediction using KNN")
st.write("Predict traffic signal state based on **Brightness** and **Saturation**")

# -------------------------------
# Generate Synthetic Dataset
# -------------------------------
np.random.seed(42)
n_samples_per_class = 50

# Red signal
red_brightness = np.random.normal(0.3, 0.1, n_samples_per_class)
red_saturation = np.random.normal(0.8, 0.1, n_samples_per_class)
red_features = np.column_stack((red_brightness, red_saturation))
red_labels = ['Red'] * n_samples_per_class

# Yellow signal
yellow_brightness = np.random.normal(0.7, 0.1, n_samples_per_class)
yellow_saturation = np.random.normal(0.6, 0.1, n_samples_per_class)
yellow_features = np.column_stack((yellow_brightness, yellow_saturation))
yellow_labels = ['Yellow'] * n_samples_per_class

# Green signal
green_brightness = np.random.normal(0.5, 0.1, n_samples_per_class)
green_saturation = np.random.normal(0.3, 0.1, n_samples_per_class)
green_features = np.column_stack((green_brightness, green_saturation))
green_labels = ['Green'] * n_samples_per_class

# Combine data
all_features = np.vstack((red_features, yellow_features, green_features))
all_labels = red_labels + yellow_labels + green_labels

df = pd.DataFrame(all_features, columns=["brightness", "saturation"])
df["signal_state"] = all_labels

# -------------------------------
# Train KNN Model
# -------------------------------
X = df[["brightness", "saturation"]]
y = df["signal_state"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, knn_model.predict(X_test))

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🔧 Input Parameters")

brightness = st.sidebar.slider(
    "Brightness", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)
saturation = st.sidebar.slider(
    "Saturation", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

# -------------------------------
# Prediction
# -------------------------------
if st.sidebar.button("Predict Traffic Signal"):
    input_data = np.array([[brightness, saturation]])
    prediction = knn_model.predict(input_data)[0]

    st.subheader("🔮 Prediction Result")
    if prediction == "Red":
        st.error("🔴 RED SIGNAL – STOP")
    elif prediction == "Yellow":
        st.warning("🟡 YELLOW SIGNAL – READY")
    else:
        st.success("🟢 GREEN SIGNAL – GO")

# -------------------------------
# Model Info
# -------------------------------
st.subheader("📊 Model Performance")
st.write(f"**Accuracy:** {accuracy:.2f}")

st.subheader("📁 Sample Dataset")
st.dataframe(df.head())
