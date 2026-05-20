import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Face Recognition App", layout="centered")

st.title("🧑 Face Recognition App")
st.write("Upload an image and the model will predict the person")

# Load model
@st.cache_resource
def load_my_model():
    return load_model("face_recognition.h5")

model = load_my_model()

# Class names (same order as training)
class_names = ['David', 'Sri', 'prajin']

# Upload image
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction[0])
    predicted_name = class_names[predicted_index]
    confidence = prediction[0][predicted_index] * 100

    st.success(f"### ✅ Prediction: **{predicted_name}**")
    st.write(f"Confidence: **{confidence:.2f}%**")
