import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Brain Tumor Detection App")
st.write("Upload an MRI image to check whether a brain tumor is present.")

# Load trained model
@st.cache_resource
def load_cnn_model():
    return load_model("project.h5")  # model file name

model = load_cnn_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded MRI Image", use_container_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prediction
    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")
    st.write(f"Prediction Score: **{prediction:.6f}**")

    if prediction >= 0.5:
        st.error("⚠️ Brain Tumor Detected")
    else:
        st.success("✅ No Brain Tumor Detected")

