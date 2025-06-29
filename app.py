# app.py

import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = "mask_model_small.keras"
IMG_SIZE = (128, 128)

# Load model
model = load_model(MODEL_PATH)

# Streamlit Page Config
st.set_page_config(page_title="Face Mask Detector", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #4CAF50;'>😷 Face Mask Detector</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Upload an image to check whether the person is wearing a mask.</p>",
    unsafe_allow_html=True
)

# Prediction function
def predict(img: Image.Image):
    img = img.resize(IMG_SIZE)
    img_array = np.asarray(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    label = "😷 Mask" if pred > 0.5 else "🚫 No Mask"
    confidence = max(pred, 1 - pred)
    return label, confidence

# Upload UI
uploaded_file = st.file_uploader("📁 Upload a face image", type=["jpg", "jpeg", "png"])

# Predict if image is uploaded
if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="🖼 Uploaded Image", use_column_width=True)
    label, confidence = predict(img)

    st.markdown(f"<h3 style='text-align: center;'>{label}</h3>", unsafe_allow_html=True)
    st.progress(int(confidence * 100))
    st.info(f"Confidence: {confidence:.2%}")


else:
    st.warning("Please upload a face image (jpg/png).")
