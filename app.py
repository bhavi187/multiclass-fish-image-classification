import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

model = load_model("fish_classifier.h5")

classes = [
    'fish sea_food black_sea_sprat',
    'fish sea_food gilt_head_bream',
    'fish sea_food hourse_mackerel',
    'fish sea_food red_mullet',
    'fish sea_food red_sea_bream',
    'fish sea_food sea_bass',
    'fish sea_food shrimp',
    'fish sea_food striped_red_mullet',
    'fish sea_food trout'
]

st.title("🐟 Fish Classifier")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

def preprocess(img):
    img = img.resize((150, 150))
    img = image.img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_arr = preprocess(img)
    preds = model.predict(input_arr)
    pred_idx = np.argmax(preds)
    confidence = preds[0][pred_idx]

    st.write(f"**Prediction:** {classes[pred_idx].replace('fish sea_food ', '').replace('_', ' ').title()}")
    st.write(f"**Confidence:** {confidence * 100:.2f}%")
