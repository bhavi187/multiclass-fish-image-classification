import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

interpreter = tf.lite.Interpreter(model_path="small_fish_classifier_quant.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

classes = [
    'animal fish',
    'animal fish bass',
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

st.title("🐟 Fish Classifier with TFLite")

uploaded_file = st.file_uploader("Upload a fish image", type=["jpg", "jpeg", "png"])

def preprocess(img):
    img = img.resize((150, 150))
    img = np.array(img).astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess(img)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output_data)

    if pred_idx >= len(classes):
        st.error("Prediction index out of range.")
    else:
        confidence = output_data[0][pred_idx]
        pred_label = classes[pred_idx].replace('fish sea_food ', '').replace('_', ' ').title()
        st.write(f"**Prediction:** {pred_label}")
        st.write(f"**Confidence:** {confidence * 100:.2f}%")
