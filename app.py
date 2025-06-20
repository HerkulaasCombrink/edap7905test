import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title='SASL Letter A Classifier', layout='centered')
st.title('🤟 SASL Letter A Image Classifier')
st.markdown("""
Upload a grayscale image of a signed letter. The model will predict whether it is the letter **A** or **Not A**.
""")

# Load model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('letter_a_model_from_images.h5')
    return model

model = load_model()

# Image preprocessing
def preprocess_image(image, size=(64, 64)):
    img = np.array(image.convert('L'))  # Convert to grayscale
    img = cv2.resize(img, size)
    img = img.astype('float32') / 255.0
    img = img.reshape(1, size[0], size[1], 1)
    return img

# Upload interface
uploaded_file = st.file_uploader("Upload a hand sign image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess and predict
    input_data = preprocess_image(image)
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    label_map = {0: "Not A", 1: "A"}
    confidence = prediction[0][predicted_class]

    st.subheader(f"Prediction: {label_map[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")
