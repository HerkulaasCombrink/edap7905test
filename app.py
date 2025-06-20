import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(page_title='SASL Letter A Classifier', layout='centered')
st.title('🤟 SASL Letter A Classifier')
st.markdown("""
Upload a grayscale image of a signed letter. The app will try to classify it as **A** or **Not A**.
""")

# Define path to model
MODEL_PATH = 'letter_a_model_from_images.h5'

# Load the model safely
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning(f"Model file not found: `{MODEL_PATH}`. Please upload it to the app directory.")
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# Image preprocessing function
def preprocess_image(image, target_size=(64, 64)):
    try:
        img = image.convert('L')  # Convert to grayscale
        img = img.resize(target_size)
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, target_size[0], target_size[1], 1)
        return img_array
    except Exception as e:
        st.error(f"Image preprocessing failed: {e}")
        return None

# File uploader
uploaded_file = st.file_uploader("Upload a grayscale image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)
    if input_data is not None and model:
        try:
            prediction = model.predict(input_data)
            predicted_class = np.argmax(prediction)
            confidence = float(prediction[0][predicted_class])
            label = "A" if predicted_class == 1 else "Not A"

            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {confidence:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    elif model is None:
        st.info("Model is not loaded. Please upload the model file to run predictions.")
