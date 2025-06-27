import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json

# Load the saved model
# Ensure the path matches where you saved the model
model_path = 'deployed_model.h5' # Or 'best_resnet_model.h5'
if os.path.exists(model_path):
    pass
elif os.path.exists('best_resnet_model.h5'):
    model_path = 'best_resnet_model.h5'
else:
    st.error("Model file not found!")
    st.stop()


@st.cache_resource # Cache the model loading
def load_my_model(path):
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

deployed_model = load_my_model(model_path)

# Load class names from the saved JSON file
class_names_path = 'class_names.json'
class_names = []
if os.path.exists(class_names_path):
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
    except Exception as e:
        st.error(f"Error loading class names from {class_names_path}: {e}")
else:
    st.error(f"Class names file not found at {class_names_path}!")
    st.stop()


# Define a function to preprocess an uploaded image and make a prediction
def predict_image_class(uploaded_file, model, target_size=(224, 224)):
    if model is None:
        return None, "Model is not loaded."

    try:
        img = Image.open(uploaded_file)
        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0) # shape: (1, height, width, channels)

        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)

        return predicted_class_index, None
    except Exception as e:
        return None, f"Error processing image or making prediction: {e}"


# --- Streamlit UI Enhancements ---

st.set_page_config(page_title="Fabric Pattern Classifier", layout="wide")

st.title("👕🧵 Fabric Pattern Classifier")

st.markdown(f"""
    Welcome to the Fabric Pattern Classifier!
    Upload an image of a fabric, and the model will predict its pattern.
    Currently supports the following patterns: **{", ".join(class_names)}**.
""")

st.sidebar.header("About the App")
st.sidebar.info("This application uses a pre-trained deep learning model to classify fabric patterns.")
st.sidebar.info("Upload an image on the left to see the prediction!")

col1, col2 = st.columns(2)

with col1:
    st.header("Upload Your Image")
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.success("File uploaded successfully!")
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            with col2:
                st.header("Prediction Result")
                if deployed_model and class_names:
                    predicted_index, error = predict_image_class(uploaded_file, deployed_model)

                    if error:
                        st.error(f"Prediction error: {error}")
                    elif predicted_index is not None and predicted_index < len(class_names):
                        predicted_class_name = class_names[predicted_index]
                        st.write(f"Predicted class: **{predicted_class_name}**")
                        st.balloons()
                    else:
                         if predicted_index is not None:
                             st.warning(f"Prediction index {predicted_index} is out of bounds for class names.")
                         else:
                             st.warning("Prediction failed.")
                else:
                    st.error("Model or class names not loaded. Cannot make predictions.")
    else:
        with col2:
             st.info("Upload an image to see the prediction result here.")


st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow")
