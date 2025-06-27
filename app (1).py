
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json # Import json

# Load the saved model
# Ensure the path matches where you saved the model
model_path = 'deployed_model.h5' # Or 'best_resnet_model.h5'
if os.path.exists('deployed_model.h5'):
    model_path = 'deployed_model.h5'
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
class_names = [] # Initialize class_names as an empty list
if os.path.exists(class_names_path):
    try:
        with open(class_names_path, 'r') as f:
            class_names = json.load(f)
        # print(f"Loaded class names: {class_names}") # Optional: for debugging
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
        # Open the image from the uploaded file object
        img = Image.open(uploaded_file)

        # Resize the image
        img = img.resize(target_size)

        # Convert to numpy array and normalize
        img_array = np.array(img) / 255.0

        # Expand dimensions to include batch size
        img_array = np.expand_dims(img_array, axis=0) # shape: (1, height, width, channels)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)

        return predicted_class_index, None # Return index and no error
    except Exception as e:
        return None, f"Error processing image or making prediction: {e}"


# Set the title of the Streamlit application
st.title("Fabric Pattern Classifier")

# Add a file uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Process the uploaded file
if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Add a button to trigger prediction
    if st.button('Predict'):
        if deployed_model and class_names:
            # Call the modified prediction function
            predicted_index, error = predict_image_class(uploaded_file, deployed_model)

            if error:
                st.error(f"Prediction error: {error}")
            elif predicted_index is not None and predicted_index < len(class_names):
                # Get the predicted class name using the loaded class_names list
                predicted_class_name = class_names[predicted_index]
                st.write(f"Predicted class: **{predicted_class_name}**")
            else:
                 if predicted_index is not None:
                     st.warning(f"Prediction index {predicted_index} is out of bounds for class names.")
                 else:
                     st.warning("Prediction failed.")
        else:
            st.error("Model or class names not loaded. Cannot make predictions.")
