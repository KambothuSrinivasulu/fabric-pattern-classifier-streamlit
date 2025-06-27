# Fabric-pattern-classifier-streamlit
"Deep learning model deployed as a Streamlit app for identifying different fabric patterns."
Description
This project is a web application that uses a deep learning model to classify different types of fabric patterns. Built with Streamlit, it provides an easy-to-use interface for users to upload an image of a fabric and receive a prediction of its pattern type.

Live Demo
You can access the live application here:

Access the Fabric Pattern Classifier App on Streamlit Cloud

Features
Upload fabric images in JPG, JPEG, or PNG format.
Get instant predictions of the fabric pattern.
Simple and intuitive user interface.
Dataset
The model was trained on a dataset containing images of various fabric patterns.

Number of Classes: 10
Pattern Categories: [List your 10 specific class names here, e.g., "Checks", "Floral", "Stripes", etc.]
( add a note aboutOptional: You can data distribution if you wish, e.g., "Note: The dataset is primarily composed of images of female dress patterns, which may impact performance on other types of fabric.")
Model
Architecture: A Convolutional Neural Network (CNN) based on the ResNet architecture.
Framework: Developed using TensorFlow and Keras.
Training: The model was trained on the fabric pattern dataset to classify images into the 10 defined categories.
Technical Details
Libraries Used: Streamlit, TensorFlow, NumPy, Pillow, JSON.
Deployment Platform: Deployed on Streamlit Cloud.
How to Run Locally (Optional)
If you want to run this application on your local machine:

Clone this repository:
 cd fabric-pattern-classifier-streamlit
  pip install -r requirements.txt
  streamlit run app.py
