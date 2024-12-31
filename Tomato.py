import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TFLite model
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="tomato_classifier_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize image to match model input size
    image = np.array(image) / 255.0  # Normalize pixel values to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image.astype(np.float32)  # Ensure float32 data type for TFLite

# Perform inference with the TFLite model
def predict_with_tflite(interpreter, image):
    # Get input and output tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the input tensor with preprocessed image
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output predictions
    predictions = interpreter.get_tensor(output_details[0]['index'])
    return predictions

# Define class names
class_names = ["Reject", "Ripe", "Unripe"]

# Streamlit UI
st.title("Tomato Classifier")
st.write("Classify tomatoes as Reject, Ripe, or Unripe.")

# Option to upload an image or take a picture
option = st.radio("Choose an input method:", ["Upload an image", "Take a picture"])

uploaded_file = None

if option == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

elif option == "Take a picture":
    uploaded_file = st.camera_input("Take a picture")

if uploaded_file is not None:
    # Display the uploaded or captured image
    image = Image.open(uploaded_file)
    st.image(image, caption="Input Image", width=300)
    st.write("Classifying...")

    # Preprocess the image
    processed_image = preprocess_image(image)

    # Load TFLite model
    interpreter = load_tflite_model()

    # Predict with the TFLite model
    predictions = predict_with_tflite(interpreter, processed_image)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display results
    st.write(f"Prediction: **{predicted_class}**")
    st.write(f"Confidence: **{confidence * 100:.2f}%**")
