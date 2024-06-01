
import gradio as gr
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model_path = "/content/brain_tumor_model.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Function to preprocess input image
def preprocess_image(image):
    image = Image.fromarray(image.astype('uint8'), 'RGB')
    image = image.resize((128, 128))  # Adjust the size as needed
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function for making predictions
def predict_tumor(image):
    preprocessed_img = preprocess_image(image)
    prediction = loaded_model.predict(preprocessed_img)[0]
    class_labels = ["glioma", "meningioma", "no tumor", "pituitary"]
    class_index = np.argmax(prediction)
    confidence = prediction[class_index]

    return f"Predicted Class: {class_labels[class_index]}, Confidence: {confidence:.4f}"

# Gradio Interface without specifying background image or custom HTML
iface = gr.Interface(
    fn=predict_tumor,
    inputs=gr.Image(),
    outputs=gr.Textbox(),
    live=True,
    title="Brain Tumor Detection"  # Set the title
)

iface.launch()
