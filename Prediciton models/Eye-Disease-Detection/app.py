# eye_disease/eye_disease_prediction.py
import os
import streamlit as st
import tensorflow as tf
import numpy as np
from pathlib import Path

# Define your own paths here
MODEL_PATH = 'Prediciton models/Eye-Disease-Detection/model.h5'  # Update this path
print("=========== OPEN CHECK =============")

def load_model():
    """Load the trained model."""
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict(model, input_image):
    """Make prediction on input image."""
    try:
        # Preprocessing
        input_image = tf.convert_to_tensor(input_image)
        input_image = tf.image.resize(input_image, [224, 224])
        input_image = tf.expand_dims(input_image, 0) / 255.0

        # Prediction
        predictions = model.predict(input_image)
        labels = ['Cataract', 'Conjunctivitis', 'Glaucoma', 'Normal']

        # Get confidence score for each class
        disease_confidence = {label: np.round(predictions[0][idx] * 100, 3) 
                            for idx, label in enumerate(labels)}

        # Get confidence percentage for the "Normal" class
        normal_confidence = disease_confidence['Normal']

        # Check if Normal confidence is greater than 50%
        if normal_confidence > 50:
            return "Normal", f"Congrats! No disease detected with confidence: {normal_confidence}%"

        # Identify the most likely disease
        detected_disease = max(disease_confidence, key=disease_confidence.get)
        confidence = disease_confidence[detected_disease]
        return detected_disease, f"{detected_disease}: {confidence}%"

    except Exception as e:
        return "Error", f"An error occurred: {e}"

def app():
    """Main application function."""
    st.subheader("👁️ Eye Disease Detection")
    st.write("This model identifies common eye diseases such as Cataract, "
             "Conjunctivitis, and Glaucoma. Upload an eye image to see how "
             "the model classifies its condition.")

    # Load the model
    print("======MODEL LOAD CHECK======")
    model = load_model()
    print("======MODEL LOADED ======")
    
    if model is None:
        st.error("Failed to load model. Please try again later.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

    if uploaded_file is not None:
        try:
            # Display the uploaded image
            image = tf.image.decode_image(uploaded_file.read(), channels=3)
            image_np = image.numpy()
            st.image(image_np, caption='Uploaded Image.', use_column_width=True)

            # Perform prediction
            disease, prediction = predict(model, image_np)
            st.write("Prediction : ")
            st.write(prediction)

            # Display 3D model if the detected disease is Cataract
            if disease == "Cataract":
                st.subheader("3D Visualization of Cataract")
                st.write("Explore this 3D model for a detailed understanding of Cataract:")
                cataract_model_iframe = """
                <div class="sketchfab-embed-wrapper"> 
                    <iframe title="eye_Cataracts" 
                            frameborder="0" 
                            allowfullscreen 
                            mozallowfullscreen="true" 
                            webkitallowfullscreen="true" 
                            allow="autoplay; fullscreen; xr-spatial-tracking" 
                            xr-spatial-tracking 
                            execution-while-out-of-viewport 
                            execution-while-not-rendered 
                            web-share 
                            src="https://sketchfab.com/models/1ea32d78279f421f8f93f360dcaf10a8/embed" 
                            style="width: 100%; height: 500px;">
                    </iframe> 
                    <p style="font-size: 13px; font-weight: normal; margin: 5px; color: #4A4A4A;"> 
                        <a href="https://sketchfab.com/3d-models/eye-cataracts-1ea32d78279f421f8f93f360dcaf10a8?utm_medium=embed&utm_campaign=share-popup&utm_content=1ea32d78279f421f8f93f360dcaf10a8" 
                           target="_blank" 
                           rel="nofollow" 
                           style="font-weight: bold; color: #1CAAD9;">
                           eye_Cataracts
                        </a> 
                        by 
                        <a href="https://sketchfab.com/StrangeBeastDesign?utm_medium=embed&utm_campaign=share-popup&utm_content=1ea32d78279f421f8f93f360dcaf10a8" 
                           target="_blank" 
                           rel="nofollow" 
                           style="font-weight: bold; color: #1CAAD9;">
                           Strange Beast Design
                        </a> 
                        on 
                        <a href="https://sketchfab.com?utm_medium=embed&utm_campaign=share-popup&utm_content=1ea32d78279f421f8f93f360dcaf10a8" 
                           target="_blank" 
                           rel="nofollow" 
                           style="font-weight: bold; color: #1CAAD9;">
                           Sketchfab
                        </a>
                    </p>
                </div>
                """
                st.components.v1.html(cataract_model_iframe, height=500)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
