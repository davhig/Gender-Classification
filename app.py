import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the pre-trained model
model = load_model('cnn_model.h5')

# Function to preprocess the uploaded image
def preprocess_image(image):
    # Resize image to the required input shape of the model
    image = image.resize((150, 150))
    # Convert image to grayscale
    image = image.convert('L')
    # Convert image to array and normalize pixel values
    image = np.array(image) / 255.0
    # Add batch dimension and reshape to match model input shape
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    return image

# Function to predict gender from the image
def predict_gender(image):
    # Preprocess the image
    processed_image = preprocess_image(image)
    # Predict gender
    gender = model.predict(processed_image)
    return gender

# Streamlit UI
def main():
    st.title("Gender Classification App")
        
    # Add an image to your Streamlit app
    image = st.image('gender_classification.webp')

    st.write("""
    ## About
    **Welcome to the Gender Classification App! This app allows you to classify whether the Gender is Male or Female**

    The notebook, model and documentation are available on [GitHub.](https://github.com/dars180602/Gender-Classification/)        

    **Contributors:** 
    - **Cecille Jatulan**
    - **David Higuera**
    - **Diana Reyes**
    - **Mike Montanez**
    - **Maria Melencio**         
    """)

    st.write("Upload an image and we will estimate the gender!")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform gender estimation on the uploaded image
        gender = predict_gender(image)
        if gender <= 0.5:
            st.write("Predicted gender: Female")
        else:
            st.write("Predicted gender: Male")

if __name__ == "__main__":
    main()
