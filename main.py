import streamlit as st
import os
import models
import tensorflow as tf
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
from tempfile import NamedTemporaryFile

# Function to preprocess the input image
def preprocess_image(uploaded_file):
    try:
        # Open the uploaded image
        img = Image.open(uploaded_file)
        img_array = img_to_array(img)

        # Normalize the image to 0-1 range
        img_array = img_array / 255.0

        # Check if the model requires 64 channels
        if img_array.shape[2] == 3:
            img_array = np.repeat(img_array, 64, axis=-1)
            st.write(f"Preprocessed image shape: {img_array.shape}")  # Debugging output

        # Save the preprocessed image temporarily
        with NamedTemporaryFile(delete=False, suffix='.bmp') as temp_file:
            temp_img_path = temp_file.name
            img.save(temp_img_path)

        return temp_img_path
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

# Function to construct the enhanced image filename
def construct_filename(path, suffix, scale_factor):
    base, ext = os.path.splitext(path)  # Split the path into base name and extension
    filename = f"{base}_{suffix}({scale_factor}x){ext}"
    return filename

# Streamlit App
st.title("🖼️ Super-Resolution Image Enhancer")

# Upload file
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is None:
    st.warning("Please upload an image to enhance.")
else:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    if st.button("Enhance Image 🚀"):
        try:
            with st.spinner('Enhancing... please wait...'):
                # Preprocess the image
                temp_img_path = preprocess_image(uploaded_file)

                if not temp_img_path:
                    st.error("Image preprocessing failed. Please try again.")
                    exit(0)  # Exit gracefully if preprocessing fails

                # Load model and perform upscale
                with tf.device('/CPU:0'):
                    model = models.SuperResolutionModel(scale_factor=2)  # Default scale factor 2x
                    model.upscale(temp_img_path, save_intermediate=True, mode="patch", patch_size=8, suffix="enhanced")

                # Construct the enhanced image filename
                enhanced_image_path = construct_filename(temp_img_path, suffix="enhanced", scale_factor=2)
                st.write(f"Enhanced image path: {enhanced_image_path}")  # Debugging output

                # After enhancing, check if the new image exists
                if os.path.exists(enhanced_image_path):
                    enhanced_img = Image.open(enhanced_image_path)
                    st.write(f"Enhanced image type: {type(enhanced_img)}")  # Debugging output
                    st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)

                    # Allow download of the enhanced image
                    with open(enhanced_image_path, "rb") as file:
                        st.download_button(label="Download Enhanced Image", data=file, file_name="enhanced_image.bmp", mime="image/bmp")
                    
                    # Cleanup
                    os.remove(enhanced_image_path)
                else:
                    st.error(f"Enhanced image not found. Check model output or logs for issues.")
                    st.warning("The enhancement process failed. Please try again.")

                # Remove temp image
                os.remove(temp_img_path)

            st.success("Enhancement Complete ✅")
        except Exception as e:
            st.error(f"Error during image enhancement: {e}")
            st.warning("The enhancement process encountered an error. Please try again.")

