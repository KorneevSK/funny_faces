import streamlit as st
import requests
from PIL import Image
from io import BytesIO

st.title("Googly Eyes App")
st.write("Upload an image to add googly eyes!")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the original image
    image = Image.open(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    if st.button("Add Googly Eyes"):
        # Send the image to the FastAPI backend
        with st.spinner("Processing..."):
            response = requests.post(
                "http://backend:8000/apply_googly_eyes/",
                files={"file": uploaded_file.getvalue()}
            )
            if response.status_code == 200:
                # Load the processed image from the response
                processed_image = Image.open(BytesIO(response.content))
                st.image(processed_image, caption="Image with Googly Eyes", use_column_width=True)
            else:
                st.error("Error processing the image.")
