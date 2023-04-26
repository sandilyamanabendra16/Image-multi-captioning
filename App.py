import io
import os
import streamlit as st
import requests
from PIL import Image
from model import generate_caption

st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Image Captioning Demo")

@st.cache(suppress_st_warning=True)
def load_image(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img

def predict():
    # Get the uploaded image from the user
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Save the uploaded file to disk
        img = Image.open(uploaded_file)
        img_path = 'tmp/img.jpg'
        img = img.convert('RGB')  # convert image to RGB mode
        img.save(img_path)

        # Generate and display the captions for the image
        num_captions = st.slider('Number of captions', 1, 4, 1)
        captions = generate_caption(img_path, num_captions=num_captions)
        for caption in captions:
            st.write(caption)


if __name__ == '__main__':
    predict()
