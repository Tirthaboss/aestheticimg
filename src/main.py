import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np

# Initialize the model
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)

# Streamlit app
st.title("Normal to Aesthetic Image Converter")

# Upload image
image_file = st.file_uploader("Upload a normal image", type=["jpg", "png"])

if image_file:
    # Load the normal image
    image = Image.open(image_file)

    # Preprocess the image
    image = image.resize((512, 512))
    image = np.array(image)
    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

    # Generate the aesthetic image
    with torch.autocast("cuda"):
        output = pipe(image.unsqueeze(0), guidance_scale=7.5, num_inference_steps=50)  # Add unsqueeze(0) to add batch dimension

    # Postprocess the output
    aesthetic_image = output.images[0]
    aesthetic_image = aesthetic_image.resize((512, 512))

    # Display the aesthetic image
    st.image(aesthetic_image, caption="Aesthetic Image")

    # Download the aesthetic image
    # Convert the PIL image to bytes for download
    img_byte_arr = io.BytesIO()
    aesthetic_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    st.download_button("Download Aesthetic Image", img_byte_arr, file_name="aesthetic_image.png", mime="image/png")
