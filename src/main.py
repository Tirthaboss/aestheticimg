import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import io  # Import io for handling byte streams

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
    with torch.autocast("cuda"):  # Ensure you have a CUDA-capable GPU
        output = pipe(image.unsqueeze(0), guidance_scale=7.5, num_inference_steps=50)  # Add batch dimension

    # Postprocess the output
    aesthetic_image = output.images[0]
    aesthetic_image = aesthetic_image.resize((512, 512))

    # Display the aesthetic image
    st.image(aesthetic_image, caption="Aesthetic Image")

    # Download the aesthetic image
    img_byte_arr = io.BytesIO()  # Create a byte stream
    aesthetic_image.save(img_byte_arr, format='PNG')  # Save the image to the byte stream
    img_byte_arr.seek(0)  # Move to the beginning of the byte stream

    st.download_button("Download Aesthetic Image", img_byte_arr, file_name="aesthetic_image.png", mime="image/png")
