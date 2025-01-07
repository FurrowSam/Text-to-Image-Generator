import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
import os

# Function to generate an image
def generate_image(prompt, output_dir="generated_images", model_path="CompVis/stable-diffusion-v1-4",
                   num_inference_steps=100, guidance_scale=8.5, height=768, width=768, 
                   negative_prompt="blurry, low quality, bad anatomy, low resolution"):
    # Determine the available device
    if torch.backends.mps.is_available():  # For Apple Silicon or Metal on macOS
        device = "mps"
    elif hasattr(torch, "has_mps") and torch.has_mps:  # For DirectML (AMD GPUs on Windows)
        device = "dml"
    elif torch.cuda.is_available():  # For NVIDIA GPUs with CUDA
        device = "cuda"
    else:
        device = "cpu"  # Fallback to CPU

    # Load Stable Diffusion pipeline
    st.write(f"Using device: {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    pipeline.to(device)

    # Generate the image
    st.write(f"Generating image for prompt: '{prompt}'...")
    image = pipeline(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        negative_prompt=negative_prompt,
        height=height,
        width=width
    ).images[0]

    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prompt.replace(' ', '_')}.png")
    image.save(output_path)
    return output_path, image

# Streamlit App
st.title("🖼️ Text-to-Image Generator")
st.markdown("Generate images from text prompts using Stable Diffusion.")

# Input fields
prompt = st.text_input("Enter your text prompt:", "A futuristic city with flying cars at sunset")
num_inference_steps = st.slider("Number of Inference Steps:", 10, 150, 50)
guidance_scale = st.slider("Guidance Scale:", 1.0, 15.0, 7.5)
height = st.slider("Image Height (pixels):", 512, 1024, 768, step=64)
width = st.slider("Image Width (pixels):", 512, 1024, 768, step=64)

# Generate button
if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        output_path, image = generate_image(prompt, num_inference_steps=num_inference_steps, 
                                            guidance_scale=guidance_scale, height=height, width=width)
    st.image(image, caption=f"Generated Image: {prompt}", use_column_width=True)
    st.success(f"Image saved to {output_path}")
