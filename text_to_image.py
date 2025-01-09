import torch
from diffusers import StableDiffusionPipeline
import os

def generate_image(
    prompt,
    output_dir="generated_images",
    model_path="CompVis/stable-diffusion-v1-4",
    num_inference_steps=100,  # More steps for better quality
    guidance_scale=8.5,       # Fine-tuned for better adherence to prompt
    height=768,               # Higher resolution
    width=768,
    negative_prompt="blurry, low quality, bad anatomy, low resolution"
):
    # Determine the available device
    if torch.backends.mps.is_available():  # For Apple Silicon or Metal on macOS
        device = "mps"
    elif hasattr(torch, "has_mps") and torch.backends.mps.is_built():  # For DirectML (AMD GPUs on Windows)
        device = "dml"
    elif torch.cuda.is_available():  # For NVIDIA GPUs with CUDA
        device = "cuda"
    else:
        device = "cpu"  # Fallback to CPU

    # Load Stable Diffusion pipeline
    print(f"Using device: {device}")
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )
    pipeline.to(device)

    # Generate the image
    print(f"Generating image for prompt: '{prompt}'...")
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
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    prompt = input("Enter a text prompt for the image: ")
    generate_image(prompt)
