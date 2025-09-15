import gradio as gr
from keras_cv.models import StableDiffusion
from PIL import Image

model = StableDiffusion(img_height=512, img_width=512)

def generate_image(prompt):
    images = model.text_to_image([prompt])
    img_array = images[0].numpy().astype("uint8")
    pil_img = Image.fromarray(img_array)
    return pil_img

iface = gr.Interface(
    fn=generate_image,
    inputs=gr.Textbox(lines=2, placeholder="Enter prompt..."),
    outputs=gr.Image(type="pil"),  # Return PIL Image for better compatibility
    title="Stable Diffusion Text-to-Image",
    description="Generate images and download as PNG or JPG"
)

iface.launch()
