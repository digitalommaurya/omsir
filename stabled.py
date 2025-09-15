from keras_cv.models import StableDiffusion

# Load the Stable Diffusion model
model = StableDiffusion(img_height=512, img_width=512)

# Generate an image from text
images = model.text_to_image("a cyberpunk city at night", batch_size=1)

# Visualize the result
import matplotlib.pyplot as plt
plt.imshow(images[0])
plt.axis("off")
plt.show()
