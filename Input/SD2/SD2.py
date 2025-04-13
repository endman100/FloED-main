import torch
import cv2
from PIL import Image
import numpy as np
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float32,
)
pipe.to("cuda")
prompt = ""
image = Image.open("path to your img").convert('RGB')
image = image.resize((512,512), Image.NEAREST)

m = Image.open( "path to your mask")

m = m.resize((512,512), Image.NEAREST)
m = np.array(m.convert('L'))
m = np.array(m > 0).astype(np.uint8)
m = cv2.dilate(m,
                cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)),
                iterations=5)
mask_image = Image.fromarray(m * 255) 

image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("SD2_inpainting.png")

