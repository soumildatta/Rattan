import os
from PIL import Image, ImageEnhance

input_dir = "./data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/"
output_dir = "./data/brightened_coated"

os.makedirs(output_dir, exist_ok=True)

brightness_factor = 1.5

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            enhancer = ImageEnhance.Brightness(img)
            brightened_img = enhancer.enhance(brightness_factor)
            
            output_path = os.path.join(output_dir, filename)
            brightened_img.save(output_path)
            print(f"Brightened and saved: {output_path}")

print("All images brightened and saved in 'brightened' directory.")
