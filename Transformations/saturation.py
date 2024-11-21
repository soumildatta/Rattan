import os
from PIL import Image, ImageEnhance

input_dir = "./data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/"
output_dir = "./data/saturated_coated"

os.makedirs(output_dir, exist_ok=True)

saturation_factor = 2.0

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            enhancer = ImageEnhance.Color(img)
            saturated_img = enhancer.enhance(saturation_factor)
            
            output_path = os.path.join(output_dir, filename)
            saturated_img.save(output_path)
            print(f"Processed and saved: {output_path}")

print("All images processed and saved in 'saturated_coated' directory.")
