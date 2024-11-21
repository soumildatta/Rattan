import os
from PIL import Image

input_dir = "./data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/"
output_dir = "./data/quantized_8bit_coated"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            quantized_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
            
            output_path = os.path.join(output_dir, filename)
            quantized_img.save(output_path)
            print(f"Quantized to 8-bit and saved: {output_path}")

print("All images quantized to 8-bit and saved in 'quantized_8bit' directory.")
