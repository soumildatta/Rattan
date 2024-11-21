import os
from PIL import Image

input_dir = "./data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/"
output_dir = "./data/hue_green_coated"

os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            hsv_img = img.convert("HSV")
            data = hsv_img.getdata()
            
            green_hue_data = [(90, s, v) for h, s, v in data]  # Set hue to 90 (green)
            hsv_img.putdata(green_hue_data)
            
            green_img = hsv_img.convert("RGB")
            output_path = os.path.join(output_dir, filename)
            green_img.save(output_path)
            print(f"Hue changed to green and saved: {output_path}")

print("All images hue-changed to green and saved in 'hue_green' directory.")
