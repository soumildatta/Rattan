import os
from PIL import Image

# Define input and output directories
input_dir = "./data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/"
output_dir = "./data/cropped_coated"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            # Calculate crop margins
            width, height = img.size
            crop_margin_w, crop_margin_h = int(0.1 * width), int(0.1 * height)
            cropped_img = img.crop((crop_margin_w, crop_margin_h, width - crop_margin_w, height - crop_margin_h))
            
            # Save the cropped image
            output_path = os.path.join(output_dir, filename)
            cropped_img.save(output_path)
            print(f"Cropped and saved: {output_path}")

print("All images cropped and saved in 'cropped' directory.")
