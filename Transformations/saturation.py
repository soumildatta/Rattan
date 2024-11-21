import os
from PIL import Image, ImageEnhance

# Define the input and output directories
input_dir = "./data/traindata_p1.0_wanet_unconditional_s2.0_k128_removeeval/train/"  # Replace with the path to your image directory
output_dir = "./data/saturated_coated"

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Saturation factor
saturation_factor = 2.0

# Loop through each file in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # Adjust extensions if necessary
        # Open the image
        img_path = os.path.join(input_dir, filename)
        with Image.open(img_path) as img:
            # Enhance saturation
            enhancer = ImageEnhance.Color(img)
            saturated_img = enhancer.enhance(saturation_factor)
            
            # Save the saturated image to the output directory
            output_path = os.path.join(output_dir, filename)
            saturated_img.save(output_path)
            print(f"Processed and saved: {output_path}")

print("All images processed and saved in 'saturated_coated' directory.")
