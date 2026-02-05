import cv2
import os
import numpy as np

def combine_images_with_mask(image_dir, mask_dir, output_dir):
    # 1. Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. Get all image files from the source directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in image_files:
        # Get the base filename without extension (e.g., 'image01')
        base_name = os.path.splitext(filename)[0]
        
        img_path = os.path.join(image_dir, filename)
        # Force looking for the corresponding .png file in the mask directory
        mask_path = os.path.join(mask_dir, base_name + ".png") 

        # Check if the corresponding mask exists
        if not os.path.exists(mask_path):
            print(f"Skipped: Mask file {base_name}.png not found.")
            continue

        # 3. Read the image and the mask
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) # Read mask in grayscale mode

        # Basic error handling for file reading
        if img is None or mask is None:
            print(f"Error: Could not read {filename} or its corresponding mask.")
            continue

        # 4. Dimension alignment: Ensure mask size matches image size
        if img.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

        # 5. Composite logic: Add the mask as the Alpha channel (Transparency)
        # Split the image into Blue, Green, and Red channels
        b, g, r = cv2.split(img)
        # Merge B, G, R channels with the mask as the 4th channel
        combined = cv2.merge([b, g, r, mask])

        # 6. Save the result (Must save as .png to keep alpha channel)
        output_path = os.path.join(output_dir, base_name + "_combined.png")
        cv2.imwrite(output_path, combined)
        print(f"Successfully processed and saved: {base_name}_combined.png")

# --- Path Configurations ---
image_folder = '/home/tec/Desktop/Project/Datasets/Matte/HRS10K/TE-HRS10K/im'
mask_folder = '/home/tec/Desktop/Project/TSR-Matte/results'
output_folder = '/home/tec/Desktop/Project/TSR-Matte/combines'

# Execute the function
if __name__ == "__main__":
    combine_images_with_mask(image_folder, mask_folder, output_folder)