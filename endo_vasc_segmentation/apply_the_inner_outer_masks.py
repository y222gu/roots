# this file assumes masks and images have naming convention "[set identifier] [name] whatever.[extension]"

import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def apply_mask(image, outer_mask, inner_mask, option):
    # Convert the image to a numpy array 
    image = np.array(image, dtype=np.float32)
    
    # Ensure masks are compatible with the image dimensions
    outer_mask_array = np.array(outer_mask, dtype=np.float32)
    inner_mask_array = np.array(inner_mask, dtype=np.float32)

    # Resize masks if they are not the same size as the image
    if outer_mask_array.shape != image.shape[:2]:
        print(outer_mask_array.shape)
        print(image.shape)

        outer_mask_array = np.resize(outer_mask_array, image.shape[:2])
        print("Resized outer mask to match image dimensions.")
    if inner_mask_array.shape != image.shape[:2]:
        print(inner_mask_array.shape)
        print(image.shape)
        inner_mask_array = np.resize(inner_mask_array, image.shape[:2])
        print("Resized inner mask to match image dimensions.")
    
    if option == "endo":
        # Calculate the combined mask
        combined_mask = outer_mask_array - inner_mask_array
    elif option == "vasc":
        # Calculate the combined mask
        combined_mask = inner_mask_array
    
    # Make sure the combined_mask has the same number of channels as the image
    if len(image.shape) == 3 and image.shape[2] == 3:  # RGB image
        combined_mask = np.stack([combined_mask] * 3, axis=-1)  # Repeat for each channel
    
    # Apply mask subtraction logic
    masked_img_array = np.where(combined_mask, image, 0)  # Keep original pixel values where mask is true
    
    # Ensure the result is within the valid range
    masked_img_array = np.clip(masked_img_array, 0, 65535)
    
    # Convert back to PIL Image
    masked_img = Image.fromarray(masked_img_array.astype(np.uint16))
    return masked_img

def process_images(input_image_folder, outer_mask_folder, inner_mask_folder, output_folder, option):
    """Process images with corresponding outer and inner masks."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_extensions = ('.tif', '.png', '.jpg', '.jpeg')
    images = [img for img in os.listdir(input_image_folder) if img.lower().endswith(image_extensions)]

    for image_name in images:
        img_path = os.path.join(input_image_folder, image_name)
        # Extract set identifier and name
        base_name = "_".join(os.path.splitext(image_name)[0].split("_")[:1])
        # add "_" to the end of the base name
        base_name = base_name + "_"
        # Find corresponding outer and inner mask files
        outer_mask_path = None
        inner_mask_path = None
        
        # Searching for mask files
        for file in os.listdir(outer_mask_folder):
            if file.startswith(base_name) and file.endswith('.npy'):
                outer_mask_path = os.path.join(outer_mask_folder, file)
                break
        
        for file in os.listdir(inner_mask_folder):
            if file.startswith(base_name) and file.endswith('.npy'):
                inner_mask_path = os.path.join(inner_mask_folder, file)
                break
        
        # Only process if both masks are found
        if outer_mask_path and inner_mask_path:
            # Load image with 16-bit pixel values
            img = Image.open(img_path).convert("I;16")

            # Load masks from npy files
            outer_mask = np.load(outer_mask_path)
            inner_mask = np.load(inner_mask_path)
            
            print(f"Processing {image_name}")
            # Apply masks
            masked_img = apply_mask(img, outer_mask, inner_mask, option = option)

            # Save the result
            output_path = os.path.join(output_folder, image_name)
            masked_img = masked_img.resize(masked_img.size, Image.LANCZOS)
            masked_img.save(output_path, "TIFF", compression="tiff_lzw", resolution_unit=2, resolution=(300, 300))

        else:
            print(f"Outer or inner mask not found for {image_name}, skipping.")
        

def get_masked_images(input_folder):
    """Process images with corresponding outer and inner masks."""
    outer_mask_folder = os.path.join(input_folder, "outer_masks")
    inner_mask_folder = os.path.join(input_folder, "inner_masks")
    GFP_folder = os.path.join(input_folder, "GFP_cropped")
    TRITC_folder = os.path.join(input_folder, "TRITC_cropped")
    
    GFP_results_endo = os.path.join(input_folder, "GFP_results_endo")
    TRITC_results_endo = os.path.join(input_folder, "TRITC_results_endo")

    GFP_results_vasc = os.path.join(input_folder, "GFP_results_vasc")
    TRITC_results_vasc = os.path.join(input_folder, "TRITC_results_vasc")

    if not os.path.exists(GFP_results_endo):
        os.makedirs(GFP_results_endo)

    if not os.path.exists(TRITC_results_endo):
        os.makedirs(TRITC_results_endo)

    if not os.path.exists(GFP_results_vasc):
        os.makedirs(GFP_results_vasc)

    if not os.path.exists(TRITC_results_vasc):
        os.makedirs(TRITC_results_vasc)

    process_images(GFP_folder, outer_mask_folder, inner_mask_folder, GFP_results_endo, option = "endo")
    process_images(TRITC_folder, outer_mask_folder, inner_mask_folder, TRITC_results_endo, option = "endo")

    process_images(GFP_folder, outer_mask_folder, inner_mask_folder, GFP_results_vasc, option = "vasc")
    process_images(TRITC_folder, outer_mask_folder, inner_mask_folder, TRITC_results_vasc, option = "vasc")


if __name__ == "__main__":
    human = False
    input_image_folder = r'C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed'
    get_masked_images(input_image_folder)
    print("All images processed!")