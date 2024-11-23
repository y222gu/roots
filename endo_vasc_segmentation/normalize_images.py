import os
from PIL import Image
import numpy as np

def normalize_brightness(img_array, target_mean):
    # Convert to float for calculations
    img_float = img_array.astype(float)
    
    # Calculate the current mean brightness
    current_mean = np.mean(img_float)
    
    # Calculate scaling factor
    scale_factor = target_mean / current_mean if current_mean > 0 else 1
    
    # Apply scaling
    img_normalized = img_float * scale_factor
    
    # Clip values to valid range and convert back to 16-bit
    img_normalized = np.clip(img_normalized, 0, 65535).astype(np.uint16)
    
    return img_normalized

def calculate_mean_brightness(folder):
    """Calculate the mean brightness of all images in a folder."""
    image_extensions = ('.tif', '.png', '.jpg', '.jpeg')
    images = [img for img in os.listdir(folder) if img.lower().endswith(image_extensions)]
    total_brightness = 0
    total_pixels = 0

    for image_name in images:
        img_path = os.path.join(folder, image_name)
        img = Image.open(img_path)
        img_array = np.array(img).astype(float)
        total_brightness += np.sum(img_array)
        total_pixels += img_array.size

    return total_brightness / total_pixels if total_pixels > 0 else 0

def normalize_images(input_folder):
    reference_folder = r"C:\Users\Root Project\Pictures\test3\pics norm"
    dapi_folder = os.path.join(input_folder, "DAPI_cropped")
    output_folder =  os.path.join(input_folder, "DAPI_cropped_normalized")


    """Normalize images from folder2 using the average brightness of folder1 and save the normalized images."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Calculate the average brightness of folder1
    mean_folder1 = calculate_mean_brightness(reference_folder)

    # Process and save normalized images from folder2
    image_extensions = ('.tif', '.png', '.jpg', '.jpeg')
    images = [img for img in os.listdir(dapi_folder) if img.lower().endswith(image_extensions)]

    for image_name in images:
        img_path = os.path.join(dapi_folder, image_name)
        img = Image.open(img_path)
        img_array = np.array(img)
        
        normalized_img_array = normalize_brightness(img_array, mean_folder1)
        normalized_img = Image.fromarray(normalized_img_array)
        
        output_path = os.path.join(output_folder, image_name)
        normalized_img.save(output_path)
        print(f"Processed and saved: {output_path}")

if __name__ == "__main__":

    input_folder = r"C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed"  # Second input folder, the images in this folder will be normalized
    normalize_images(input_folder)