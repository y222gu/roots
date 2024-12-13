import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def adaptive_normalization(image_np):
    """
    Normalize 16-bit image using adaptive min-max scaling.
    """
    min_val = np.min(image_np)
    max_val = np.max(image_np)
    normalized_image = (image_np - min_val) / (max_val - min_val)  # Scale to [0, 1]
    return normalized_image


def clip_and_normalize(image_np, low_percentile=0.1, high_percentile=99.9):
    """
    Clip intensity range based on percentiles and normalize to [0, 1].
    """
    low, high = np.percentile(image_np, (low_percentile, high_percentile))
    clipped_image = np.clip(image_np, low, high)
    normalized_image = (clipped_image - low) / (high - low)  # Scale to [0,1]
    return normalized_image


def apply_clahe(image_np):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast.
    """
    # CLAHE expects 8-bit input, so scale and convert
    scaled_image = (image_np / 65535.0 * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(scaled_image)
    # Scale back to [0, 1]
    enhanced_image = enhanced_image / 255.0
    return enhanced_image


def log_scaling(image_np):
    """
    Apply logarithmic scaling to compress high dynamic range.
    """
    image_np = image_np.astype(np.float32) + 1  # Avoid log(0)
    log_image = np.log(image_np)
    log_image = (log_image - np.min(log_image)) / (np.max(log_image) - np.min(log_image))  # Normalize to [0,1]
    return log_image


def preprocess_image(image_path, method='adaptive'):
    """
    Preprocess a single 16-bit image with a selected method.
    """
    # Load image
    image = Image.open(image_path).convert('I')  # Load as 16-bit grayscale
    image_np = np.array(image).astype(np.float32)

    # Apply chosen preprocessing method
    if method == 'adaptive':
        processed_image = adaptive_normalization(image_np)
    elif method == 'clip':
        processed_image = clip_and_normalize(image_np)
    elif method == 'clahe':
        processed_image = apply_clahe(image_np)
    elif method == 'log':
        processed_image = log_scaling(image_np)
    else:
        raise ValueError("Unsupported preprocessing method")

    # Scale to 8-bit range [0, 255] for saving
    processed_image_8bit = (processed_image * 255).astype(np.uint8)

    return processed_image_8bit


def preprocess_dataset(input_dir, output_dir, method='adaptive'):
    """
    Preprocess all 16-bit images in a directory and save the preprocessed versions as 8-bit images.
    """
    if output_dir is None:
        output_dir = input_dir + '_preprocessed'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all images in the input directory
    for subdir, dirs, files in os.walk(input_dir):
        for file in tqdm(files):
            if file.lower().endswith(('.png', '.tiff', '.tif')):  # Common formats for 16-bit images
                img_path = os.path.join(subdir, file)

                # Preprocess the image
                processed_image = preprocess_image(img_path, method=method)

                # Save the preprocessed image
                output_subdir = subdir.replace(input_dir, output_dir)  # Maintain subdirectory structure
                if not os.path.exists(output_subdir):
                    os.makedirs(output_subdir)

                output_path = os.path.join(output_subdir, file)
                # if the image already exists, overwrite it
                cv2.imwrite(output_path, processed_image)

    print(f"Preprocessing complete. Processed images saved in {output_dir}")


if __name__ == '__main__':
    input_dir = r"C:\Users\Yifei\Documents\roots\aerenchyma_segmentation\data_for_segmentation\images\val_"  # Input directory of images

    # Choose preprocessing method: 'adaptive', 'clip', 'clahe', 'log'
    method = 'adaptive'  # Change this to experiment with different methods

    # Preprocess all images
    preprocess_dataset(input_dir, method=method)
