import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np

# Define the folder containing the images
image_folder = os.path.join(os.getcwd(),'aerenchyma_segmentation','data_for_segmentation', 'images', 'val_normalized')

# Get all image paths in the folder
image_paths = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.tif')]

# Initialize lists to store results
root_areas = []
masks = []

# Process each image
for image_path in image_paths:

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Thresholding to create a binary image, where the root area becomes white (255) and background black (0).
    _, binary_image = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY_INV)

    # Apply morphological operations to fill in gaps within the root
    kernel = np.ones((10, 10), np.uint8)
    morphed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Find contours of the segmented root
    contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Assuming the largest contour is the root, get its area
    root_contour = max(contours, key=cv2.contourArea)
    root_area = cv2.contourArea(root_contour)

    # Plot the segmented image and the area
    segmented_image = np.zeros_like(image)
    cv2.drawContours(segmented_image, [root_contour], -1, (255), thickness=cv2.FILLED)

    # Display the result
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.title("Segmented Root")
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Root Area: {:.2f} pixels".format(root_area))
    plt.imshow(image, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    cv2.imwrite(os.path.join(image_folder, 'mask_' + os.path.basename(image_path)), morphed_image)


