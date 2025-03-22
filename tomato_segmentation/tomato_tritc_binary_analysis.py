import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_and_normalize_images(folder_path, output_folder):
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".png", ".jpeg", ".jpg", ".tif", ".tiff")):
            file_base_name = filename.split(".")[0]
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                normalized_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
                images.append((filename, normalized_img))
                plt.imsave(os.path.join(output_folder, file_base_name + "_normalized.png"), normalized_img, cmap='gray')

    # plot the total histogram of the pixel intensity of all images
    pixel_intensity = np.concatenate([img[1].flatten() for img in images])
    plt.figure()
    plt.hist(pixel_intensity.flatten(), bins=1000, edgecolor='black', alpha=0.7)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Histogram (8-bit)')
    plt.grid(True, alpha=0.3)
    plt.show()
    return images

# function to turn images into a binary mask based on a threshold
def binarize_images(images, threshold, output_folder):
    binary_images = []
    for filename, img in images:
        file_base_name = filename.split(".")[0]
        _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
        binary_images.append((filename, binary_img))
        plt.imsave(os.path.join(output_folder, file_base_name + "_binary.png"), binary_img, cmap='gray')

    return binary_images

if __name__ == "__main__":
    folder_path = r'C:\Users\Yifei\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\Kevin_Cropped_Images\All_Folders_Compiled_for_test_processed\TRITC_results_endo'
    output_folder = r'C:\Users\Yifei\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\Kevin_Cropped_Images\All_Folders_Compiled_for_test_processed\TRITC_endo_binary_analysis'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    images = load_and_normalize_images(folder_path, output_folder)
    binary_images = binarize_images(images, 60, output_folder)