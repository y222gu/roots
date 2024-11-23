import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv

def plot_pixel_intensity_histogram(image_path, output_folder, csv_writer, x_min=None, x_max=None):
    """Plots histogram of pixel intensities, saves it, and writes stats to CSV."""
    image = np.array(Image.open(image_path))  # Load image as a numpy array
    if image.ndim > 2:
        raise ValueError("This script is designed for grayscale images only.")

    non_zero_pixels = image[image > 0]  # Filter zero pixels
    avg_brightness, std_deviation = np.mean(non_zero_pixels), np.std(non_zero_pixels)

    # Set histogram range to ±3 std deviations if not specified
    x_min = x_min or max(0, avg_brightness - 3 * std_deviation)
    x_max = x_max or min(65535, avg_brightness + 3 * std_deviation)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(non_zero_pixels, bins=1000, range=(x_min, x_max), edgecolor='black', alpha=0.7)
    plt.axvline(avg_brightness, color='red', linestyle='--', label=f'Average = {avg_brightness:.2f}')
    plt.axvline(avg_brightness + std_deviation, color='green', linestyle='--', label=f'Std Dev = {std_deviation:.2f}')
    plt.axvline(avg_brightness - std_deviation, color='green', linestyle='--')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.title('Pixel Intensity Histogram (16-bit)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(x_min, x_max)
    plt.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
    
    # Save histogram
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Write stats to CSV
    csv_writer.writerow([base_name, f"{avg_brightness:.2f}", f"{std_deviation:.2f}"])

def get_mean_intensity(input_folder, x_min=0, x_max=60000):
    """Processes a folder of images, saving histograms and recording statistics."""

    channels = ["GFP", "TRITC"]
    structures = ["endo", "vasc"]

    for channel in channels:
        for structure in structures:
            channel_folder = os.path.join(input_folder, f"{channel}_results_{structure}")
            if not os.path.exists(channel_folder):
                print(f"Folder {channel_folder} not found, skipping...")
                continue

            output_folder = os.path.join(input_folder, f"{channel}_results_{structure}_histograms")
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Create CSV for statistics
            csv_file_path = os.path.join(input_folder, f"{channel}_{structure}_statistics.xlsx")
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["Image Name", "Average Pixel Intensity", "Standard Deviation"])

                # Process each image file in input folder
                for image_file in filter(lambda f: f.lower().endswith(('.tif', '.tiff')), os.listdir(channel_folder)):
                    plot_pixel_intensity_histogram(os.path.join(channel_folder, image_file), output_folder, csv_writer, x_min, x_max)

            print(f"CSV file created at {csv_file_path}")
    

# Usage example
if __name__ == "__main__":
    input_folder = r'C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed'
    get_mean_intensity(input_folder)