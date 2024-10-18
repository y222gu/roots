import os
import json
from PIL import Image

# last updated Aug 16, 2024 by Lucas DeMello (Brady Lab)
# my email is lucasxbox288@gmail.com if you have questions
# this code takes a COCO annotation file with bounding box coordinates, and crops 
# images based on that. The cropped images are then saved in the specified folder. 
# Note that this program assumes the images are titled "{set identifier}_{image name}_{channel}.tif", for example, "STEFAN_A1_DAPI.tif"
# If you want to crop another channel with the same bounding box, put that in 'additional_image_folder', for example, if you wanted to crop the GFP channel
# The program will compare the set identifier and image name to find the associated images and annotations

# All inputs are here, no need to scroll through the code

coco_file = r'C:\Users\Root Project\Documents\yolov2\runs\detect\val10\pred.json'
main_image_folder = r'C:\Users\Root Project\Documents\yolov2\dataset\val\images'
output_folder = r'C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_1_preprocessed\DAPI_cropped'
additional_image_folder = r'C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_1_preprocessed\GFP'  # Set to 'None' if not used
additional_output_folder = r'C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_1_preprocessed\GFP_cropped'

###################################################################
# Conversion code below
###################################################################

def extract_image_info(filename):
    parts = filename.split('_')
    # if len(parts) >= 3:
    #     return '_'.join(parts[:-1])  # Set identifier and image name
    # return None
    return parts[0]

def crop_images(coco_file, main_image_folder, output_folder, additional_image_folder=None):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(additional_output_folder, exist_ok= True)

    # Load COCO annotations
    with open(coco_file, 'r') as f:
        coco_data = json.load(f)

    # Create a mapping of image filenames to their full paths
    main_image_map = {f: os.path.join(main_image_folder, f) for f in os.listdir(main_image_folder) if f.lower().endswith(('.tif', '.tiff'))}
    
    if additional_image_folder:
        additional_image_map = {f: os.path.join(additional_image_folder, f) for f in os.listdir(additional_image_folder) if f.lower().endswith(('.tif', '.tiff'))}
    
    # Process each annotation
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        x_min, y_min, width, height = map(int, bbox)
        x_max = x_min + width
        y_max = y_min + height

        # Find the corresponding image
        image_info = next((img for img in coco_data['images'] if img['id'] == image_id), None)
        if image_info is None:
            print(f"No image info found for image_id: {image_id}")
            continue
        
        image_filename = image_info['file_name']
        image_key = extract_image_info(image_filename)

        if image_key:
            # Crop main image
            if image_filename in main_image_map:
                crop_and_save(main_image_map[image_filename], output_folder, x_min, y_min, x_max, y_max)

            # Crop additional image if provided
            if additional_image_folder:
                additional_files = [f for f in additional_image_map if extract_image_info(f) == image_key]
                for add_file in additional_files:
                    crop_and_save(additional_image_map[add_file], additional_output_folder, x_min, y_min, x_max, y_max)
        else:
            print(f"Could not process image: {image_filename}")

def crop_and_save(image_path, output_folder, x_min, y_min, x_max, y_max):
    try:
        with Image.open(image_path) as img:
            cropped_img = img.crop((x_min, y_min, x_max, y_max))
            base_name = os.path.basename(image_path)
            name_parts = os.path.splitext(base_name)
            output_filename = f"{name_parts[0]}_cropped{name_parts[1]}"
            output_path = os.path.join(output_folder, output_filename)
            cropped_img.save(output_path)
            print(f"Cropped and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

if __name__ == "__main__":
    crop_images(coco_file, main_image_folder, output_folder, additional_image_folder)