import json
import re
import os

def correct_filename(filename: str) -> str:
    # Extract image name and keep the name as A1, A2 ...
    sample_name = filename.split('_')[0]
    if "ROI" in sample_name:
        sample_name = sample_name.split("ROI")[0]

    # Determine the stain from the filename
    channel = "DAPI"

    # Prepare new filename and destination paths
    new_filename = f"{sample_name}_{channel}.tif"

    return new_filename

def remove_files(data, files_to_remove):
    # Remove images and annotations that match the files in files_to_remove
    updated_images = []
    updated_annotations = []
    image_ids_to_remove = set()
    
    # Collect image IDs to remove based on file names
    for image in data['images']:
        if image['file_name'] in files_to_remove:
            image_ids_to_remove.add(image['id'])
        else:
            updated_images.append(image)
    
    # Filter annotations by checking the image_id
    for annotation in data['annotations']:
        if annotation['image_id'] not in image_ids_to_remove:
            updated_annotations.append(annotation)
    
    # update the id_class to 0
    for annotation in updated_annotations:
        annotation['category_id'] = 0

    # Update the 'images' and 'annotations' lists with the filtered data
    data['images'] = updated_images
    data['annotations'] = updated_annotations

def update_annotations_json(json_path: str, output_json_path: str, files_to_remove):
    # Load the existing annotations JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Remove files that should be excluded
    remove_files(data, files_to_remove)
    
    # Update the 'file_name' in each image entry
    for image in data['images']:
        original_filename = image['file_name']
        corrected_filename = correct_filename(original_filename)
        image['file_name'] = corrected_filename
    
    # Save the updated JSON back to a new file
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated annotations saved to {output_json_path}")

# Specify the path to your original annotations JSON file
json_file_path =os.path.join(os.getcwd(),"aerenchyma_segmentation", "data", "annotations", "instances_default.json")
output_json_file_path = os.path.join(os.getcwd(),"aerenchyma_segmentation", "data", "annotations", "updated_annotation.json")

# List of file names to remove from the annotations
files_to_remove = [
    "E1ROI1_-1_3_1_Stitched[Read 2-Confocal DAPI 405,442]_001.tif",  # Replace with the actual file names to remove
    "E9ROI1_-1_3_1_Stitched[Read 2-Confocal DAPI 405,442]_001.tif",
    "E11ROI1_-1_3_1_Stitched[Read 2-Confocal DAPI 405,442]_001.tif"
]

# Run the script to update the file names and remove the specified files
update_annotations_json(json_file_path, output_json_file_path, files_to_remove)