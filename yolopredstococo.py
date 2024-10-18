import json
import os
from datetime import datetime

# last updated Aug 16, 2024 by Lucas DeMello (Brady Lab)
# my email is lucasxbox288@gmail.com if you have questions
# this code takes a .json file produced by the YOLOv8 prediction code and turns it into
# a usable standard COCO annotation format.

# All inputs are here, no need to scroll through the code

input_file_path = r'C:\Users\Root Project\Documents\yolov2\runs\detect\val10\predictions.json'  # Replace with the path to your prediction results
output_file_path = r'C:\Users\Root Project\Documents\yolov2\runs\detect\val10\pred.json'   # Replace with the desired path for the output file
file_extension = ".tif"
user_class_name = "endodermis" 

###################################################################
# Conversion code below
###################################################################

# Load the prediction results
with open(input_file_path, 'r') as f:
    predictions = json.load(f)

# Dictionary to keep track of the best prediction for each image
best_predictions = {}

for prediction in predictions:
    image_id = prediction["image_id"]
    if image_id not in best_predictions or prediction["score"] > best_predictions[image_id]["score"]:
        best_predictions[image_id] = prediction

# Initialize the COCO formatted dictionary
coco_format = {
    "licenses": [{
        "name": "",
        "id": 0,
        "url": ""
    }],
    "info": {
        "contributor": "",
        "date_created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "description": "",
        "url": "",
        "version": "",
        "year": datetime.now().year
    },
    "categories": [{
        "id": 1,  # Assuming single category starts from 1
        "name": user_class_name,  # Using the user-specified class name
        "supercategory": ""
    }],
    "images": [],
    "annotations": []
}

# Helper function to generate unique annotation IDs
def get_annotation_id():
    if not hasattr(get_annotation_id, "counter"):
        get_annotation_id.counter = 0  # it doesn't exist yet, so initialize it
    get_annotation_id.counter += 1
    return get_annotation_id.counter

# Dictionary to map image IDs to COCO image IDs
image_id_map = {}

def add_image(image_id):
    if image_id not in image_id_map:
        coco_image_id = len(image_id_map) + 1
        image_id_map[image_id] = coco_image_id
        coco_format["images"].append({
            "id": coco_image_id,
            "file_name": f"{image_id}{file_extension}",  # Assuming the image file name follows this format
            "width": 0,  # Replace with actual width if known
            "height": 0,  # Replace with actual height if known
            "license": 0,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": 0
        })

# Convert each best prediction to a COCO format annotation
for image_id, prediction in best_predictions.items():
    add_image(image_id)
    
    bbox = prediction["bbox"]
    x, y, width, height = bbox
    
    # COCO bbox format: [x_min, y_min, width, height]
    coco_annotation = {
        "id": get_annotation_id(),
        "image_id": image_id_map[image_id],
        "category_id": 1,  # Since we have only one category
        "segmentation": [],  # Add empty segmentation list
        "area": width * height,
        "bbox": [x, y, width, height],
        "iscrowd": 0,
        "attributes": {  # Add attributes field if needed
            "occluded": False,
            "rotation": 0.0
        }
    }
    
    coco_format["annotations"].append(coco_annotation)

# Write the COCO formatted annotations to a file
with open(output_file_path, 'w') as f:
    json.dump(coco_format, f, indent=4)

print(f"COCO formatted annotations saved to {output_file_path}")
