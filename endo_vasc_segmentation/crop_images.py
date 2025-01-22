import os
from ultralytics import YOLO
import cv2
import json

# Function to crop images based on YOLO predictions from DAPI channel
def crop_images_with_trained_YOLO(input_folder):

    # Most recent trained model path in the weights folder
    model_path = os.path.join(os.path.dirname(__file__), "weights", "YOLO.pt")
    model = YOLO(model_path)

    # Dictionary to store annotations
    annotations = {}

    # Set up input and output folders
    DAPI_input_folder = os.path.join(input_folder, "DAPI")
    DAPI_output_folder = os.path.join(input_folder, "DAPI_cropped")
    os.makedirs(DAPI_output_folder, exist_ok=True)

    # Create cropped output folders for GFP and TRITC
    GFP_input_folder = os.path.join(input_folder, "GFP")
    GFP_output_folder = os.path.join(input_folder, "GFP_cropped")
    os.makedirs(GFP_output_folder, exist_ok=True)

    TRITC_input_folder = os.path.join(input_folder, "TRITC")
    TRITC_output_folder = os.path.join(input_folder, "TRITC_cropped")
    os.makedirs(TRITC_output_folder, exist_ok=True)

    # Loop over each DAPI image
    for filename in os.listdir(DAPI_input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            dapi_image_path = os.path.join(DAPI_input_folder, filename)
            
            # Run prediction on the DAPI image
            results = model(dapi_image_path)
            
            # Load the DAPI image
            dapi_image = cv2.imread(dapi_image_path, cv2.IMREAD_UNCHANGED)

            # List to store bounding box info for this DAPI image
            bounding_boxes = []

            # Process each detected bounding box
            for box, conf, cls in zip(results[0].boxes.xyxy, results[0].boxes.conf, results[0].boxes.cls):
                # Extract bounding box coordinates and convert to integers
                x1, y1, x2, y2 = map(int, box)
                label = model.names[int(cls)]
                
                # Save annotation
                bounding_boxes.append({
                    "label": label,
                    "confidence": float(conf),
                    "bbox": [x1, y1, x2, y2]
                })

                # Crop the DAPI image using the bounding box
                cropped_dapi = dapi_image[y1:y2, x1:x2]
                cropped_dapi_filename = f"{os.path.splitext(filename)[0]}_cropped.tif"
                cv2.imwrite(os.path.join(DAPI_output_folder, cropped_dapi_filename), cropped_dapi, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

            # Add annotations for the DAPI image
            annotations[filename] = bounding_boxes

            # Load GFP and TRITC images if they exist
            for channel_name, channel_folder, output_folder in [("GFP", GFP_input_folder, GFP_output_folder), ("TRITC", TRITC_input_folder, TRITC_output_folder)]:
                channel_filename = filename.replace("DAPI", channel_name)
                channel_image_path = os.path.join(channel_folder, channel_filename)
                
                if os.path.exists(channel_image_path):
                    # Load the channel image, do not convert to grayscale
                    channel_image = cv2.imread(channel_image_path, cv2.IMREAD_UNCHANGED)

                    # Crop the GFP/TRITC image using the same bounding boxes
                    for box_info in bounding_boxes:
                        x1, y1, x2, y2 = box_info["bbox"]
                        label = box_info["label"]

                        # Crop the channel image and save it
                        cropped_channel = channel_image[y1:y2, x1:x2]
                        cropped_channel_filename = f"{os.path.splitext(channel_filename)[0]}_cropped.tif"
                        cv2.imwrite(os.path.join(output_folder, cropped_channel_filename), cropped_channel, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

    # Save all annotations to a JSON file
    annotations_file = os.path.join(input_folder, "cropping_annotations.json")
    with open(annotations_file, 'w') as f:
        json.dump(annotations, f, indent=4)

    print("Bounding box annotations and cropped images saved successfully.")

# Run the function
if __name__ == '__main__':
    input_folder = r'C:\Users\Root Project\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\20240715-19_TAMERA_PLATES_1-3\20240715-19_automated_3-plates\plate_2_processed'
    crop_images_with_trained_YOLO(input_folder)
