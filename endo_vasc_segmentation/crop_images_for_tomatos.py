import os
import cv2
import json

# Function to crop images based on YOLO format annotations
def crop_images_with_yolo_annotations(input_image_folder, input_annotation_folder):

    # Dictionary to store annotations
    annotations = {}

    # Set up input and output folders
    DAPI_input_folder = os.path.join(input_image_folder, "DAPI")
    DAPI_output_folder = os.path.join(input_image_folder, "DAPI_cropped")
    os.makedirs(DAPI_output_folder, exist_ok=True)

    # Create cropped output folders for GFP and TRITC
    GFP_input_folder = os.path.join(input_image_folder, "GFP")
    GFP_output_folder = os.path.join(input_image_folder, "GFP_cropped")
    os.makedirs(GFP_output_folder, exist_ok=True)

    TRITC_input_folder = os.path.join(input_image_folder, "TRITC")
    TRITC_output_folder = os.path.join(input_image_folder, "TRITC_cropped")
    os.makedirs(TRITC_output_folder, exist_ok=True)

    # Loop over each DAPI image
    for filename in os.listdir(DAPI_input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            dapi_image_path = os.path.join(DAPI_input_folder, filename)
            annotation_path = os.path.join(input_annotation_folder, os.path.splitext(filename)[0] + '.txt')
            
            # Check if the annotation file exists
            if not os.path.exists(annotation_path):
                continue

            # Load the DAPI image
            dapi_image = cv2.imread(dapi_image_path, cv2.IMREAD_UNCHANGED)
            height, width = dapi_image.shape[:2]

            # List to store bounding box info for this DAPI image
            bounding_boxes = []

            # Read the annotation file
            with open(annotation_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.strip().split())
                    x_center, y_center, bbox_width, bbox_height = x_center * width, y_center * height, bbox_width * width, bbox_height * height
                    x1 = int(x_center - bbox_width / 2)
                    y1 = int(y_center - bbox_height / 2)
                    x2 = int(x_center + bbox_width / 2)
                    y2 = int(y_center + bbox_height / 2)
                    
                    # Save annotation
                    bounding_boxes.append({
                        "label": int(class_id),
                        "bbox": [x1, y1, x2, y2]
                    })

                    # Crop the DAPI image using the bounding box
                    cropped_dapi = dapi_image[y1:y2, x1:x2]
                    cropped_dapi_filename = f"{os.path.splitext(filename)[0]}_cropped.tif"
                    cv2.imwrite(os.path.join(DAPI_output_folder, cropped_dapi_filename), cropped_dapi, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

            # Add annotations for the DAPI image
            annotations[filename] = bounding_boxes

            # # Load GFP and TRITC images if they exist
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
                        cropped_channel_filename = f"{os.path.splitext(channel_filename)[0]}_cropped_{label}.tif"
                        cv2.imwrite(os.path.join(output_folder, cropped_channel_filename), cropped_channel, [cv2.IMWRITE_TIFF_COMPRESSION, 1])

            # for Kevin's images
            # find corresponding GFP and TRITC images
            # base_name = filename.split("_")[0] + "_" + filename.split("_")[1]
            # # loop through files and find the file with the same base name in gfp and tritc folders
            # for channel_folder, output_folder in [(GFP_input_folder, GFP_output_folder), (TRITC_input_folder, TRITC_output_folder)]:
            #     for channel_filename in os.listdir(channel_folder):
            #         if base_name in channel_filename:
            #             channel_image_path = os.path.join(channel_folder, channel_filename)
            #             # Load the channel image, do not convert to grayscale
            #             channel_image = cv2.imread(channel_image_path, cv2.IMREAD_UNCHANGED)

            #             # Crop the GFP/TRITC image using the same bounding boxes
            #             for box_info in bounding_boxes:
            #                 x1, y1, x2, y2 = box_info["bbox"]
            #                 label = box_info["label"]

            #                 # Crop the channel image and save it
            #                 cropped_channel = channel_image[y1:y2, x1:x2]
            #                 cropped_channel_filename = f"{os.path.splitext(channel_filename)[0]}_cropped.tif"
            #                 cv2.imwrite(os.path.join(output_folder, cropped_channel_filename), cropped_channel, [cv2.IMWRITE_TIFF_COMPRESSION, 1])
            #             print('find the file')                                                                                     


    # Save all annotations to a JSON file
    # annotations_file = os.path.join(input_folder, "cropping_annotations.json")
    # with open(annotations_file, 'w') as f:
    #     json.dump(annotations, f, indent=4)

    print("Bounding box annotations and cropped images saved successfully.")

# Run the function
if __name__ == '__main__':
    input_image_folder = r'C:\Users\Yifei\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\Kevin_Cropped_Images\All_Folders_Compiled_for_test_processed'
    input_annotation_folder = r'C:\Users\Yifei\Box\Carney Lab Shared\Data\C10\Dustin\ROOTS-Images from C10\Kevin_Cropped_Images\Cropped_Tomato_Endodermis1_YOLO\obj_Annotation1_data'
    crop_images_with_yolo_annotations(input_image_folder, input_annotation_folder)
