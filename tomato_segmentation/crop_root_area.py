import os
import cv2
import torch
import numpy as np
from torchvision import transforms, models
import glob

def segment_root(model, image, transform, option='highest_confidence', confidence_threshold=0.1):
    """
    Segment the root from the input image using the given model and transformation,
    and return the binary mask, its area, and the bounding box of the mask.
    """
    # Convert image from BGR to RGB and apply transformation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(image_rgb).unsqueeze(0)

    # Perform segmentation using the model
    with torch.no_grad():
        prediction = model(input_tensor)

    masks = prediction[0]['masks']
    scores = prediction[0]['scores']  # Confidence scores for each mask

    def get_bbox(mask):
        # Find contours in the mask and get the bounding rectangle of the largest contour.
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            return (x, y, w, h)
        return None

    if option == 'largest_area':
        valid_masks = []
        for i in range(masks.shape[0]):
            if scores[i] > confidence_threshold:
                mask = masks[i, 0] > 0.5
                mask = mask.cpu().numpy().astype(np.uint8)
                valid_masks.append(mask)

        if not valid_masks:
            best_mask_idx = torch.argmax(scores).item()
            best_mask = masks[best_mask_idx, 0] > 0.5
            best_mask = best_mask.cpu().numpy().astype(np.uint8)
            area = np.sum(best_mask)
            bbox = get_bbox(best_mask)
            return best_mask, area, bbox

        best_mask = None
        largest_area = 0
        for mask in valid_masks:
            area = np.sum(mask)
            if area > largest_area:
                largest_area = area
                best_mask = mask
        bbox = get_bbox(best_mask)
        return best_mask, largest_area, bbox

    elif option == 'highest_confidence':
        best_mask_idx = torch.argmax(scores).item()
        best_mask = masks[best_mask_idx, 0] > 0.5
        best_mask = best_mask.cpu().numpy().astype(np.uint8)
        area = np.sum(best_mask)
        bbox = get_bbox(best_mask)
        return best_mask, area, bbox

    else:
        raise ValueError("Invalid option. Choose 'largest_area' or 'highest_confidence'.")

def update_annotations_for_crop(annotations, crop_bbox, orig_img_width, orig_img_height):
    """
    Update YOLO-format polygon annotations after cropping an image.

    Parameters:
        annotations (list of str): Each annotation is a string in the format:
                                   "<class_id> x1 y1 x2 y2 ... xN yN"
                                   with coordinates normalized to the original image dimensions.
        crop_bbox (tuple): Crop bounding box in the form (crop_x, crop_y, crop_w, crop_h),
                           where (crop_x, crop_y) are the top-left pixel coordinates of the crop.
        orig_img_width (int): Width of the original image.
        orig_img_height (int): Height of the original image.

    Returns:
        list of str: Updated annotations in YOLO format with normalized coordinates relative to the cropped image.
    """
    updated_annotations = []
    crop_x, crop_y, crop_w, crop_h = crop_bbox

    for ann in annotations:
        parts = ann.split()
        class_id = parts[0]
        norm_coords = list(map(float, parts[1:]))

        # Convert normalized coordinates to absolute pixel values.
        abs_coords = []
        for i, coord in enumerate(norm_coords):
            if i % 2 == 0:  # x-coordinate
                abs_coord = coord * orig_img_width
            else:           # y-coordinate
                abs_coord = coord * orig_img_height
            abs_coords.append(abs_coord)

        # Adjust coordinates based on crop offset.
        cropped_abs_coords = []
        for i, coord in enumerate(abs_coords):
            if i % 2 == 0:  # x-coordinate
                new_coord = coord - crop_x
            else:           # y-coordinate
                new_coord = coord - crop_y
            cropped_abs_coords.append(new_coord)

        # Re-normalize coordinates relative to cropped image dimensions.
        updated_norm_coords = []
        for i, coord in enumerate(cropped_abs_coords):
            if i % 2 == 0:  # x-coordinate
                norm_coord = coord / crop_w
            else:           # y-coordinate
                norm_coord = coord / crop_h
            updated_norm_coords.append(norm_coord)

        new_ann = " ".join([class_id] + [f"{coord:.6f}" for coord in updated_norm_coords])
        updated_annotations.append(new_ann)

    return updated_annotations

# -------------------------------------------------------------------
# Helper functions to load your model and transformation.
def load_model():
    # Replace with your actual model loading code.
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def get_transform():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform

# -------------------------------------------------------------------
# The root segmentation is done on the DAPI channel and the detected bbox is applied to all channels.
def process_images_and_annotations(image_folder, annotation_folder, output_image_folder, output_annotation_folder):
    # Define channel subfolders.
    channels = ["DAPI", "GFP", "TRITC"]
    
    # Create output directories for images and annotations.
    os.makedirs(output_annotation_folder, exist_ok=True)
    for ch in channels:
        os.makedirs(os.path.join(output_image_folder, ch), exist_ok=True)
    
    model = load_model()
    transform = get_transform()
    
    # Process DAPI images only to obtain the bbox.
    dapi_folder = os.path.join(image_folder, "DAPI")
    dapi_image_paths = glob.glob(os.path.join(dapi_folder, "*_DAPI.tif"))
    
    for dapi_path in dapi_image_paths:
        base_name = os.path.splitext(os.path.basename(dapi_path))[0]
        # Expected filename format: <identifier>_DAPI.tif
        identifier = base_name.replace("_DAPI", "")
        
        annotation_path = os.path.join(annotation_folder, identifier + '_DAPI.txt')
        
        # Read DAPI image.
        image_dapi = cv2.imread(dapi_path)
        if image_dapi is None:
            print(f"Failed to read DAPI image: {dapi_path}")
            continue
        orig_h, orig_w = image_dapi.shape[:2]
        
        # # Read annotations (if available).
        # annotations = []
        # if os.path.exists(annotation_path):
        #     with open(annotation_path, 'r') as f:
        #         annotations = f.read().strip().splitlines()
        
        # Get segmentation on the DAPI image.
        mask, area, bbox = segment_root(model, image_dapi, transform, option='highest_confidence', confidence_threshold=0.1)
        if bbox is None:
            print(f"No bounding box found for DAPI image: {dapi_path}")
            continue
        x, y, w, h = bbox
        
        # Crop and save images for each channel using the same bbox.
        for ch in channels:
            channel_folder = os.path.join(image_folder, ch)
            channel_filename = f"{identifier}_{ch}.tif"
            channel_image_path = os.path.join(channel_folder, channel_filename)
            if not os.path.exists(channel_image_path):
                print(f"Channel image not found: {channel_image_path}")
                continue
            image_channel = cv2.imread(channel_image_path)
            if image_channel is None:
                print(f"Failed to read image: {channel_image_path}")
                continue
            
            # Crop using the bbox computed from the DAPI channel.
            cropped_channel = image_channel[y:y+h, x:x+w]
            output_channel_folder = os.path.join(output_image_folder, ch)
            output_channel_path = os.path.join(output_channel_folder, channel_filename)
            cv2.imwrite(output_channel_path, cropped_channel)
        
        # # Update the annotations relative to the crop.
        # updated_annotations = update_annotations_for_crop(annotations, bbox, orig_w, orig_h)
        # output_annotation_path = os.path.join(output_annotation_folder, identifier + '_DAPI.txt')
        # with open(output_annotation_path, 'w') as f:
        #     f.write("\n".join(updated_annotations))
        
        print(f"Processed sample: {identifier}")

# -------------------------------------------------------------------
# Example usage:
if __name__ == "__main__":
    image_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\Folder4_processed'          # Folder containing your original images.
    annotation_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\annotation_folder1'     # Folder containing YOLO annotation text files.
    output_image_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\Folder4_cropped' # Folder where cropped images will be saved.
    output_annotation_folder = r'C:\Users\Yifei\Documents\roots\tomato_segmentation\data\annotation_folder4_cropped'  # Folder for updated annotations.
    
    process_images_and_annotations(image_folder, annotation_folder, output_image_folder, output_annotation_folder)