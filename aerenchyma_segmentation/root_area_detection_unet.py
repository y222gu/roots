import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models, transforms
import os

def segment_root(model, image, transform, option='highest_confidence', confidence_threshold=0.1):
    # Convert image to RGB and apply transformation
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed = transform(image_rgb)
    input_tensor = transformed.unsqueeze(0)

    # Perform segmentation with the pre-trained model
    with torch.no_grad():
        prediction = model(input_tensor)

    masks = prediction[0]['masks']
    scores = prediction[0]['scores']  # Confidence scores for each mask

    if option == 'largest_area':
        # Filter out masks based on the confidence threshold
        valid_masks = []
        for i in range(masks.shape[0]):
            if scores[i] > confidence_threshold:
                mask = masks[i, 0] > 0.5
                mask = mask.cpu().numpy().astype(np.uint8)
                valid_masks.append(mask)

        # If there are no valid masks above the threshold, return
        if not valid_masks:
            # return the mask with highest confidence
            best_mask_idx = torch.argmax(scores).item()
            best_mask = masks[best_mask_idx, 0] > 0.5
            best_mask = best_mask.cpu().numpy().astype(np.uint8)
            area = np.sum(best_mask)
            return best_mask, area, transformed

        # If there are valid masks then find the mask with the largest area
        best_mask = None
        largest_area = 0
        for mask in valid_masks:
            area = np.sum(mask)
            if area > largest_area:
                largest_area = area
                best_mask = mask
        return best_mask, largest_area, transformed

    elif option == 'highest_confidence':
        # Check if there are any masks in the prediction
        if masks.shape[0] == 0:
            return None, 0, transformed
        
        # Find the mask with the highest confidence score
        best_mask_idx = torch.argmax(scores).item()
        best_mask = masks[best_mask_idx, 0] > 0.5
        best_mask = best_mask.cpu().numpy().astype(np.uint8)
        area = np.sum(best_mask)
        return best_mask, area, transformed
    
    else:
        raise ValueError("Invalid option. Choose 'largest_area' or 'highest_confidence'.")

if __name__ == "__main__":

    # Load pre-trained Mask R-CNN model
    model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # Define a transformation to normalize the input image
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    image_folder = os.path.join(os.getcwd(),'aerenchyma_segmentation','data_for_segmentation', 'images', 'val_text')
    output_path = os.path.join(os.getcwd(),"aerenchyma_segmentation","data_for_segmentation","images", "val_text_root_segmented")
    os.makedirs(output_path, exist_ok=True)
    
    list_of_image_files = os.listdir(image_folder)

    for image_file in list_of_image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        best_mask, area = segment_root(model, image, transform, option='highest_confidence')

        # Visualize the overlay of the mask on the original image with red color opacity 0.5
        overlay = image.copy()
        overlay[best_mask == 1] = [255, 0, 255]
        segmented_image = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

        # Save the segmented imag        
        new_file_name = 'root_segmented_' + image_file
        save_path = os.path.join(output_path, new_file_name)

        # Save the segmented image
        cv2.imwrite(save_path, segmented_image)

        print(f"{image_file} root area: {area} pixels")