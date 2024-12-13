import os
import cv2
import numpy as np
from aerenchyma_segmentation.root_area_detection_unet import segment_root
from ultralytics import YOLO


if __name__ == "__main__":
    # Load the trained YOLOv8 model
    model = YOLO(r'C:\Users\Yifei\Documents\roots\aerenchyma_segmentation\runs\yolov8_segmentation3\weights\best.pt')  # Replace with your trained model path

    image_folder = os.path.join(os.getcwd(),'aerenchyma_segmentation','data_for_segmentation', 'images', 'val_text')
    output_path = os.path.join(os.getcwd(),"aerenchyma_segmentation","data_for_segmentation","images", "val_text_predictions")
    os.makedirs(output_path, exist_ok=True)
    
    list_of_image_files = os.listdir(image_folder)

    for image_file in list_of_image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        root_mask, root_area = segment_root(image, option='highest_confidence')


        # Run inference
        results = model.predict(image, imgsz=1024, conf=0.2, task='segment', verbose=False)
        
        # Extract segmentation masks
        masks = results[0].masks  # List of masks (one per detected object)
        
        if masks is not None:
            for mask in masks.data:  # Iterate over each mask
                mask = mask.cpu().numpy()  # Convert to NumPy array
                mask = (mask > 0.5).astype(np.uint8)  # Threshold to binary mask
                
                # Resize mask to match image dimensions
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LANCZOS4)
                
                # Colorize mask and overlay on the image
                colored_mask = np.zeros_like(image, dtype=np.uint8)
                colored_mask[mask_resized == 1] = mask_color
                image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)  # Blend original image and mask

        # Save the annotated image
        save_path = os.path.join(output_path, f"pred_{image_file}")
        cv2.imwrite(save_path, image)
































        # Visualize the overlay of the mask on the original image with red color opacity 0.5
        overlay = image.copy()
        overlay[root_mask == 1] = [255, 0, 255]
        segmented_image = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)

        # Save the segmented imag        
        new_file_name = 'segmented_' + image_file
        save_path = os.path.join(image_folder, new_file_name)

        # Save the segmented image
        cv2.imwrite(save_path, segmented_image)

        print(f"{image_file} root area: {root_area} pixels")