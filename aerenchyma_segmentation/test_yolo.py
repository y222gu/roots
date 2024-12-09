from ultralytics import YOLO
import cv2
import os
import numpy as np

if __name__ == '__main__':
    # Load the trained YOLOv8 model
    model = YOLO(r'C:\Users\Yifei\Documents\roots\aerenchyma_segmentation\runs\yolov8_segmentation3\weights\best.pt')  # Replace with your trained model path

    # Path to validation images
    val_images_path = os.path.join(os.getcwd(), "aerenchyma_segmentation","data_for_segmentation","images", "val")
    output_path = os.path.join(os.getcwd(),"aerenchyma_segmentation","data_for_segmentation","images", "val_predictions")
    os.makedirs(output_path, exist_ok=True)

    mask_color = (0, 255, 0)  # Green

    # Loop through validation images
    for image_file in os.listdir(val_images_path):
        image_path = os.path.join(val_images_path, image_file)
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load {image_path}")
            continue
        
        # Run inference
        results = model.predict(image, imgsz=1024, conf=0.8, task='segment', verbose=False)
        
        # Extract segmentation masks
        masks = results[0].masks  # List of masks (one per detected object)
        
        if masks is not None:
            for mask in masks.data:  # Iterate over each mask
                mask = mask.cpu().numpy()  # Convert to NumPy array
                mask = (mask > 0.5).astype(np.uint8)  # Threshold to binary mask
                
                # Resize mask to match image dimensions
                mask_resized = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
                
                # Colorize mask and overlay on the image
                colored_mask = np.zeros_like(image, dtype=np.uint8)
                colored_mask[mask_resized == 1] = mask_color
                image = cv2.addWeighted(image, 1, colored_mask, 0.5, 0)  # Blend original image and mask

        # Save the annotated image
        save_path = os.path.join(output_path, f"pred_{image_file}")
        cv2.imwrite(save_path, image)
