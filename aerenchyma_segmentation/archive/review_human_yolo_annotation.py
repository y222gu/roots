import numpy as np
from PIL import Image
import os
import cv2
def yolo_to_masks(image_dir, yolo_txt_dir, output_dir):
    """
    Convert YOLOv8 annotations into binary mask images.
    
    Parameters:
        yolo_txt_dir (str): Directory containing YOLO annotation text files.
        output_dir (str): Directory to save generated mask images.
        image_shape (tuple): Shape of the output mask (height, width).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    txt_files = [f for f in os.listdir(yolo_txt_dir) if f.endswith('.txt')]
    
    for txt_file in txt_files:
        image = os.path.join(image_dir, txt_file.replace('.txt', '.tif'))
        image_shape = cv2.imread(image).shape[:2]
        print(image_shape)
        mask = np.zeros(image_shape, dtype=np.uint8)
        file_path = os.path.join(yolo_txt_dir, txt_file)
        
        with open(file_path, 'r') as f:
            for line in f:
                data = line.strip().split()
                # Parse polygon coordinates (YOLO format)
                points = np.array(data[1:], dtype=float).reshape(-1, 2)
                points[:, 0] *= image_shape[1]  # Scale X to image width
                points[:, 1] *= image_shape[0]  # Scale Y to image height
                polygon = np.round(points).astype(np.int32)
                
                # Draw filled polygon on the mask
                mask = cv2.fillPoly(mask, [polygon], 255)
        
        # Save the mask with the same name as the text file (replacing extension)
        mask_img = Image.fromarray(mask)
        mask_img.save(os.path.join(output_dir, txt_file.replace('.txt', '.tif')))

if __name__ == "__main__":
    image_dir = os.path.join(os.getcwd(), "aerenchyma_segmentation", "data", "train","images")
    annotation_file_path = os.path.join(os.getcwd(), "aerenchyma_segmentation", "data", "train", "annotations")
    output_masks_dir = os.path.join(os.getcwd(), "aerenchyma_segmentation", "data", "train", "masks")
    yolo_to_masks(image_dir, annotation_file_path, output_masks_dir)
