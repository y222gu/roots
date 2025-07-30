import os
import cv2
import numpy as np
from root_area_detection_unet import segment_root
from yolo_segment_aerenchyma_run_prediction import segment_aerenchyma
from ultralytics import YOLO
from torchvision import models, transforms
import onnxruntime as ort


if __name__ == "__main__":

    ########################################################
    # Input Parameters
    ########################################################
    image_folder = os.path.join(os.getcwd(),"Sorghum Genotype Screen 5cm")
    output_path = os.path.join(os.getcwd(),"Sorghum Genotype Screen 5cm output")
    aerenchyma_model_path =os.path.join(os.getcwd(),'aerenchyma_segmentation','runs', 'yolov8_segmentation3','weights','best.pt')

    
    ########################################################
    # Analyze the images
    ########################################################
    model_for_aerenchyma = YOLO(aerenchyma_model_path)
    model_for_root = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model_for_root.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Define a transformation to normalize the input image
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    list_of_image_files = os.listdir(image_folder)

    for image_file in list_of_image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        # # normalize the image by mean
        # stretched = cv2.normalize(
        #     image,              # src
        #     None,              # dst (if None, returns a new array)
        #     alpha=0,           # new min value
        #     beta=255,          # new max value
        #     norm_type=cv2.NORM_MINMAX
        # )

        # # show the image
        # cv2.imshow("Normalized Image", stretched)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
    
        root_mask, root_area, transformed = segment_root(model_for_root, image, transform, option='highest_confidence',)
        aerenchyma_mask, aerenchyma_area = segment_aerenchyma(model_for_aerenchyma, image)


        # Ensure the masks and root area are not None
        if root_mask is None or aerenchyma_mask is None or root_area == 0:
            print(f"Skipping {image_file} due to missing masks or zero root area.")
            continue

        aerenchyma_ratio = aerenchyma_area / root_area

        # Visualize the overlay of the aerenchyma_mask with blue at opacity of 0.3 on the original image
        # (root_mask - aerenchyma_mask) with magenta at opacity of 0.3 on the original image
        overlay = image.copy()

        # turn the image to float for scaling
        overlay = overlay.astype(float)
        # Apply scaling
        overlay = overlay * 1.5
        
        # Clip values to valid range and convert back to 16-bit
        overlay = np.clip(overlay, 0, 65535).astype(np.uint8)

        # convert back to cv2 image format
        overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

        root_mask_subtracted =  np.logical_and(root_mask, np.logical_not(aerenchyma_mask)).astype(np.uint8)
        root_mask_color = np.stack([root_mask_subtracted * 255, root_mask_subtracted * 0, root_mask_subtracted * 255], axis=-1)  # Red
        aerenchyma_mask_color = np.stack([aerenchyma_mask * 255, aerenchyma_mask * 0, aerenchyma_mask * 0], axis=-1)  # Blue
        overlay = cv2.addWeighted(overlay, 1, root_mask_color, 0.3, 0)
        overlay = cv2.addWeighted(overlay, 1, aerenchyma_mask_color, 0.6, 0)
        combined_image = np.hstack((overlay, image))

        #Add title at the top of the image
        title_overlay = combined_image.copy()
        legend_height = int(combined_image.shape[0] * 0.1)  # Legend height is 10% of image height
        font_scale = combined_image.shape[0] / 800  # Font scale proportional to image size
        thickness = max(1, int(font_scale * 2))  # Scaled thickness
        spacing = int(legend_height * 0.2)  # Spacing between legend items

        title = f"Aerenchyma/Root Ratio: {aerenchyma_ratio * 100}%"
        text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.5, thickness)[0]
        text_x = (title_overlay.shape[1] - text_size[0]) // 2
        text_y = text_size[1] + spacing
        cv2.putText(title_overlay, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale * 1.5, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save the segmented imag        
        new_file_name = 'ratio' + image_file
        save_path = os.path.join(output_path, new_file_name)
        cv2.imwrite(save_path, title_overlay)
        print(f"{image_file} aerenchyma/root ratio: {aerenchyma_ratio *100}% ")