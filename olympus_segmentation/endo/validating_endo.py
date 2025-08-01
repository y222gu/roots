import os
import torch
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from segmentation_models_pytorch.utils.metrics import IoU
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader
from endo_dataset import MultiChannelSegDataset
from olympus_segmentation.endo.transforms import get_val_transforms
from visualizing_endo_predictions import visualize_endo_predictions

# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    channels = ['DAPI', 'FITC', 'TRITC']
    model_path =os.path.join(os.path.dirname(__file__), "weights", 'endo_model_for_olympus.pth')
    val_data_folder = r'C:\Users\Yifei\Documents\new_endo_model_training\test'          # Contains subfolders: image, annotation.          # Contains subfolders: image, annotation.
    output_folder = r'C:\Users\Yifei\Documents\new_endo_model_training\results'          # Directory to save the best model.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    # initialize the dataset with transforms.
    val_dataset = MultiChannelSegDataset(val_data_folder, channels, transform=get_val_transforms(), manual_annotation='True')

    # # load the best model for inference.
    best_model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    )  # Use the same device as the data loader
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()  # Set the model to evaluation mode
    
    # Plot predictions on a few samples from the test (validation) dataset.
    results_folder = os.path.join(output_folder)
    visualize_endo_predictions(best_model, channels, val_dataset, results_folder, manual_annotation='True', alpha=0.5)
    print("All predictions plotted and saved.")
