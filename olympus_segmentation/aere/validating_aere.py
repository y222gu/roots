import os
import torch
import segmentation_models_pytorch as smp
from torch.amp import autocast, GradScaler
from segmentation_models_pytorch.utils.metrics import IoU
import matplotlib.pyplot as plt
import cv2
import numpy as np
from torch.utils.data import DataLoader
from aere_dataset import BinarySegDataset
from olympus_segmentation.endo.transforms import get_val_transforms
from olympus_segmentation.aere.visualizing_aere_with_binary_models import visualize_all_predictions_with_manual_annotation
from torchvision import models

# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    channels = ['DAPI', 'FITC', 'TRITC']
    model_path =os.path.join(os.path.dirname(__file__), "weights", 'whole_root_tamera.pth')
    val_data_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\test'         # Contains subfolders: image, annotation.          # Contains subfolders: image, annotation.
    output_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\results'           # Directory to save the best model.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    # initialize the dataset with transforms.
    val_dataset = BinarySegDataset(val_data_folder, channels, transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # # load the best model for inference.
    model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    )  # Use the same device as the data loader
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Plot predictions on a few samples from the test (validation) dataset.
    results_folder = os.path.join(output_folder)
    visualize_all_predictions_with_manual_annotation(model, channels, val_dataset, results_folder)
    print("All predictions plotted and saved.")
