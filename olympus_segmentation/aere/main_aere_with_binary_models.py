import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from aere_dataset import BinarySegDataset
from olympus_segmentation.endo.transforms import get_val_transforms
from olympus_segmentation.aere.visualizing_aere_with_binary_models import visualize_all_predictions_without_manual_annotation
from torchvision import models

# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    channels = ['DAPI', 'FITC', 'TRITC']
    aere_model_path =os.path.join(os.path.dirname(__file__), "weights", 'aere_binary_model_trained_on_tamera.pth') # or 'aere_binary_model_trained_on_stefan.pth'
    root_model_path = os.path.join(os.path.dirname(__file__), "weights", 'whole_root_binary_model_trained_tamera.pth')
    data_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\val'         # Contains subfolders: image, annotation. Image folder contains another layer of subfolders for each sample. Each sample folder contains 3-channel images.
    output_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\results'          # Directory to save the prediction results.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    # initialize the dataset with transforms.
    val_dataset = BinarySegDataset(data_folder, channels, transform=get_val_transforms(), manual_annotation='False')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # model for aere segmentation.
    aere_model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    )  
    aere_model.load_state_dict(torch.load(aere_model_path))
    aere_model.eval()  # Set the model to evaluation mode

    # model for whole root area segmentation.
    whole_root_model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    )  
    whole_root_model.load_state_dict(torch.load(root_model_path))
    whole_root_model.eval()  # Set the model to evaluation mode

    # Plot predictions on a new images without manual annotation dataset.
    results_folder = os.path.join(output_folder)
    visualize_all_predictions_without_manual_annotation(aere_model, whole_root_model, channels, val_dataset, results_folder)
    print("All predictions plotted and saved.")