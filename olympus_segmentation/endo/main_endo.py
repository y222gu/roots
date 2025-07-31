import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from endo_dataset import MultiChannelSegDataset
from olympus_segmentation.endo.transforms import get_val_transforms
from visualizing_endo_predictions import visualize_endo_predictions

# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    channels = ['DAPI', 'FITC', 'TRITC']
    model_path =os.path.join(os.path.dirname(__file__), "weights", 'endo_model_for_olympus.pth')
    data_folder = r'C:\Users\Yifei\Documents\new_endo_model\test'         # Contains subfolders: image, annotation. Image folder contains another layer of subfolders for each sample. Each sample folder contains 3-channel images.
    output_folder = r'C:\Users\Yifei\Documents\new_endo_model\results'          # Directory to save the prediction results.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    # initialize the dataset with transforms.
    val_dataset = MultiChannelSegDataset(data_folder, channels, transform=get_val_transforms(), manual_annotation='False')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # load the best model for inference.
    best_model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=1,
        activation=None
    )  
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()  # Set the model to evaluation mode
    
    # Plot predictions on samples.
    results_folder = os.path.join(output_folder)
    visualize_endo_predictions(best_model, channels, val_dataset, results_folder, manual_annotation='False', alpha=0.5)
    print("All predictions plotted and saved.")
