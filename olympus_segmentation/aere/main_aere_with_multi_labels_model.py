import os
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from aere_dataset import MultiLabelSegDataset
from olympus_segmentation.endo.transforms import get_val_transforms
from olympus_segmentation.aere.visualizing_aere_with_multi_labels_model import visualize_all_predictions_without_manual_annotation
from torchvision import models

# --------------------- Main ---------------------
if __name__ == '__main__':
    # Set your directories.
    channels = ['DAPI', 'FITC', 'TRITC']
    model_path =os.path.join(os.path.dirname(__file__), "weights", 'aere_multilabels_model_train_on_tamera.pth') # or 'aere_multilabels_model_train_on_stefan.pth'
    data_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\val'         # Contains subfolders: image, annotation. Image folder contains another layer of subfolders for each sample. Each sample folder contains 3-channel images.
    output_folder = r'C:\Users\Yifei\Documents\new_aere_model_training_on_tamera\results'          # Directory to save the prediction results.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
 
    # initialize the dataset with transforms.
    val_dataset = MultiLabelSegDataset(data_folder, channels, transform=get_val_transforms(), manual_annotation='False')
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=4)

    # model for aere segmentation.
    model = smp.Unet(
        encoder_name='resnet34',
        in_channels=3,
        classes=2,
        activation=None
    )  
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode

    # Plot predictions on a new images without manual annotation dataset.
    results_folder = os.path.join(output_folder)
    visualize_all_predictions_without_manual_annotation(model, val_dataset, results_folder)
    print("All predictions plotted and saved.")