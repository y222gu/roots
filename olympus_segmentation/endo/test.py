import os
import cv2
from sklearn import base
import torch
import numpy as np
from torch.utils.data import DataLoader
from segmentation_models_pytorch import Unet, DeepLabV3
from endo_dataset import MultiChannelSegDataset
from transforms import get_val_transforms
import csv
from endo_training_multiple_model import dice_coef_multilabel, iou_coef_multilabel, precision_multilabel, accuracy_multilabel  # import your metrics


def test_and_evaluate(model, loader, output_folder, device, threshold=0.5):
    model.eval()
    all_dice, all_iou, all_precision, all_accuracy = [], [], [], []
    os.makedirs(output_folder, exist_ok=True)

    dataset = loader.dataset
    bs = loader.batch_size

    with torch.no_grad():
        for batch_idx, (imgs, masks, sids) in enumerate(loader):
            imgs  = imgs.to(device)
            masks = masks.to(device).permute(0,3,1,2)   # (B,2,H,W)
            logits = model(imgs)
            probs  = torch.sigmoid(logits)

            all_dice.append(dice_coef_multilabel(probs, masks))
            all_iou.append(iou_coef_multilabel(probs, masks))
            all_precision.append(precision_multilabel(probs, masks))
            all_accuracy.append(accuracy_multilabel(probs, masks))

            pred_mask = (probs > threshold).cpu().numpy().astype(np.uint8)
            gt_mask   = masks.cpu().numpy().astype(np.uint8)

            for i, sid in enumerate(sids):
                pred = pred_mask[i]  # 2×H×W
                gt   = gt_mask[i]

                # --- build side-by-side overlay on your preprocess() output ---
                # 1) reload & preprocess the raw stack
                dataset_idx = batch_idx * bs + i
                img_dir, _, _ = dataset.samples[dataset_idx]
                raw_stack = dataset._load_image_stack(img_dir)  # H,W,C float32
                proc      = dataset.preprocess(raw_stack)       # H,W,C in [0,1]

                # 2) pick channel 0 for background, scale to uint8
                base = (proc[...,0] * 255).astype(np.uint8)
                _, h, w = pred.shape
                base_resized = cv2.resize(base, (w, h),
                                          interpolation=cv2.INTER_LINEAR)
                base_bgr = cv2.cvtColor(base_resized, cv2.COLOR_GRAY2BGR)

                # 3) create prediction overlay
                overlay_pred = base_bgr.copy()
                overlay_pred[pred[0] == 1] = [255, 0, 0]
                overlay_pred[pred[1] == 1] = [0, 0, 255]
                blended_pred = cv2.addWeighted(base_bgr, 0.5, overlay_pred, 0.5, 0)

                # 4) create ground-truth overlay
                overlay_gt = base_bgr.copy()
                overlay_gt[gt[0] == 1] = [255, 0, 0]
                overlay_gt[gt[1] == 1] = [0, 0, 255]
                blended_gt = cv2.addWeighted(base_bgr, 0.5, overlay_gt, 0.5, 0)

                # 5) concatenate side by side: [GT | Pred]
                combined = cv2.hconcat([blended_gt, blended_pred])
                cv2.imwrite(os.path.join(output_folder, f'{sid}_combined_overlay.png'),
                            combined)

    mean_dice = float(np.mean(all_dice))
    mean_iou  = float(np.mean(all_iou))
    mean_precision = float(np.mean(all_precision))
    mean_accuracy  = float(np.mean(all_accuracy))
    
    csv_path = os.path.join(output_folder, 'evaluation_metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Dice', 'IoU', 'Precision', 'Accuracy'])
        writer.writerow([mean_dice, mean_iou, mean_precision, mean_accuracy])
    return mean_dice, mean_iou, mean_precision, mean_accuracy

if __name__ == '__main__':
    # --- config ---
    test_data_dir = r'C:\Users\Yifei\Documents\data_for_publication\test_preprocessed'
    # save each channel separately to a specific folder
    output_folder = r'C:\Users\Yifei\Documents\data_for_publication\results\test_preprocessed_results'
    os.makedirs(output_folder, exist_ok=True)
    model_name    = 'unet_resnet34'   # or whichever you just trained
    channels      = ['DAPI','FITC','TRITC']
    best_model_path = os.path.join(
        r'C:\Users\Yifei\Documents\data_for_publication\results\unet_resnet34_DAPI_FITC_TRITC',
        f'best_model_{model_name}_{"_".join(channels)}.pth'
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- dataset & loader ---
    test_ds = MultiChannelSegDataset(
        test_data_dir, channels, transform=get_val_transforms()
    )
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=4)

    # --- model init & load ---
    constructor = {
        'unet_resnet34': lambda **kw: Unet(encoder_name='resnet34', **kw),
        'deeplabv3_resnet50': lambda **kw: DeepLabV3(encoder_name='resnet50', **kw),
    }[model_name]

    model = constructor(
        encoder_weights=None,
        in_channels=len(channels),
        classes=2
    ).to(device)
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # --- run test ---
    dice, iou, precision, accuracy = test_and_evaluate(model, test_loader, output_folder, device)
    print(f"Test set results → Mean Dice: {dice:.4f}, Mean IoU: {iou:.4f}, Mean Precision: {precision:.4f}, Mean Accuracy: {accuracy:.4f}")
    print("All raw and overlay masks saved under ./test_predictions/")
