import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from endo_dataset import MultiChannelSegDataset
from transforms import get_train_transforms, get_val_transforms


class DiceBCELoss(nn.Module):
    def __init__(self, dice_kwargs={}, bce_kwargs={}, w_dice=1.0, w_bce=1.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss(**dice_kwargs)
        self.bce  = smp.losses.SoftBCEWithLogitsLoss(**bce_kwargs)
        self.w_dice = w_dice
        self.w_bce  = w_bce

    def forward(self, logits, targets):
        return self.w_dice * self.dice(logits, targets) + self.w_bce * self.bce(logits, targets)


class DiceSoftBCELoss(nn.Module):
    def __init__(self, w_dice=1.0, w_bce=1.0, dice_mode='multilabel', bce_kwargs={}):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode=dice_mode)
        self.bce  = smp.losses.SoftBCEWithLogitsLoss(**bce_kwargs)
        self.w_dice, self.w_bce = w_dice, w_bce

    def forward(self, logits, targets):
        return self.w_dice * self.dice(logits, targets) + self.w_bce * self.bce(logits, targets)


def dice_coef_multilabel(probs, labels, smooth=1e-6):
    preds = (probs > 0.5).float()
    intersect = (preds * labels).sum(dim=(2,3))
    union     = preds.sum(dim=(2,3)) + labels.sum(dim=(2,3))
    dice      = (2*intersect + smooth) / (union + smooth)
    return dice.mean().item()


def iou_coef_multilabel(probs, labels, smooth=1e-6):
    preds = (probs > 0.5).float()
    intersect = (preds * labels).sum(dim=(2,3))
    union     = preds.sum(dim=(2,3)) + labels.sum(dim=(2,3)) - intersect
    iou       = (intersect + smooth) / (union + smooth)
    return iou.mean().item()


def precision_multilabel(probs, labels, smooth=1e-6):
    preds = (probs > 0.5).float()
    tp    = (preds * labels).sum()
    fp    = (preds * (1 - labels)).sum()
    return ((tp + smooth) / (tp + fp + smooth)).item()


def accuracy_multilabel(probs, labels, smooth=1e-6):
    preds   = (probs > 0.5).float()
    correct = (preds == labels).sum()
    total   = torch.numel(labels)
    return (correct.float() / (total + smooth)).item()


def train_one_epoch(model, loader, loss_fn, optimizer, device):
    model.train()
    running = 0.0
    for imgs, masks, _ in loader:
        imgs  = imgs.to(device)
        # convert masks from (B, H, W, C) to (B, C, H, W)
        masks = masks.to(device).permute(0, 3, 1, 2)
        optimizer.zero_grad()
        logits = model(imgs)            # (B, C, H, W)
        loss   = loss_fn(logits, masks)
        loss.backward()
        optimizer.step()
        running += loss.item() * imgs.size(0)
    return running / len(loader.dataset)


def validate_one_epoch(model, loader, loss_fn, device):
    model.eval()
    tot_loss = 0.0
    dice_scores, iou_scores = [], []
    prec_scores, acc_scores = [], []

    with torch.no_grad():
        for imgs, masks, _ in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device).permute(0, 3, 1, 2)
            logits = model(imgs)
            tot_loss += loss_fn(logits, masks).item() * imgs.size(0)

            probs = torch.sigmoid(logits)
            dice_scores.append(dice_coef_multilabel(probs, masks))
            iou_scores.append(iou_coef_multilabel(probs, masks))
            prec_scores.append(precision_multilabel(probs, masks))
            acc_scores.append(accuracy_multilabel(probs, masks))

    return (
        tot_loss / len(loader.dataset),
        np.mean(dice_scores),
        np.mean(iou_scores),
        np.mean(prec_scores),
        np.mean(acc_scores),
    )


def train_and_evaluate(train_data_dir, val_data_dir, channels, model_name, constructor, hparams, out_dir):
    # --- data ---
    train_ds = MultiChannelSegDataset(
        train_data_dir, channels,
        transform=get_train_transforms()
    )
    val_ds = MultiChannelSegDataset(
        val_data_dir, channels,
        transform=get_val_transforms()
    )
    train_loader = DataLoader(
        train_ds, batch_size=hparams['batch_size'],
        shuffle=True,  num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=hparams['batch_size'],
        shuffle=False, num_workers=4
    )

    # --- model ---
    model = constructor(
        encoder_weights='imagenet',
        in_channels=len(channels),
        classes=2,
    ).to(hparams['device'])

    # --- loss / optimizer / scheduler ---
    loss_fn = DiceBCELoss(dice_kwargs={'mode': 'multilabel'})
    optimizer = optim.AdamW(
        model.parameters(),
        lr=hparams['lr'],
        weight_decay=hparams['weight_decay']
    )
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams['lr'],
        steps_per_epoch=len(train_loader),
        epochs=hparams['epochs']
    )

    # --- early stopping setup ---
    patience = hparams.get('patience', 10)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_model_path = os.path.join(
        out_dir,
        f"best_model_{model_name}_{'_'.join(channels)}.pth"
    )

    # --- history dict ---
    history = {
        'epoch':[],
        'train_loss':[], 'val_loss':[],
        'val_dice':[],  'val_iou':[],
        'val_prec':[],  'val_acc':[]
    }

    for epoch in range(1, hparams['epochs']+1):
        tr_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, hparams['device']
        )
        vl_loss, vd, vi, vp, va = validate_one_epoch(
            model, val_loader, loss_fn, hparams['device']
        )
        scheduler.step()

        # record metrics
        history['epoch'].append(epoch)
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(vl_loss)
        history['val_dice'].append(vd)
        history['val_iou'].append(vi)
        history['val_prec'].append(vp)
        history['val_acc'].append(va)

        print(
            f"{model_name} | Ch={channels} | Ep {epoch}/{hparams['epochs']} "
            f"– train_loss: {tr_loss:.4f}, val_loss: {vl_loss:.4f}, "
            f"dice: {vd:.4f}, iou: {vi:.4f}, "
            f"prec: {vp:.4f}, acc: {va:.4f}"
        )

        # check for improvement
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best val_loss: {best_val_loss:.4f} (epoch {epoch - patience})"
                )
                break

    # --- save metrics to CSV ---
    df = pd.DataFrame(history)
    csv_path = os.path.join(
        out_dir, f"metrics_{model_name}_{'_'.join(channels)}.csv"
    )
    df.to_csv(csv_path, index=False)

    # --- plot curves ---
    epochs = history['epoch']
    plt.figure()
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'],   label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(out_dir, f"loss_{model_name}_{'_'.join(channels)}.png"))

    plt.figure()
    plt.plot(epochs, history['val_dice'], label='Val Dice')
    plt.plot(epochs, history['val_iou'],  label='Val IoU')
    plt.plot(epochs, history['val_prec'], label='Val Precision')
    plt.plot(epochs, history['val_acc'],  label='Val Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Metric'); plt.legend()
    plt.savefig(os.path.join(out_dir, f"metrics_{model_name}_{'_'.join(channels)}.png"))


if __name__ == '__main__':
    # configuration
    train_data_dir = r'C:\Users\Yifei\Documents\data_for_publication\train_preprocessed'
    val_data_dir   = r'C:\Users\Yifei\Documents\data_for_publication\val_preprocessed'
    output_dir     = r'C:\Users\Yifei\Documents\data_for_publication\results'
    os.makedirs(output_dir, exist_ok=True)

    hyperparams = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'batch_size': 32,
        'lr': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'patience': 10,  # early stopping patience
    }

    # model architectures to compare
    model_configs = {
        'unet_resnet34': lambda **kw: smp.Unet(encoder_name='resnet34', **kw),
    }

    # channel combinations to test
    channel_sets = [
        ['DAPI','FITC','TRITC']
    ]

    # run experiments
    for channels in channel_sets:
        for mname, constructor in model_configs.items():
            exp_dir = os.path.join(output_dir, f"{mname}_{'_'.join(channels)}")
            os.makedirs(exp_dir, exist_ok=True)
            train_and_evaluate(
                train_data_dir, val_data_dir, channels, mname, constructor,
                hyperparams, exp_dir
            )
