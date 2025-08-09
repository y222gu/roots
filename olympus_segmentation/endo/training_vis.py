import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_training_progress(json_file_path):
    """
    Plot training progress metrics vs epochs
    
    Args:
        json_file_path (str): Path to the JSON file containing training metrics
    """
    
    # Load the JSON data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, dict):
        # Check if it's nested (model_name -> metrics) or direct
        if any('train_metrics' in str(v) for v in data.values() if isinstance(v, dict)):
            model_name = list(data.keys())[0]  # Get first model
            model_data = data[model_name]
        elif 'train_metrics' in data and 'val_metrics' in data:
            model_name = "Training Progress"
            model_data = data
        else:
            raise ValueError("Cannot find train_metrics and val_metrics in JSON")
    else:
        raise ValueError(f"Expected dictionary, got {type(data)}")
    
    # Extract training and validation metrics
    train_metrics = pd.DataFrame(model_data['train_metrics'])
    val_metrics = pd.DataFrame(model_data['val_metrics'])
    epochs = np.arange(1, len(train_metrics) + 1)
    
    print(f"Loaded {len(epochs)} epochs of training data")
    print(f"Model: {model_name}")
    
    # Create the comprehensive training progress plot
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    # fig.suptitle(f'{model_name} - Training Progress Over Epochs', fontsize=16, y=0.98)
    
    # Define colors
    train_color = '#1f77b4'  # Blue
    val_color = '#ff7f0e'    # Orange
    
    # 1. Loss vs Epochs
    axes[0, 0].plot(epochs, train_metrics['loss'], color=train_color, linewidth=2, label='Training')
    axes[0, 0].plot(epochs, val_metrics['loss'], color=val_color, linewidth=2, label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss vs Epoch')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Dice Score vs Epochs
    axes[0, 1].plot(epochs, train_metrics['dice'], color=train_color, linewidth=2, label='Training')
    axes[0, 1].plot(epochs, val_metrics['dice'], color=val_color, linewidth=2, label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Dice Score')
    axes[0, 1].set_title('Dice Score vs Epoch')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. IoU vs Epochs
    axes[0, 2].plot(epochs, train_metrics['iou'], color=train_color, linewidth=2, label='Training')
    axes[0, 2].plot(epochs, val_metrics['iou'], color=val_color, linewidth=2, label='Validation')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('IoU')
    axes[0, 2].set_title('IoU vs Epoch')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision vs Epochs
    axes[1, 0].plot(epochs, train_metrics['precision'], color=train_color, linewidth=2, label='Training')
    axes[1, 0].plot(epochs, val_metrics['precision'], color=val_color, linewidth=2, label='Validation')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].set_title('Precision vs Epoch')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Recall vs Epochs
    axes[1, 1].plot(epochs, train_metrics['recall'], color=train_color, linewidth=2, label='Training')
    axes[1, 1].plot(epochs, val_metrics['recall'], color=val_color, linewidth=2, label='Validation')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].set_title('Recall vs Epoch')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. F1 Score vs Epochs
    axes[1, 2].plot(epochs, train_metrics['f1'], color=train_color, linewidth=2, label='Training')
    axes[1, 2].plot(epochs, val_metrics['f1'], color=val_color, linewidth=2, label='Validation')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('F1 Score')
    axes[1, 2].set_title('F1 Score vs Epoch')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Accuracy vs Epochs
    axes[2, 0].plot(epochs, train_metrics['accuracy'], color=train_color, linewidth=2, label='Training')
    axes[2, 0].plot(epochs, val_metrics['accuracy'], color=val_color, linewidth=2, label='Validation')
    axes[2, 0].set_xlabel('Epoch')
    axes[2, 0].set_ylabel('Accuracy')
    axes[2, 0].set_title('Accuracy vs Epoch')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # 8. Learning Rate vs Epochs
    axes[2, 1].plot(epochs, train_metrics['lr'], color='#2ca02c', linewidth=2)
    axes[2, 1].set_xlabel('Epoch')
    axes[2, 1].set_ylabel('Learning Rate')
    axes[2, 1].set_title('Learning Rate vs Epoch')
    axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Training Time vs Epochs
    axes[2, 2].plot(epochs, train_metrics['epoch_time'], color='#d62728', linewidth=2)
    axes[2, 2].set_xlabel('Epoch')
    axes[2, 2].set_ylabel('Time (seconds)')
    axes[2, 2].set_title('Training Time vs Epoch')
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'training_progress_over_epochs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training progress plot saved as: {output_path}")
    
    plt.show()
    
    # Create a focused plot on key metrics
    fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig2.suptitle('Key Training Metrics Progress', fontsize=16)
    
    # Loss
    ax1.plot(epochs, train_metrics['loss'], 'b-', linewidth=3, label='Training', alpha=0.8)
    ax1.plot(epochs, val_metrics['loss'], 'r-', linewidth=3, label='Validation', alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Loss Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Dice Score
    ax2.plot(epochs, train_metrics['dice'], 'b-', linewidth=3, label='Training', alpha=0.8)
    ax2.plot(epochs, val_metrics['dice'], 'r-', linewidth=3, label='Validation', alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Dice Score', fontsize=12)
    ax2.set_title('Dice Score Progress', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # IoU
    ax3.plot(epochs, train_metrics['iou'], 'b-', linewidth=3, label='Training', alpha=0.8)
    ax3.plot(epochs, val_metrics['iou'], 'r-', linewidth=3, label='Validation', alpha=0.8)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('IoU', fontsize=12)
    ax3.set_title('IoU Progress', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Accuracy
    ax4.plot(epochs, train_metrics['accuracy'], 'b-', linewidth=3, label='Training', alpha=0.8)
    ax4.plot(epochs, val_metrics['accuracy'], 'r-', linewidth=3, label='Validation', alpha=0.8)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Accuracy', fontsize=12)
    ax4.set_title('Accuracy Progress', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=11)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save focused plot
    focused_output = 'key_metrics_progress.png'
    plt.savefig(focused_output, dpi=300, bbox_inches='tight')
    print(f"Key metrics plot saved as: {focused_output}")
    
    plt.show()
    
    # Print progress summary
    print("\n" + "="*70)
    print("TRAINING PROGRESS SUMMARY")
    print("="*70)
    
    # Find best epochs
    best_val_dice_epoch = val_metrics['dice'].idxmax() + 1
    best_val_loss_epoch = val_metrics['loss'].idxmin() + 1
    best_val_iou_epoch = val_metrics['iou'].idxmax() + 1
    
    print(f"\nBest Performance:")
    print(f"├─ Best Validation Dice: {val_metrics['dice'].max():.4f} at epoch {best_val_dice_epoch}")
    print(f"├─ Lowest Validation Loss: {val_metrics['loss'].min():.4f} at epoch {best_val_loss_epoch}")
    print(f"└─ Best Validation IoU: {val_metrics['iou'].max():.4f} at epoch {best_val_iou_epoch}")
    
    # Progress from start to end
    print(f"\nProgress from Epoch 1 to {len(epochs)}:")
    print(f"├─ Dice Score: {train_metrics['dice'].iloc[0]:.4f} → {train_metrics['dice'].iloc[-1]:.4f} (Train)")
    print(f"│               {val_metrics['dice'].iloc[0]:.4f} → {val_metrics['dice'].iloc[-1]:.4f} (Val)")
    print(f"├─ IoU:        {train_metrics['iou'].iloc[0]:.4f} → {train_metrics['iou'].iloc[-1]:.4f} (Train)")
    print(f"│               {val_metrics['iou'].iloc[0]:.4f} → {val_metrics['iou'].iloc[-1]:.4f} (Val)")
    print(f"└─ Loss:       {train_metrics['loss'].iloc[0]:.4f} → {train_metrics['loss'].iloc[-1]:.4f} (Train)")
    print(f"                {val_metrics['loss'].iloc[0]:.4f} → {val_metrics['loss'].iloc[-1]:.4f} (Val)")
    
    # Training time stats
    total_time = train_metrics['epoch_time'].sum()
    avg_time = train_metrics['epoch_time'].mean()
    print(f"\nTraining Time:")
    print(f"├─ Total: {total_time/3600:.2f} hours")
    print(f"├─ Average per epoch: {avg_time:.1f} seconds")
    print(f"└─ Fastest epoch: {train_metrics['epoch_time'].min():.1f}s, Slowest: {train_metrics['epoch_time'].max():.1f}s")

def main():
    # UPDATE THIS PATH to your training metrics JSON file
    # This should NOT be the inference results file
    json_file_path = r"C:\Users\Yifei\Documents\data_for_publication\results\models\Unet_resnet34\detailed_metrics.json"
    
    try:
        plot_training_progress(json_file_path)
        
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        print("\nPossible solutions:")
        print("1. Check if the file path is correct")
        print("2. Make sure you're pointing to your TRAINING metrics file, not inference results")
        print("3. Look for a file that contains epoch-by-epoch training data")
        print("   (should have train_metrics and val_metrics arrays)")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nMake sure your JSON file contains training metrics with:")
        print("- train_metrics: array of per-epoch training metrics")
        print("- val_metrics: array of per-epoch validation metrics")

if __name__ == "__main__":
    main()