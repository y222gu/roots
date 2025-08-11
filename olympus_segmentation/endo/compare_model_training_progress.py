import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def plot_training_progress(folder_path):
    """
    Plot training progress metrics vs epochs
    
    Args:
        json_file_path (str): Path to the JSON file containing training metrics
    """
    
    # find all JSON files in the directory
    folder_path = Path(folder_path)
    json_files = list(folder_path.glob("*.json"))

    # Collect summary statistics for each model
    summary_stats = []

    for json_file in json_files:
        # Load the JSON data
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON structures
        if isinstance(data, dict):
            model_arch = list(data.keys())[0]  # Get first model
            model_data = data[model_arch]
            model_name = json_file.stem
            training_size = model_name.split('_')[2]  # Extract training size from filename
        else:
            raise ValueError(f"Expected dictionary, got {type(data)}")
        
        # Extract training and validation metrics
        train_metrics = pd.DataFrame(model_data['train_metrics'])
        val_metrics = pd.DataFrame(model_data['val_metrics'])
        epochs = np.arange(1, len(train_metrics) + 1)
        
        # Collect summary statistics
        summary_stats.append({
            "model": model_name,
            "training_size": int(training_size),
            "best_val_dice": val_metrics['dice'].max(),
            "best_val_dice_epoch": val_metrics['dice'].idxmax() + 1,
            "best_val_loss": val_metrics['loss'].min(),
            "best_val_loss_epoch": val_metrics['loss'].idxmin() + 1,
            "best_val_iou": val_metrics['iou'].max(),
            "best_val_iou_epoch": val_metrics['iou'].idxmax() + 1,
            "train_dice_start": train_metrics['dice'].iloc[0],
            "train_dice_end": train_metrics['dice'].iloc[-1],
            "val_dice_start": val_metrics['dice'].iloc[0],
            "val_dice_end": val_metrics['dice'].iloc[-1],
            "train_iou_start": train_metrics['iou'].iloc[0],
            "train_iou_end": train_metrics['iou'].iloc[-1],
            "val_iou_start": val_metrics['iou'].iloc[0],
            "val_iou_end": val_metrics['iou'].iloc[-1],
            "train_loss_start": train_metrics['loss'].iloc[0],
            "train_loss_end": train_metrics['loss'].iloc[-1],
            "val_loss_start": val_metrics['loss'].iloc[0],
            "val_loss_end": val_metrics['loss'].iloc[-1],
            "total_time_hours": train_metrics['epoch_time'].sum() / 3600,
            "avg_epoch_time": train_metrics['epoch_time'].mean(),
            "min_epoch_time": train_metrics['epoch_time'].min(),
            "max_epoch_time": train_metrics['epoch_time'].max()
        })

    # Convert summary_stats to DataFrame for plotting
    summary_df = pd.DataFrame(summary_stats).sort_values(by='training_size').reset_index(drop=True)

    # Plot best validation metrics vs models
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    # Define consistent colors for metrics
    dice_color = 'tab:blue'
    iou_color = 'tab:green'
    loss_color = 'tab:red'

    # Plot best validation metrics vs models (subplot 1)
    axes[0].plot(summary_df['training_size'], summary_df['best_val_dice'], marker='o', label='Best Val Dice', color=dice_color, linestyle='dashed')
    axes[0].plot(summary_df['training_size'], summary_df['best_val_loss'], marker='o', label='Best Val Loss', color=loss_color, linestyle='dashed')
    axes[0].plot(summary_df['training_size'], summary_df['best_val_iou'], marker='o', label='Best Val IoU', color=iou_color, linestyle='dashed')

    # Annotate each point with the best epoch number
    for i, row in summary_df.iterrows():
        axes[0].annotate(f"Ep {row['best_val_dice_epoch']}", 
                         (row['training_size'], row['best_val_dice']),
                         textcoords="offset points", xytext=(0,8), ha='center', color=dice_color, fontsize=8)
        axes[0].annotate(f"Ep {row['best_val_loss_epoch']}", 
                         (row['training_size'], row['best_val_loss']),
                         textcoords="offset points", xytext=(0,8), ha='center', color=loss_color, fontsize=8)
        axes[0].annotate(f"Ep {row['best_val_iou_epoch']}", 
                         (row['training_size'], row['best_val_iou']),
                         textcoords="offset points", xytext=(0,8), ha='center', color=iou_color, fontsize=8)
    axes[0].set_title('Best Validation Metrics vs Model')
    axes[0].set_ylabel('Metric Value')
    axes[0].set_xlabel('Training Size')
    axes[0].set_xticks(summary_df['training_size'])
    axes[0].set_xticklabels(summary_df['training_size'], rotation=45, ha='right')
    axes[0].grid(True)
    axes[0].set_ylim(0, 1.1)
    axes[0].legend()

    # Plot the end metrics of validation and training for each model (subplot 3)
    axes[1].plot(summary_df['training_size'], summary_df['train_dice_end'], marker='o', label='Train Dice End', color=dice_color, linestyle='-')
    axes[1].plot(summary_df['training_size'], summary_df['val_dice_end'], marker='o', label='Val Dice End', color=dice_color, linestyle='dashed')
    axes[1].plot(summary_df['training_size'], summary_df['train_iou_end'], marker='o', label='Train IoU End', color=iou_color, linestyle='-')
    axes[1].plot(summary_df['training_size'], summary_df['val_iou_end'], marker='o', label='Val IoU End', color=iou_color, linestyle='dashed')
    axes[1].plot(summary_df['training_size'], summary_df['train_loss_end'], marker='o', label='Train Loss End', color=loss_color, linestyle='-')
    axes[1].plot(summary_df['training_size'], summary_df['val_loss_end'], marker='o', label='Val Loss End', color=loss_color, linestyle='dashed')
    axes[1].set_title(' Metrics at 350th epoch VS Training Size')
    axes[1].set_ylabel('Metric Value')
    axes[1].set_xlabel('Training Size')
    axes[1].set_xticks(summary_df['training_size'])
    axes[1].set_xticklabels(summary_df['training_size'], rotation=45, ha='right')
    axes[1].grid(True)
    axes[1].set_ylim(0, 1.1)
    axes[1].legend()

    # Plot training time comparison (subplot 2)
    axes[2].plot(summary_df['training_size'], summary_df['total_time_hours'], marker='o', linestyle='-', color='tab:purple')
    axes[2].set_title('Total Training Time (hours) vs Training Size')
    axes[2].set_ylabel('Hours')
    axes[2].set_xlabel('Training Size')
    axes[2].set_xticks(summary_df['training_size'])
    axes[2].set_xticklabels(summary_df['training_size'], rotation=45, ha='right')
    axes[2].grid(True)
    axes[2].set_ylim(0, summary_df['total_time_hours'].max() * 1.1)  # Add some padding to the y-axis

    plt.tight_layout()
    plt.show()
    plt.close()


def main():
    # UPDATE THIS PATH to your training metrics JSON file
    # This should NOT be the inference results file
    json_file_path = r"C:\Users\yifei\Documents\data_for_publication\only_train_on_sorghum\results"
    
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