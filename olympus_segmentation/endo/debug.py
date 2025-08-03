import torch
import segmentation_models_pytorch as smp
from utils import monitor_gpu_memory, aggressive_cleanup
import gc

def test_model_memory():
    """Test different model configurations for memory usage"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    # Test configurations
    test_configs = [
        ('mobilenet_v2', 'Unet', 512, 16),
        ('mobilenet_v2', 'Unet', 512, 8),
        ('mobilenet_v2', 'Unet', 256, 16),
        ('resnet50', 'Unet', 512, 8),
        ('resnet50', 'Unet', 256, 16),
        ('densenet121', 'Unet', 512, 4),
        ('efficientnet-b4', 'Unet', 512, 4),
        ('efficientnet-b4', 'Unet', 256, 8),
    ]
    
    print("\nTesting different configurations...")
    print("="*80)
    
    for encoder, arch, img_size, batch_size in test_configs:
        print(f"\nTesting: {encoder} + {arch}, size={img_size}, batch={batch_size}")
        
        try:
            # Clean before test
            aggressive_cleanup()
            print("Initial memory:")
            monitor_gpu_memory()
            
            # Create model
            model = smp.Unet(
                encoder_name=encoder,
                encoder_weights='imagenet',
                in_channels=3,
                classes=2
            )
            model = model.to(device)
            
            print("After model creation:")
            monitor_gpu_memory()
            
            # Test forward pass
            dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
            with torch.no_grad():
                output = model(dummy_input)
            
            print("After forward pass:")
            monitor_gpu_memory()
            
            # Test backward pass
            model.train()
            optimizer = torch.optim.Adam(model.parameters())
            loss_fn = torch.nn.BCEWithLogitsLoss()
            
            dummy_target = torch.randn_like(output)
            loss = loss_fn(output, dummy_target)
            loss.backward()
            optimizer.step()
            
            print("After backward pass:")
            monitor_gpu_memory()
            
            print(f"✅ Configuration successful!")
            
            # Cleanup
            del model, dummy_input, output, loss, optimizer
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"❌ OOM Error: {str(e)}")
            else:
                print(f"❌ Other Error: {str(e)}")
        
        except Exception as e:
            print(f"❌ Unexpected Error: {str(e)}")
        
        finally:
            aggressive_cleanup()
    
    print("\n" + "="*80)
    print("Testing complete!")


def test_dataset_loading():
    """Test dataset loading with different batch sizes"""
    from utils import MultiChannelSegDataset
    from torch.utils.data import DataLoader
    
    print("\nTesting dataset loading...")
    print("="*80)
    
    # Create a small test dataset
    try:
        dataset = MultiChannelSegDataset(
            data_dir=r"C:\Users\yifei\Documents\data_for_publication\train_preprocessed",
            channels=['DAPI', 'FITC', 'TRITC'],
        )
        
        print(f"Dataset size: {len(dataset)} samples")
        
        # Test different batch sizes
        for batch_size in [1, 2, 4, 8, 16]:
            try:
                loader = DataLoader(
                    dataset, 
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=0,
                    drop_last=True
                )
                
                # Load one batch
                for i, batch in enumerate(loader):
                    if len(batch) == 3:
                        imgs, masks, ids = batch
                        print(f"Batch size {batch_size}: imgs shape = {imgs.shape}, masks shape = {masks.shape}")
                    else:
                        imgs, ids = batch
                        print(f"Batch size {batch_size}: imgs shape = {imgs.shape} (no masks)")
                    break
                    
            except Exception as e:
                print(f"Batch size {batch_size}: Failed - {str(e)}")
    
    except Exception as e:
        print(f"Dataset creation failed: {str(e)}")


if __name__ == "__main__":
    # Add this function to utils.py first
    def aggressive_cleanup():
        """More aggressive memory cleanup"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        for _ in range(3):
            gc.collect()
    
    print("GPU Memory Debugging Script")
    print("="*80)
    
    # Show initial state
    print("Initial GPU state:")
    monitor_gpu_memory()
    
    # Test models
    test_model_memory()
    
    # Test dataset
    test_dataset_loading()
    
    print("\nFinal GPU state:")
    monitor_gpu_memory()
