#!/usr/bin/env python3
"""
Usage example for McWilliams2DDataset
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_utils.datasets import McWilliams2DDataset

def main():
    """Example usage of McWilliams2DDataset"""
    
    # Configuration
    config = {
        'file_path': 'McWilliams2d_128x128_N1000_Re5000_T30.pt',
        'data_res': (128, 128),      # Spatial resolution for training data
        'pde_res': (128, 128),       # Spatial resolution for PDE loss
        'raw_res': (128, 128, 30),   # Raw data dimensions (H, W, T)
        'n_samples': 800,             # Number of training samples
        'batch_size': 8,              # Batch size
        't_duration': 1.0,            # Use full time duration
    }
    
    print("McWilliams2D Dataset Usage Example")
    print("=" * 50)
    
    # Create training dataset
    print("Creating training dataset...")
    train_dataset = McWilliams2DDataset(
        file_path=config['file_path'],
        data_res=config['data_res'],
        pde_res=config['pde_res'],
        raw_res=config['raw_res'],
        n_samples=config['n_samples'],
        offset=0,                     # Start from beginning
        t_duration=config['t_duration']
    )
    
    # Create validation dataset (remaining samples)
    print("Creating validation dataset...")
    val_dataset = McWilliams2DDataset(
        file_path=config['file_path'],
        data_res=config['data_res'],
        pde_res=config['pde_res'],
        raw_res=config['raw_res'],
        n_samples=200,                # Validation samples
        offset=800,                   # Start after training samples
        t_duration=config['t_duration']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Example training loop
    print("\nExample training loop:")
    print("-" * 30)
    
    for epoch in range(2):  # Just 2 epochs for example
        print(f"Epoch {epoch + 1}")
        
        # Training
        for batch_idx, (trajectory, a_data) in enumerate(train_loader):
            # trajectory: (batch_size, T, H, W, 2) - velocity field over time
            # a_data: (batch_size, H, W, T, 5) - grid coords + initial conditions
            
            if batch_idx == 0:  # Print shapes for first batch
                print(f"  Batch {batch_idx}: trajectory {trajectory.shape}, a_data {a_data.shape}")
            
            # Your model training code would go here
            # model_output = your_model(a_data)
            # loss = your_loss_function(model_output, trajectory)
            # loss.backward()
            # optimizer.step()
            
            if batch_idx >= 2:  # Just show first few batches
                break
        
        # Validation
        val_loss = 0.0
        for batch_idx, (trajectory, a_data) in enumerate(val_loader):
            # Your validation code would go here
            # model_output = your_model(a_data)
            # loss = your_loss_function(model_output, trajectory)
            # val_loss += loss.item()
            
            if batch_idx >= 1:  # Just show first batch
                break
        
        print(f"  Validation loss: {val_loss:.6f}")
    
    print("\nExample completed!")

def example_with_different_resolutions():
    """Example using different resolutions for memory efficiency"""
    
    print("\n" + "="*60)
    print("Example with Different Resolutions")
    print("="*60)
    
    # Low resolution for faster training
    low_res_dataset = McWilliams2DDataset(
        file_path='McWilliams2d_128x128_N1000_Re5000_T30.pt',
        data_res=(64, 64),       # Half resolution
        pde_res=(32, 32),        # Even lower resolution for PDE loss
        raw_res=(128, 128, 30),
        n_samples=100,
        t_duration=0.5,          # Half time duration
    )
    
    trajectory, a_data = low_res_dataset[0]
    print(f"Low resolution - trajectory: {trajectory.shape}, a_data: {a_data.shape}")
    
    # High resolution for final evaluation
    high_res_dataset = McWilliams2DDataset(
        file_path='McWilliams2d_128x128_N1000_Re5000_T30.pt',
        data_res=(128, 128),     # Full resolution
        pde_res=(128, 128),      # Full resolution
        raw_res=(128, 128, 30),
        n_samples=10,
        t_duration=1.0,          # Full time duration
    )
    
    trajectory, a_data = high_res_dataset[0]
    print(f"High resolution - trajectory: {trajectory.shape}, a_data: {a_data.shape}")

if __name__ == "__main__":
    try:
        main()
        example_with_different_resolutions()
    except FileNotFoundError:
        print("❌ Data file not found. Please make sure 'McWilliams2d_128x128_N1000_Re5000_T30.pt' exists.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
