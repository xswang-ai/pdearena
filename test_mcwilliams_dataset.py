#!/usr/bin/env python3
"""
Test script for McWilliams2DDataset
"""

import torch
from torch.utils.data import DataLoader
import sys
import os

# Add the current directory to Python path to import train_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_utils.datasets import McWilliams2DDataset

def test_mcwilliams_dataset():
    """Test the McWilliams2DDataset class"""
    
    # Test parameters
    file_path = "McWilliams2d_128x128_N1000_Re5000_T30.pt"  # Your data file
    data_res = (128, 128)      # Full resolution
    pde_res = (128, 128)       # Full resolution for PDE loss
    raw_res = (128, 128, 30)   # Raw data dimensions
    n_samples = 100            # Load 100 samples for testing
    t_duration = 1.0           # Use full time duration
    
    print("Testing McWilliams2DDataset...")
    print(f"File: {file_path}")
    print(f"Data resolution: {data_res}")
    print(f"PDE resolution: {pde_res}")
    print(f"Raw resolution: {raw_res}")
    print(f"Number of samples: {n_samples}")
    print(f"Time duration: {t_duration}")
    print("-" * 50)
    
    try:
        # Create dataset
        dataset = McWilliams2DDataset(
            file_path=file_path,
            data_res=data_res,
            pde_res=pde_res,
            raw_res=raw_res,
            n_samples=n_samples,
            t_duration=t_duration
        )
        
        print(f"Dataset created successfully!")
        print(f"Dataset length: {len(dataset)}")
        print("-" * 50)
        
        # Test getting a single sample
        print("Testing single sample retrieval...")
        trajectory, a_data = dataset[0]
        
        print(f"Trajectory shape: {trajectory.shape}")
        print(f"a_data shape: {a_data.shape}")
        print(f"Trajectory dtype: {trajectory.dtype}")
        print(f"a_data dtype: {a_data.dtype}")
        print("-" * 50)
        
        # Test DataLoader
        print("Testing DataLoader...")
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        for i, (batch_trajectory, batch_a_data) in enumerate(dataloader):
            print(f"Batch {i+1}:")
            print(f"  Batch trajectory shape: {batch_trajectory.shape}")
            print(f"  Batch a_data shape: {batch_a_data.shape}")
            
            if i >= 2:  # Test only first 3 batches
                break
        
        print("-" * 50)
        print("✅ All tests passed!")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ File not found: {file_path}")
        print("Please make sure the data file exists in the current directory.")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_different_resolutions():
    """Test with different resolutions"""
    
    file_path = "McWilliams2d_128x128_N1000_Re5000_T30.pt"
    
    print("\n" + "="*60)
    print("Testing different resolutions...")
    print("="*60)
    
    test_configs = [
        {
            'name': 'Full Resolution',
            'data_res': (128, 128),
            'pde_res': (128, 128),
            'raw_res': (128, 128, 30),
        },
        {
            'name': 'Half Resolution',
            'data_res': (64, 64),
            'pde_res': (64, 64),
            'raw_res': (128, 128, 30),
        },
        {
            'name': 'Quarter Resolution',
            'data_res': (32, 32),
            'pde_res': (32, 32),
            'raw_res': (128, 128, 30),
        },
    ]
    
    for config in test_configs:
        print(f"\n--- {config['name']} ---")
        
        try:
            dataset = McWilliams2DDataset(
                file_path=file_path,
                data_res=config['data_res'],
                pde_res=config['pde_res'],
                raw_res=config['raw_res'],
                n_samples=10,  # Small number for testing
                t_duration=1.0
            )
            
            trajectory, a_data = dataset[0]
            print(f"✅ Success: trajectory {trajectory.shape}, a_data {a_data.shape}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")

if __name__ == "__main__":
    print("McWilliams2D Dataset Test")
    print("=" * 60)
    
    # Test basic functionality
    success = test_mcwilliams_dataset()
    
    # Test different resolutions
    if success:
        test_different_resolutions()
    
    print("\n" + "="*60)
    print("Test completed!")
