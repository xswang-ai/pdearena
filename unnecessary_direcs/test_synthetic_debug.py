#!/usr/bin/env python3
"""
Quick debugging test with synthetic data.
This bypasses the config files and tests everything directly.
"""

import sys
import os
import torch
import numpy as np
import h5py
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.getcwd())

def create_minimal_synthetic_data(output_dir="./debug_data", n_samples=3):
    """Create minimal synthetic data for quick testing."""
    
    print(f"Creating minimal synthetic data in {output_dir}")
    
    # Create directories
    train_dir = Path(output_dir) / "train"
    test_dir = Path(output_dir) / "test"
    
    for dir_path in [train_dir, test_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    # Create simple synthetic data
    H, W, T, C = 32, 32, 14, 3  # Small for fast testing
    
    for split, split_dir in [("train", train_dir), ("test", test_dir)]:
        for i in range(n_samples):
            # Create simple patterns
            data = np.zeros((H, W, T, C), dtype=np.float32)
            
            for t in range(T):
                # Simple time-varying patterns
                x = np.linspace(0, 2*np.pi, W)
                y = np.linspace(0, 2*np.pi, H)
                X, Y = np.meshgrid(x, y, indexing='ij')
                
                # Channel 0: u velocity
                data[:, :, t, 0] = np.sin(X + t/T) * np.cos(Y)
                # Channel 1: v velocity  
                data[:, :, t, 1] = np.cos(X) * np.sin(Y + t/T)
                # Channel 2: pressure
                data[:, :, t, 2] = np.sin(X + Y + t/T)
            
            # Save as HDF5
            filepath = split_dir / f"data_{i}.hdf5"
            with h5py.File(filepath, 'w') as f:
                f.create_dataset('data', data=data)
        
        print(f"  Created {n_samples} {split} samples")
    
    return str(output_dir)

def test_synthetic_data():
    """Test the integration with synthetic data."""
    
    print("="*50)
    print("SYNTHETIC DATA DEBUG TEST")
    print("="*50)
    
    # Create synthetic data
    data_path = create_minimal_synthetic_data()
    abs_path = os.path.abspath(data_path)
    
    # Import our classes
    try:
        from pdearena.data.temporal_dataset_pytorch import TemporalDataset2DPyTorch, TemporalDataModule
        print("✓ Successfully imported temporal dataset classes")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Create a custom dataset config for synthetic data
    synthetic_config = {
        'train_path': f'{abs_path}/train',
        'test_path': f'{abs_path}/test',
        'train_size': 3,
        'test_size': 3,
        'scatter_storage': True,
        't_test': 5,
        't_in': 4,
        't_total': 14,
        'in_size': (32, 32),
        'n_channels': 3,
        'downsample': (1, 1)
    }
    
    # Temporarily add to DATASET_DICT
    from pdearena.data.dataset_config import DATASET_DICT
    DATASET_DICT['synthetic_debug'] = synthetic_config
    
    print("\n1. Testing dataset creation...")
    try:
        dataset = TemporalDataset2DPyTorch(
            data_name="synthetic_debug",
            mode="train",
            t_in=4,
            t_ar=1,
            limit_trajectories=3
        )
        print("✓ Dataset created successfully")
        print(f"  - Shape: {dataset.res}")
        print(f"  - Channels: {dataset.n_channels}")
        print(f"  - Size: {len(dataset)}")
    except Exception as e:
        print(f"✗ Dataset creation failed: {e}")
        return False
    
    print("\n2. Testing data loading...")
    try:
        x, y, cond, grid = dataset[0]
        print("✓ Data loading successful")
        print(f"  - Input shape: {x.shape}")   # Should be (t_in, channels, H, W)
        print(f"  - Output shape: {y.shape}")  # Should be (t_ar, channels, H, W)
        print(f"  - Condition shape: {cond.shape}")
        print(f"  - Grid: {grid is not None}")
    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        return False
    
    print("\n3. Testing DataModule...")
    try:
        datamodule = TemporalDataModule(
            data_name="synthetic_debug",
            t_in=4,
            t_ar=1,
            batch_size=2,
            num_workers=0
        )
        datamodule.setup()
        print("✓ DataModule setup successful")
        
        # Test dataloader
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        print("✓ DataLoader working")
        print(f"  - Batch input shape: {batch[0].shape}")   # (batch, t_in, channels, H, W)
        print(f"  - Batch output shape: {batch[1].shape}")  # (batch, t_ar, channels, H, W)
        
    except Exception as e:
        print(f"✗ DataModule test failed: {e}")
        return False
    
    print("\n4. Testing with PyTorch Lightning...")

    import pytorch_lightning as pl
    from pdearena.data.temporal_dataset_pytorch import TemporalLightningDataModule
    
    # Create Lightning DataModule
    lightning_dm = TemporalLightningDataModule(
        data_name="synthetic_debug",
        t_in=4,
        t_ar=1,
        batch_size=2,
        num_workers=0
    )
    lightning_dm.setup()
    
    print("✓ PyTorch Lightning DataModule working")


    from pdearena.data.utils import PDEDataConfig
    from pdearena.models.pderefiner import PDERefiner
        
    pde_config = PDEDataConfig(
        n_scalar_components=1,
        n_vector_components=2,
        trajlen=14,
        n_spatial_dim=2
    )
    
    model = PDERefiner(
        name="Unetmod-64",
        param_conditioning="scalar",
        time_history=4,
        time_future=1,
        time_gap=0,
        max_num_steps=10,
        activation="gelu",
        criterion="mse",
        lr=1e-4,
        pdeconfig=pde_config,
        predict_difference=True,
        difference_weight=0.3,
        min_noise_std=4e-7,
        num_refinement_steps=3
    )
    
    print("✓ PDERefiner model created successfully")
    
    # Test a forward pass
    sample_batch = next(iter(lightning_dm.train_dataloader()))
    x, y, cond, grid = sample_batch
    
    # Format for model
    with torch.no_grad():
        pred = model.predict_next_solution(x)
        print("✓ Model forward pass successful")
        print(f"  - Prediction shape: {pred.shape}")
        
        # Quick training test (if PDERefiner available)
        # try:
           
            
        # except ImportError as e:
        #     print(f"⚠ PDERefiner test skipped (missing dependencies): {e}")
        # except Exception as e:
        #     print(f"⚠ PDERefiner test failed: {e}")
    
    
    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED!")
    print("="*50)
    
    print(f"\nSynthetic data created in: {abs_path}")
    print("\nYou can now run full training with:")
    print(f"python scripts/temporal_train.py -c configs/temporal_dataset2d.yaml \\")
    print(f"    --data.data_name synthetic_debug \\")
    print(f"    --trainer.fast_dev_run")
    
    return True

if __name__ == "__main__":
    success = test_synthetic_data()
    sys.exit(0 if success else 1)
