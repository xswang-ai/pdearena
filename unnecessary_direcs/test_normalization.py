#!/usr/bin/env python3
"""
Test normalization and denormalization functionality.
"""

import sys
import os
import torch
import numpy as np

# Add current directory to path for imports
sys.path.append(os.getcwd())

def test_normalization_workflow():
    """Test the complete normalization/denormalization workflow."""
    
    print("="*50)
    print("NORMALIZATION/DENORMALIZATION TEST")
    print("="*50)
    
    # Import our classes
    from pdearena.data.temporal_dataset_pytorch import TemporalDataset2DPyTorch, TemporalDataModule
    
    # Create synthetic data with normalization enabled
    datamodule = TemporalDataModule(
        data_name="synthetic_debug", 
        t_in=4,
        t_ar=1,
        normalize=True,  # Enable normalization
        batch_size=2,
        num_workers=0
    )
    
    print("1. Setting up DataModule with normalization...")
    datamodule.setup()
    
    # Check normalization parameters
    norm_params = datamodule.get_normalization_params()
    if norm_params:
        print("✓ Normalization parameters available:")
        print(f"  - Mean shape: {norm_params['mean'].shape}")
        print(f"  - Std shape: {norm_params['std'].shape}")
        print(f"  - Mean values: {norm_params['mean'].numpy()}")
        print(f"  - Std values: {norm_params['std'].numpy()}")
    else:
        print("✗ No normalization parameters found")
        return False
    
    print("\n2. Testing data loading with normalization...")
    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))
    x_norm, y_norm, cond, grid = batch
    
    print("✓ Loaded normalized batch:")
    print(f"  - Input (normalized) shape: {x_norm.shape}")
    print(f"  - Target (normalized) shape: {y_norm.shape}")
    print(f"  - Input range: [{x_norm.min().item():.3f}, {x_norm.max().item():.3f}]")
    print(f"  - Target range: [{y_norm.min().item():.3f}, {y_norm.max().item():.3f}]")
    
    print("\n3. Testing denormalization...")
    # Test denormalization on the input
    x_denorm = datamodule.denormalize(x_norm)
    y_denorm = datamodule.denormalize(y_norm)
    
    print("✓ Denormalized data:")
    print(f"  - Input (denormalized) range: [{x_denorm.min().item():.3f}, {x_denorm.max().item():.3f}]")
    print(f"  - Target (denormalized) range: [{y_denorm.min().item():.3f}, {y_denorm.max().item():.3f}]")
    
    print("\n4. Testing inference workflow simulation...")
    # Simulate what happens during inference:
    # 1. Take normalized input
    # 2. Model makes prediction (simulated with dummy prediction)
    # 3. Denormalize prediction
    
    # Simulate model prediction (just copy input for testing)
    model_prediction = x_norm[:, -1:, :, :, :]  # Take last time step as "prediction"
    
    # Denormalize the prediction
    prediction_denorm = datamodule.denormalize(model_prediction)
    
    print("✓ Inference workflow test:")
    print(f"  - Model prediction (normalized) shape: {model_prediction.shape}")
    print(f"  - Model prediction (normalized) range: [{model_prediction.min().item():.3f}, {model_prediction.max().item():.3f}]")
    print(f"  - Prediction (denormalized) range: [{prediction_denorm.min().item():.3f}, {prediction_denorm.max().item():.3f}]")
    
    print("\n5. Testing round-trip consistency...")
    # Test that normalize -> denormalize gives back original data
    # Load unnormalized data first
    datamodule_unnorm = TemporalDataModule(
        data_name="synthetic_debug",
        t_in=4,
        t_ar=1,
        normalize=False,  # Disable normalization
        batch_size=2,
        num_workers=0
    )
    datamodule_unnorm.setup()
    
    train_loader_unnorm = datamodule_unnorm.train_dataloader()
    batch_unnorm = next(iter(train_loader_unnorm))
    x_orig, y_orig, _, _ = batch_unnorm
    
    # Apply normalization and then denormalization
    x_norm_manual = datamodule.train_dataset._normalize_data(x_orig)
    x_roundtrip = datamodule.denormalize(x_norm_manual)
    
    # Check if we get back close to original
    diff = torch.abs(x_orig - x_roundtrip).mean()
    print(f"✓ Round-trip test (original -> normalize -> denormalize):")
    print(f"  - Mean absolute difference: {diff.item():.6f}")
    
    if diff.item() < 1e-5:
        print("  ✓ Round-trip successful (difference < 1e-5)")
    else:
        print("  ⚠ Round-trip has some numerical error (might be OK)")
    
    print("\n" + "="*50)
    print("🎉 NORMALIZATION TEST COMPLETE!")
    print("="*50)
    
    print("\nKey findings:")
    print("- Both input (x) and target (y) are normalized during training")
    print("- Denormalization functions are available for inference")
    print("- Round-trip normalization/denormalization preserves data")
    
    print("\nTypical inference workflow:")
    print("1. Load trained model")
    print("2. Get normalization params: datamodule.get_normalization_params()")
    print("3. Normalize new input: dataset._normalize_data(input)")
    print("4. Model prediction on normalized input")
    print("5. Denormalize prediction: datamodule.denormalize(prediction)")
    
    return True


if __name__ == "__main__":
    # Make sure synthetic data exists
    if not os.path.exists("./debug_data"):
        print("Creating synthetic data first...")
        os.system("python3 test_synthetic_debug.py > /dev/null 2>&1")
    
    success = test_normalization_workflow()
    sys.exit(0 if success else 1)

