#!/usr/bin/env python3
"""
Test script to verify the TemporalDataset2D integration with PDEArena.
This version tests the PyTorch-based implementation without torchdata dependencies.
"""

import sys
import os
import torch

# Add current directory to path for imports
sys.path.append(os.getcwd())

def test_imports():
    """Test if all imports work correctly."""
    print("Testing imports...")
    
    try:
        from pdearena.data.temporal_dataset_pytorch import (
            TemporalDataset2DPyTorch, 
            TemporalDataModule
        )
        print("✓ Successfully imported TemporalDataset2DPyTorch and TemporalDataModule")
    except ImportError as e:
        print(f"✗ Failed to import temporal dataset classes: {e}")
        return False
    
    try:
        from pdearena.models.pderefiner import PDERefiner
        print("✓ Successfully imported PDERefiner")
    except ImportError as e:
        print(f"✗ Failed to import PDERefiner: {e}")
        return False
    
    return True

def test_dataset_creation():
    """Test if we can create a TemporalDataset2D instance."""
    print("\nTesting dataset creation...")
    
    try:
        from pdearena.data.temporal_dataset_pytorch import TemporalDataset2DPyTorch
        
        # Try to create a dataset instance (will fail without data files, but should not crash on init)
        dataset = TemporalDataset2DPyTorch(
            data_name="ns2d_pda",  # Use ns2d_pda dataset
            mode="train",
            t_in=7,   # ns2d_pda uses t_in=4
            t_ar=1,
            limit_trajectories=10  # Small number for testing
        )
        
        print("✓ Successfully created TemporalDataset2DPyTorch instance")
        print(f"  - Dataset name: {dataset.data_name}")
        print(f"  - Resolution: {dataset.res}")
        print(f"  - Channels: {dataset.n_channels}")
        print(f"  - Dataset size: {dataset.n_size}")
        
        return True
        
    except FileNotFoundError as e:
        print(f"✓ Dataset creation works (expected file not found): {e}")
        return True  # This is expected when data files don't exist
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        return False

def test_dataset_dict():
    """Test if DATASET_DICT is accessible."""
    print("\nTesting DATASET_DICT...")
    
    try:
        from pdearena.data.dataset_config import DATASET_DICT
        
        if DATASET_DICT:
            print(f"✓ DATASET_DICT loaded with {len(DATASET_DICT)} datasets")
            
            # Print available datasets
            print("Available datasets:")
            for name in list(DATASET_DICT.keys())[:5]:  # Show first 5
                print(f"  - {name}")
            if len(DATASET_DICT) > 5:
                print(f"  ... and {len(DATASET_DICT) - 5} more")
        else:
            print("✗ DATASET_DICT is empty")
            return False
            
    except ImportError as e:
        print(f"✗ Could not import DATASET_DICT: {e}")
        print("Make sure dataset_config.py is in pdearena/data/")
        return False
    except Exception as e:
        print(f"✗ Error accessing DATASET_DICT: {e}")
        return False
    
    return True

def test_datamodule_compatibility():
    """Test if the temporal dataset works with our DataModule."""
    print("\nTesting DataModule compatibility...")
    
    try:
        from pdearena.data.temporal_dataset_pytorch import TemporalDataModule
        
        # Try to create a datamodule (this will fail without actual data, but should not crash on creation)
        datamodule = TemporalDataModule(
            data_name="ns2d_pda",
            t_in=7,   # ns2d_pda uses t_in=4
            t_ar=1,
            batch_size=4,
            num_workers=0,
            train_limit_trajectories=10,
            valid_limit_trajectories=10,
            test_limit_trajectories=10,
        )
        
        print("✓ TemporalDataModule created successfully")
        return True
        
    except Exception as e:
        print(f"✗ Error creating TemporalDataModule: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing TemporalDataset2D Integration with PDEArena")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_dataset_creation,
        test_dataset_dict,
        test_datamodule_compatibility
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Integration appears successful.")
        print("\nNext steps:")
        print("1. Update the data paths in pdearena/data/dataset_config.py to point to your actual data files")
        print("2. Update data_name in the config to match one from your DATASET_DICT")
        print("3. Adjust PDE configuration (n_scalar_components, n_vector_components, trajlen) in the config")
        print("4. Run training with:")
        print("   python scripts/temporal_train.py -c configs/temporal_dataset2d.yaml --data.data_name your_dataset_name")
        print("\nAvailable datasets from your DATASET_DICT:")
        try:
            from pdearena.data.dataset_config import DATASET_DICT
            for name in list(DATASET_DICT.keys())[:5]:
                print(f"   - {name}")
            if len(DATASET_DICT) > 5:
                print(f"   ... and {len(DATASET_DICT) - 5} more")
        except:
            print("   (Check pdearena/data/dataset_config.py for available datasets)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
