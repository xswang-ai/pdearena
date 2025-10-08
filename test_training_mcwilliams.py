#!/usr/bin/env python3
"""
Test script to verify training works with McWilliams2D dataset
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from train_utils.datasets import McWilliams2DDataset
from models import FNO3d

def test_dataset_integration():
    """Test that the dataset works with the training framework"""
    
    print("Testing McWilliams2D dataset integration...")
    
    # Test parameters (matching config file)
    config = {
        'file_path': '/scratch3/wan410/operator_learning_data/NS_torchcfd/data/McWilliams2d_128x128_N1000_Re5000_T30.pt',
        'raw_res': [128, 128, 30],
        'data_res': [128, 128, 30],
        'pde_res': [128, 128, 30],
        'n_samples': 10,  # Small number for testing
        'offset': 0,
        't_duration': 1.0
    }
    
    try:
        # Create dataset
        dataset = McWilliams2DDataset(**config)
        print(f"✅ Dataset created successfully with {len(dataset)} samples")
        
        # Test getting a sample
        trajectory, a_data = dataset[0]
        print(f"✅ Sample retrieved - trajectory: {trajectory.shape}, a_data: {a_data.shape}")
        
        # Test batch loading
        from torch.utils.data import DataLoader
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        for batch_idx, (batch_trajectory, batch_a_data) in enumerate(dataloader):
            print(f"✅ Batch {batch_idx}: trajectory {batch_trajectory.shape}, a_data {batch_a_data.shape}")
            break
        
        # Test FNO3d model compatibility
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = FNO3d(modes1=[12, 12, 12, 12],
                      modes2=[12, 12, 12, 12], 
                      modes3=[12, 12, 12, 12],
                      fc_dim=128,
                      layers=[64, 64, 64, 64, 64],
                      act='gelu',
                      pad_ratio=[0, 0.125]).to(device)
        
        # Test model forward pass
        batch_trajectory, batch_a_data = next(iter(dataloader))
        batch_trajectory = batch_trajectory.to(device)
        batch_a_data = batch_a_data.to(device)
        
        with torch.no_grad():
            output = model(batch_a_data)
            print(f"✅ Model forward pass successful - output: {output.shape}")
        
        print("\n🎉 All tests passed! Training script should work with McWilliams2D dataset.")
        return True
        
    except FileNotFoundError:
        print("❌ Data file not found. Please make sure 'McWilliams2d_128x128_N1000_Re5000_T30.pt' exists.")
        return False
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_training_script_args():
    """Test the training script with McWilliams2D"""
    
    print("\n" + "="*60)
    print("Testing training script integration...")
    print("="*60)
    
    # Test config file parsing
    import yaml
    
    try:
        with open('config/FNO_Re1k.yaml', 'r') as f:
            config = yaml.load(f, yaml.FullLoader)
        
        print(f"✅ Config loaded - dataset name: {config['data']['name']}")
        print(f"✅ Data path: {config['data']['paths'][0]}")
        print(f"✅ Raw resolution: {config['data']['raw_res']}")
        print(f"✅ Training samples: {config['data']['n_data_samples']}")
        print(f"✅ Test samples: {config['data']['n_test_samples']}")
        
        # Test dataset creation with config
        dataset_config = {
            'file_path': config['data']['paths'][0],
            'raw_res': tuple(config['data']['raw_res']),
            'data_res': tuple(config['data']['data_res']),
            'pde_res': tuple(config['data']['pde_res']),
            'n_samples': config['data']['n_data_samples'],
            'offset': config['data']['offset'],
            't_duration': config['data']['t_duration']
        }
        
        dataset = McWilliams2DDataset(**dataset_config)
        print(f"✅ Dataset created with config - {len(dataset)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

if __name__ == "__main__":
    print("McWilliams2D Training Integration Test")
    print("="*60)
    
    # Test basic dataset integration
    success1 = test_dataset_integration()
    
    # Test training script integration
    success2 = test_training_script_args()
    
    if success1 and success2:
        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("You can now run training with:")
        print("python train_no.py --config config/FNO_Re1k.yaml")
        print("="*60)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
