# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
PyTorch Dataset implementation for TemporalDataset2D integration with PDEArena.
This approach avoids torchdata.datapipes dependencies.
"""

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from functools import partial
import sys
import os


def collate_fn_temporal(batch):
    """
    Custom collate function for temporal data that handles None values.
    """
    # Separate the components
    x_batch = []
    y_batch = []
    cond_batch = []
    grid_batch = []
    
    for x, y, cond, grid in batch:
        x_batch.append(x)
        y_batch.append(y)
        cond_batch.append(cond)
        grid_batch.append(grid)
    
    # Stack tensors
    x_stacked = torch.stack(x_batch, dim=0)
    y_stacked = torch.stack(y_batch, dim=0)
    cond_stacked = torch.stack(cond_batch, dim=0)
    
    # Handle grid (if all None, return None; otherwise stack)
    if all(grid is None for grid in grid_batch):
        grid_stacked = None
    else:
        # Convert None to zeros if needed
        grid_processed = []
        for grid in grid_batch:
            if grid is None:
                # Create dummy grid if needed
                grid_processed.append(torch.zeros(1))
            else:
                grid_processed.append(grid)
        grid_stacked = torch.stack(grid_processed, dim=0)
    
    return x_stacked, y_stacked, cond_stacked, grid_stacked

# Import the DATASET_DICT from dataset_config
try:
    from .dataset_config import DATASET_DICT
except ImportError:
    # Fallback: define a basic structure if import fails
    DATASET_DICT = {
        'ns2d_fno_1e-5': {
            'train_path': './data/large/ns2d_1e-5_train.hdf5',
            'test_path': './data/large/ns2d_1e-5_test.hdf5',
            'train_size': 1000,
            'test_size': 200,
            'scatter_storage': False,
            't_test': 10,
            't_in': 10,
            't_total': 20,
            'in_size': (64, 64),
            'n_channels': 1,
            'downsample': (1, 1)
        }
    }
    print("Warning: Could not import DATASET_DICT from dataset_config.py, using fallback")


class TemporalDataset2DPyTorch(Dataset):
    """
    PyTorch Dataset wrapper for TemporalDataset2D that integrates with PDEArena.
    
    This class adapts your original TemporalDataset2D logic to work with PDEArena's
    expected data format and structure.
    """
    
    def __init__(
        self,
        data_name: str = "ns2d_fno_1e-5",
        mode: str = "train", 
        n_train: int = None,
        t_in: int = 10,
        t_ar: int = 1,
        n_channels: int = None,
        normalize: bool = False,
        downsample: tuple = None,
        limit_trajectories: int = -1,
    ):
        """
        Initialize the temporal dataset.
        
        Args:
            data_name: Dataset name from DATASET_DICT
            mode: 'train', 'valid', or 'test'
            n_train: Number of trajectories to use (overrides dataset default)
            t_in: Number of input time steps
            t_ar: Number of autoregressive output steps
            n_channels: Number of channels (overrides dataset default)
            normalize: Whether to normalize data
            downsample: Downsample factor (overrides dataset default)
            limit_trajectories: Limit number of trajectories (-1 for no limit)
        """
        self.data_name = data_name
        self.mode = mode
        self.t_in = t_in
        self.t_ar = t_ar
        self.normalize = normalize
        
        # Validate dataset exists
        if data_name not in DATASET_DICT:
            available = list(DATASET_DICT.keys())[:5]  # Show first 5
            raise ValueError(f"Dataset {data_name} not found. Available: {available}...")
        
        # Get dataset configuration
        self.config = DATASET_DICT[data_name]
        
        # Set up dataset parameters
        self.res = self.config['in_size']
        self.n_channels = self.config['n_channels'] if n_channels is None else n_channels
        self.t_test = self.config['t_test']
        self.downsample = self.config['downsample'] if downsample is None else downsample
        
        # Determine dataset size
        if n_train is not None:
            self.n_size = n_train
        else:
            size_key = f'{mode}_size'
            if mode == 'valid':
                size_key = 'test_size'  # Use test data for validation
            self.n_size = self.config[size_key]
        
        # Apply trajectory limit
        if limit_trajectories > 0:
            self.n_size = min(self.n_size, limit_trajectories)
        
        self.train = (mode == 'train')
        
        print(f"Loading {mode} data for {data_name}")
        print(f"  - Trajectories: {self.n_size}")
        print(f"  - Resolution: {self.res}")
        print(f"  - Channels: {self.n_channels}")
        print(f"  - t_in: {t_in}, t_ar: {t_ar}")
        
        # Setup data loading
        self._setup_data_loading()
        
        # Setup normalization if requested
        if self.normalize:
            self._setup_normalization()
    
    def _setup_data_loading(self):
        """Setup data loading based on storage type."""
        if self.config['scatter_storage']:
            # Multiple HDF5 files
            def open_hdf5_file(path, idx):
                return h5py.File(f'{path}/data_{idx}.hdf5', 'r')['data'][:]
            
            path_key = f'{self.mode}_path' if self.mode != 'valid' else 'test_path'
            path = self.config[path_key]
            self.data_files = partial(open_hdf5_file, path)
        else:
            # Single HDF5 file
            path_key = f'{self.mode}_path' if self.mode != 'valid' else 'test_path'
            data_path = self.config[path_key]
            
            # Check if file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            self.data_files = h5py.File(data_path, 'r')
    
    def _setup_normalization(self):
        """Setup normalization parameters."""
        if 'normalizer_path' in self.config:
            print("Loading normalizer from saved path")
            normstat = torch.load(self.config['normalizer_path'])
            norm_mean = torch.cat([
                normstat["u"]["mean"].permute(1, 2, 0), 
                normstat["v"]["mean"].permute(1, 2, 0), 
                normstat["pres"]["mean"].unsqueeze(-1)
            ], dim=-1)
            norm_std = torch.cat([
                normstat["u"]["std"].permute(1, 2, 0), 
                normstat["v"]["std"].permute(1, 2, 0), 
                normstat["pres"]["std"].unsqueeze(-1)
            ], dim=-1)
            self.norm_mean = norm_mean
            self.norm_std = norm_std
        else:
            print("Computing normalization from training set sample")
            # Sample a few trajectories for normalization
            data_norm = []
            n_samples = min(100, self.n_size)
            for idx in range(n_samples):
                try:
                    if callable(self.data_files):
                        data = self.data_files(idx)
                    else:
                        data = self.data_files['data'][idx][:]
                    data_norm.append(torch.from_numpy(data).float())
                except:
                    break
            
            if data_norm:
                data_norm = torch.stack(data_norm)
                # Min-max normalization
                inp_min = torch.amin(data_norm, dim=(0, 1, 2, 3))
                inp_max = torch.amax(data_norm, dim=(0, 1, 2, 3))
                self.norm_mean = inp_min
                self.norm_std = (inp_max - inp_min)
                print(f"Normalization computed: min={inp_min.numpy()}, range={self.norm_std.numpy()}")
            else:
                self.norm_mean = torch.zeros(self.n_channels)
                self.norm_std = torch.ones(self.n_channels)
    
    def _downsample_x(self, u, N):
        """Downsample using FFT."""
        T, C, H, W = u.shape
        u_fft = torch.fft.rfft2(u, dim=(-2, -1))
        u_fft_crop = u_fft[..., :N//2+1, :N//2+1]
        u_downsampled = torch.fft.irfft2(u_fft_crop, s=(N, N), dim=(-2, -1))
        return u_downsampled * (N / H)**2
    
    def _normalize_data(self, x):
        """Apply normalization to data."""
        if hasattr(self, 'norm_mean') and hasattr(self, 'norm_std'):
            return (x - self.norm_mean.to(x.device)) / (self.norm_std.to(x.device))
        return x
    
    def _denormalize_data(self, x):
        """Apply denormalization to data."""
        if hasattr(self, 'norm_mean') and hasattr(self, 'norm_std'):
            return x * (self.norm_std.to(x.device)) + self.norm_mean.to(x.device)
        return x
    
    def denormalize(self, x):
        """
        Public method to denormalize data (for use during inference).
        
        Args:
            x: Tensor to denormalize
            
        Returns:
            Denormalized tensor
        """
        return self._denormalize_data(x)
    
    def get_normalization_params(self):
        """
        Get normalization parameters for external use.
        
        Returns:
            dict: Dictionary with 'mean' and 'std' if normalization is enabled, None otherwise
        """
        if hasattr(self, 'norm_mean') and hasattr(self, 'norm_std'):
            return {
                'mean': self.norm_mean,
                'std': self.norm_std
            }
        return None
    
    def _get_target_mask(self, sample, orig_size):
        """Get target mask for evaluation (from your original code)."""
        msk = torch.zeros_like(sample[..., 0:1, :])
        H, W = orig_size[:2]
        kx = max(1, H // self.res[0])
        ky = max(1, W // self.res[1])
        msk[::kx, ::ky, :, :orig_size[-1]] = 1
        return msk
    
    def __len__(self):
        return self.n_size
    
    def __getitem__(self, idx):
        """
        Get a data sample.
        
        Returns:
            Tuple of (x, y, cond, grid) where:
            - x: Input tensor [time_history, channels, height, width]
            - y: Target tensor [time_future, channels, height, width]  
            - cond: Conditioning parameters
            - grid: Spatial grid (None if not requested)
        """
        try:
            # Load data
            if callable(self.data_files):
                sample = self.data_files(idx)
            else:
                sample = self.data_files['data'][idx][:]
            
            sample = torch.from_numpy(sample).float()  # (H, W, T_all, C)
            
            # Handle channel selection (e.g., shallow water datasets)
            if sample.shape[-1] > self.n_channels:
                sample = sample[..., [0, 1, -1]]  # (u, v, pres)
            
            # Add channel dimension if needed
            if sample.ndim == 3:
                sample = sample.unsqueeze(-1)
            
            orig_size = list(sample.shape)
            
            # Training vs testing logic (from your original TemporalDataset2D)
            if self.train:
                # Random sampling for training
                max_start = max(sample.shape[-2] - (self.t_in + self.t_ar) + 1, 1)
                start_idx = np.random.randint(max_start)
                x = sample[..., start_idx:start_idx + self.t_in, :]
                y = sample[..., start_idx + self.t_in:min(start_idx + self.t_in + self.t_ar, sample.shape[-2]), :]
                msk = torch.ones([*x.shape[:2], 1, x.shape[-1]])
            else:
                # Fixed sampling for validation/test
                start_idx = 0
                x = sample[..., start_idx:start_idx + self.t_in, :]
                y = sample[..., self.t_in:self.t_in + self.t_test, :]
                msk = self._get_target_mask(sample, orig_size)
            
            # Apply normalization if enabled (normalize both input and target)
            if self.normalize:
                x = self._normalize_data(x)
                y = self._normalize_data(y)  # Also normalize target for consistency
            
            # Apply downsampling if needed
            if self.downsample != (1, 1):
                # Reshape for downsampling: (H, W, T, C) -> (T, C, H, W)
                x = x.permute(2, 3, 0, 1).contiguous()
                y = y.permute(2, 3, 0, 1).contiguous()
                target_size = sample.shape[0] // self.downsample[0]
                x = self._downsample_x(x, target_size)
                y = self._downsample_x(y, target_size)
                # Reshape back: (T, C, H, W) -> (H, W, T, C)
                x = x.permute(2, 3, 0, 1).contiguous()
                y = y.permute(2, 3, 0, 1).contiguous()
            
            # Convert to PDEArena format: (H, W, T, C) -> (T, C, H, W)
            x = x.permute(2, 3, 0, 1)  # (T, C, H, W)
            y = y.permute(2, 3, 0, 1)  # (T, C, H, W)
            
            # Create conditioning parameters (dummy for compatibility)
            cond = torch.tensor([1.0, 1.0, 1.0])[None]  # [batch_size=1, n_params]
            
            # No grid for simplicity (can be added if needed)
            grid = None
            
            return x, y, cond, grid
            
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            # Return a dummy sample to avoid crashing
            dummy_shape = (self.t_in, self.n_channels, *self.res)
            x = torch.zeros(dummy_shape)
            y = torch.zeros((self.t_ar, self.n_channels, *self.res))
            cond = torch.tensor([1.0, 1.0, 1.0])[None]
            grid = None  # Explicitly set grid to None
            return x, y, cond, grid


class TemporalDataModule:
    """
    Simple DataModule for TemporalDataset2D that works with PDEArena.
    
    This is a simplified version that doesn't inherit from LightningDataModule
    but provides the same interface.
    """
    
    def __init__(
        self,
        data_name: str = "ns2d_fno_1e-5",
        data_dir: str = None,  # Not used but kept for compatibility
        t_in: int = 10,
        t_ar: int = 1,
        n_channels: int = None,
        normalize: bool = False,
        batch_size: int = 32,
        num_workers: int = 1,
        pin_memory: bool = True,
        train_limit_trajectories: int = -1,
        valid_limit_trajectories: int = -1,
        test_limit_trajectories: int = -1,
        **kwargs
    ):
        self.data_name = data_name
        self.t_in = t_in
        self.t_ar = t_ar
        self.n_channels = n_channels
        self.normalize = normalize
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_limit_trajectories = train_limit_trajectories
        self.valid_limit_trajectories = valid_limit_trajectories
        self.test_limit_trajectories = test_limit_trajectories
        
        # Store other kwargs for potential use
        self.kwargs = kwargs
    
    def setup(self, stage=None):
        """Setup datasets for each split."""
        self.train_dataset = TemporalDataset2DPyTorch(
            data_name=self.data_name,
            mode="train",
            t_in=self.t_in,
            t_ar=self.t_ar,
            n_channels=self.n_channels,
            normalize=self.normalize,
            limit_trajectories=self.train_limit_trajectories
        )
        
        self.val_dataset = TemporalDataset2DPyTorch(
            data_name=self.data_name,
            mode="valid",
            t_in=self.t_in,
            t_ar=self.t_ar,
            n_channels=self.n_channels,
            normalize=self.normalize,
            limit_trajectories=self.valid_limit_trajectories
        )
        
        self.test_dataset = TemporalDataset2DPyTorch(
            data_name=self.data_name,
            mode="test",
            t_in=self.t_in,
            t_ar=self.t_ar,
            n_channels=self.n_channels,
            normalize=self.normalize,
            limit_trajectories=self.test_limit_trajectories
        )
    
    def train_dataloader(self):
        """Return training dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            collate_fn=collate_fn_temporal
        )
    
    def val_dataloader(self):
        """Return validation dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_temporal
        )
    
    def test_dataloader(self):
        """Return test dataloader."""
        from torch.utils.data import DataLoader
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=collate_fn_temporal
        )
    
    def get_normalization_params(self):
        """
        Get normalization parameters from the training dataset.
        
        Returns:
            dict: Normalization parameters or None if not available
        """
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.get_normalization_params()
        return None
    
    def denormalize(self, x):
        """
        Denormalize data using training dataset parameters.
        
        Args:
            x: Tensor to denormalize
            
        Returns:
            Denormalized tensor
        """
        if hasattr(self, 'train_dataset'):
            return self.train_dataset.denormalize(x)
        return x


class TemporalLightningDataModule:
    """
    PyTorch Lightning DataModule wrapper for TemporalDataModule.
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.datamodule = TemporalDataModule(**kwargs)
    
    def setup(self, stage=None):
        self.datamodule.setup(stage)
    
    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()
    
    def test_dataloader(self):
        return self.datamodule.test_dataloader()
