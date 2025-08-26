# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
"""
Training script for TemporalDataset2D integration with PDEArena.
This script bypasses the torchdata.datapipes dependency.
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Timer
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from pdearena.data.temporal_dataset_pytorch import TemporalDataModule, TemporalDataset2DPyTorch
from pdearena.models.pderefiner import PDERefiner
from pdearena.data.utils import PDEDataConfig
from pdearena import utils

logger = utils.get_logger(__name__)


def setupdir(path):
    """Create necessary directories."""
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "tb"), exist_ok=True)
    os.makedirs(os.path.join(path, "ckpts"), exist_ok=True)


class TemporalLightningDataModule(pl.LightningDataModule):
    """Lightning DataModule wrapper for TemporalDataModule."""
    
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.datamodule = TemporalDataModule(**kwargs)
    
    def setup(self, stage=None):
        self.datamodule.setup(stage)
    
    def train_dataloader(self):
        return self.datamodule.train_dataloader()
    
    def val_dataloader(self):
        return self.datamodule.val_dataloader()
    
    def test_dataloader(self):
        return self.datamodule.test_dataloader()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_pde_config(data_config):
    """Create PDEDataConfig from configuration."""
    pde_config = data_config.get('pde', {})
    return PDEDataConfig(
        n_scalar_components=pde_config.get('n_scalar_components', 1),
        n_vector_components=pde_config.get('n_vector_components', 0),
        trajlen=pde_config.get('trajlen', 20),
        n_spatial_dim=pde_config.get('n_spatial_dim', 2)
    )


def main():
    parser = argparse.ArgumentParser(description="Train TemporalDataset2D with PDEArena")
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    parser.add_argument("--data.data_name", help="Override dataset name")
    parser.add_argument("--data.data_dir", help="Override data directory (for compatibility)")
    parser.add_argument("--trainer.devices", type=int, help="Number of devices to use")
    parser.add_argument("--trainer.max_epochs", type=int, help="Maximum epochs")
    parser.add_argument("--trainer.fast_dev_run", action="store_true", help="Fast development run")
    parser.add_argument("--data.normalize", action="store_true", help="Enable data normalization")
    parser.add_argument("--data.batch_size", type=int, help="Batch size")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data.data_name:
        config['data']['data_name'] = args.data.data_name
    if args.trainer.devices:
        config['trainer']['devices'] = args.trainer.devices
    if args.trainer.max_epochs:
        config['trainer']['max_epochs'] = args.trainer.max_epochs
    if args.trainer.fast_dev_run:
        config['trainer']['fast_dev_run'] = True
    if args.data.normalize:
        config['data']['normalize'] = True
    if args.data.batch_size:
        config['data']['batch_size'] = args.data.batch_size
    
    print("Configuration:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Setup output directory
    output_dir = config['trainer'].get('default_root_dir', './outputs')
    output_dir = os.environ.get('PDEARENA_OUTPUT_DIR', output_dir)
    setupdir(output_dir)
    
    logger.info(f"Checkpoints and logs will be saved in {output_dir}")
    
    # Create PDE configuration
    pde_config = create_pde_config(config['data'])
    
    # Create data module
    data_config = config['data'].copy()
    data_config.pop('pde', None)  # Remove pde section as it's handled separately
    datamodule = TemporalLightningDataModule(**data_config)
    
    # Create model
    model_config = config['model'].copy()
    model_config['pdeconfig'] = pde_config
    model = PDERefiner(**model_config)
    
    # Setup trainer
    trainer_config = config['trainer'].copy()
    
    # Setup logger
    logger_config = trainer_config.pop('logger', {})
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(output_dir, 'tb'),
        name=logger_config.get('init_args', {}).get('name'),
        version=logger_config.get('init_args', {}).get('version')
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(output_dir, 'ckpts'),
        filename='{epoch}-{step}-{valid/loss:.2f}',
        monitor='valid/loss',
        mode='min',
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    # Other callbacks
    callbacks.extend([
        LearningRateMonitor(logging_interval='step'),
        Timer(interval='epoch')
    ])
    
    # Create trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        default_root_dir=output_dir,
        **trainer_config
    )
    
    # Setup data
    datamodule.setup()
    
    # Start training
    logger.info("Starting training...")
    trainer.fit(model, datamodule=datamodule)
    
    # Test if not fast dev run
    if not trainer_config.get('fast_dev_run', False):
        logger.info("Starting testing...")
        trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    main()
