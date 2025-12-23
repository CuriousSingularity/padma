#!/usr/bin/env python3
"""
Training script for PyTorch models using Hydra configuration.

Usage:
    # Train with default config
    python train.py

    # Train with specific experiment
    python train.py +experiment=cifar10_resnet

    # Override specific parameters
    python train.py model.name=resnet101 training.epochs=100

    # Train with different dataset
    python train.py dataset.name=cifar100 model.num_classes=100
"""

import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from padma.models import create_model, get_model_info
from padma.datasets import create_dataset, create_dataloaders
from padma.trainers import Trainer


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Print configuration
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set random seed
    set_seed(cfg.seed)

    # Create output directories
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = Path(cfg.output_dir) / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"Configuration saved to: {config_path}")

    # Create model
    print("\nCreating model...")
    model = create_model(cfg)
    model_info = get_model_info(model)
    print(f"Model: {cfg.model.name}")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset, test_dataset = create_dataset(cfg)
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset):,}")

    # Create data loaders
    train_loader, val_loader, test_loader = create_dataloaders(
        cfg, train_dataset, val_dataset, test_dataset
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        test_loader=test_loader,
    )

    # Train
    print("\nStarting training...")
    results = trainer.train()

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best metrics: {results['best_metrics']}")
    print(f"Checkpoints saved to: {cfg.checkpoint.save_dir}")
    print(f"TensorBoard logs: {cfg.logging.tensorboard_dir}")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir={cfg.logging.tensorboard_dir}")

    # Evaluate on test set if available
    if test_loader is not None:
        print("\nEvaluating on test set...")
        test_metrics = trainer.validate(test_loader)
        print(f"Test metrics: {test_metrics}")

        # Log test metrics
        for name, value in test_metrics.items():
            trainer.writer.add_scalar(f"test/{name}", value, 0)


if __name__ == "__main__":
    main()
