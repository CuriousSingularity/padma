#!/usr/bin/env python3
"""
Training script for PyTorch models using Hydra configuration and PyTorch Lightning.

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
import lightning as L
import numpy as np
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from padma.models import create_model, get_model_info
from padma.trainers import ImageClassificationModule, ImageClassificationDataModule


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    L.seed_everything(seed, workers=True)


def get_accelerator(cfg: DictConfig) -> str:
    """Get accelerator type based on configuration."""
    device = cfg.get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def get_precision(cfg: DictConfig, accelerator: str) -> str:
    """Get precision setting based on configuration and accelerator."""
    if cfg.get("mixed_precision", False):
        if accelerator == "cuda":
            return "16-mixed"
        elif accelerator == "mps":
            return "16-mixed"
    return "32-true"


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
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"Configuration saved to: {config_path}")

    # Create model
    print("\nCreating model...")
    model = create_model(cfg)
    model_info = get_model_info(model)
    print(f"Model: {cfg.model.name}")
    print(f"Total parameters: {model_info['total_params']:,}")
    print(f"Trainable parameters: {model_info['trainable_params']:,}")

    # Create Lightning module
    lightning_module = ImageClassificationModule(model=model, cfg=cfg)

    # Create DataModule
    print("\nCreating datasets...")
    datamodule = ImageClassificationDataModule(cfg=cfg)
    datamodule.setup("fit")
    dataset_info = datamodule.get_dataset_info()
    print(f"Dataset: {dataset_info['dataset_name']}")
    print(f"Train samples: {dataset_info['train_samples']:,}")
    print(f"Val samples: {dataset_info['val_samples']:,}")
    if "test_samples" in dataset_info:
        print(f"Test samples: {dataset_info['test_samples']:,}")

    # Get accelerator and precision settings
    accelerator = get_accelerator(cfg)
    precision = get_precision(cfg, accelerator)
    print(f"\nAccelerator: {accelerator}")
    print(f"Precision: {precision}")

    # Setup callbacks
    callbacks = []

    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.save_dir,
        filename="best-{epoch:02d}-{val_accuracy:.4f}",
        monitor=cfg.checkpoint.monitor_metric,
        mode=cfg.checkpoint.monitor_mode,
        save_top_k=1,
        save_last=cfg.checkpoint.save_last,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping callback
    if cfg.training.early_stopping.enabled:
        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor_metric,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.monitor_mode,
            verbose=True,
        )
        callbacks.append(early_stopping)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Progress bar
    callbacks.append(RichProgressBar())

    # Setup logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="tensorboard",
        version="",
    )

    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        accumulate_grad_batches=cfg.training.accumulation_steps,
        log_every_n_steps=cfg.logging.log_interval,
        deterministic=True,
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(lightning_module, datamodule=datamodule)

    # Print results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best {cfg.checkpoint.monitor_metric}: {checkpoint_callback.best_model_score:.4f}")
    print(f"Checkpoints saved to: {cfg.checkpoint.save_dir}")
    print(f"TensorBoard logs: {output_dir / 'tensorboard'}")
    print("\nTo view TensorBoard logs, run:")
    print(f"  tensorboard --logdir={output_dir / 'tensorboard'}")

    # Test on test set if available
    if datamodule.test_dataset is not None:
        print("\nEvaluating on test set...")
        trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
