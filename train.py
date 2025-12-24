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

import logging
from pathlib import Path

import hydra
import lightning as L
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
from padma.utils import set_seed, get_accelerator, get_precision

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Log configuration
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("=" * 60)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logger.info("=" * 60)

    # Set random seed
    set_seed(cfg.seed)

    # Create output directories
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Configuration saved to: {config_path}")

    # Create model
    logger.info("Creating model...")
    model = create_model(cfg)
    model_info = get_model_info(model)
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Total parameters: {model_info['total_params']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_params']:,}")

    # Create Lightning module
    lightning_module = ImageClassificationModule(model=model, cfg=cfg)

    # Create DataModule
    logger.info("Creating datasets...")
    datamodule = ImageClassificationDataModule(cfg=cfg)
    datamodule.setup("fit")
    dataset_info = datamodule.get_dataset_info()
    logger.info(f"Dataset: {dataset_info['dataset_name']}")
    logger.info(f"Train samples: {dataset_info['train_samples']:,}")
    logger.info(f"Val samples: {dataset_info['val_samples']:,}")
    if "test_samples" in dataset_info:
        logger.info(f"Test samples: {dataset_info['test_samples']:,}")

    # Get accelerator and precision settings
    accelerator = get_accelerator(cfg)
    precision = get_precision(cfg, accelerator)
    logger.info(f"Accelerator: {accelerator}")
    logger.info(f"Precision: {precision}")

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
    logger.info("Starting training...")
    trainer.fit(lightning_module, datamodule=datamodule)

    # Log results
    logger.info("=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
    logger.info(f"Best {cfg.checkpoint.monitor_metric}: {checkpoint_callback.best_model_score:.4f}")
    logger.info(f"Checkpoints saved to: {cfg.checkpoint.save_dir}")
    logger.info(f"TensorBoard logs: {output_dir / 'tensorboard'}")
    logger.info("To view TensorBoard logs, run:")
    logger.info(f"  tensorboard --logdir={output_dir / 'tensorboard'}")

    # Test on test set if available
    if datamodule.test_dataset is not None:
        logger.info("Evaluating on test set...")
        trainer.test(lightning_module, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
