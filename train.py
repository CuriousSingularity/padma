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
from hydra.utils import instantiate
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig, OmegaConf

from padma.models import get_model_info
from padma.trainers import ImageClassificationModule, ImageClassificationDataModule
from padma.utils import set_seed, get_accelerator, get_precision, create_callbacks

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
    model = instantiate(cfg.model)
    model_info = get_model_info(model)
    logger.info(f"Model: {cfg.model.model_name}")
    logger.info(f"Total parameters: {model_info['total_params']:,}")
    logger.info(f"Trainable parameters: {model_info['trainable_params']:,}")

    # Detect model type and use appropriate modules
    from padma.models.model_factory import ModelFactory
    is_timeseries = cfg.model.model_name.lower() in ModelFactory.TIMESERIES_MODELS

    if is_timeseries:
        from padma.trainers import TimeSeriesClassificationModule, TimeSeriesDataModule
        logger.info("Using time series classification modules")
        lightning_module = TimeSeriesClassificationModule(model=model, cfg=cfg)
        datamodule = TimeSeriesDataModule(cfg=cfg)
    else:
        logger.info("Using image classification modules")
        lightning_module = ImageClassificationModule(model=model, cfg=cfg)
        datamodule = ImageClassificationDataModule(cfg=cfg)

    # Create DataModule
    logger.info("Creating datasets...")
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

    # Setup callbacks from configuration
    callbacks = create_callbacks(cfg)

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

    # Get checkpoint callback if it was enabled
    checkpoint_callback = None
    for callback in callbacks:
        if isinstance(callback, ModelCheckpoint):
            checkpoint_callback = callback
            break

    if checkpoint_callback:
        logger.info(f"Best model checkpoint: {checkpoint_callback.best_model_path}")
        logger.info(f"Best {cfg.callbacks.model_checkpoint.monitor}: {checkpoint_callback.best_model_score:.4f}")
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
