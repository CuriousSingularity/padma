#!/usr/bin/env python3
"""
Combined training and evaluation script for PyTorch models using Hydra configuration.

This script combines the functionality of train.py and evaluate.py:
1. Trains a model on train/val sets (using train.py)
2. Evaluates the best checkpoint on the test set (using evaluate.py)

Usage:
    # Train and evaluate with default config
    python train_and_evaluate.py

    # Train and evaluate with specific experiment
    python train_and_evaluate.py +experiment=cifar10_resnet

    # Override specific parameters
    python train_and_evaluate.py model.name=resnet101 training.epochs=100

    # Train and evaluate with different dataset
    python train_and_evaluate.py dataset.name=cifar100 model.num_classes=100
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

# Import the core training and evaluation functions
from train import train
from evaluate import evaluate

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main function that orchestrates training and evaluation.

    Args:
        cfg: Hydra configuration
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logger.info("=" * 60)
    logger.info("TRAIN AND EVALUATE PIPELINE")
    logger.info("=" * 60)
    logger.info("Step 1: Training model on train/val sets")
    logger.info("Step 2: Evaluating best checkpoint on test set")
    logger.info("=" * 60)

    # Step 1: Train the model
    logger.info("\n" + "=" * 60)
    logger.info("STEP 1: TRAINING")
    logger.info("=" * 60)

    checkpoint_path = train(cfg)

    if checkpoint_path is None:
        logger.error("Training failed to produce a checkpoint. Aborting evaluation.")
        return

    logger.info("=" * 60)
    logger.info(f"Training completed. Best checkpoint: {checkpoint_path}")
    logger.info("=" * 60)

    # Step 2: Evaluate the best checkpoint on test set
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: EVALUATION ON TEST SET")
    logger.info("=" * 60)

    # Create a copy of the config and add the checkpoint path
    eval_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    eval_cfg.checkpoint_path = checkpoint_path

    # Run evaluation
    results = evaluate(eval_cfg)

    # Log final summary
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Training checkpoint: {checkpoint_path}")
    logger.info(f"Test accuracy: {results['test']['accuracy']:.4f}")
    logger.info(f"Test samples: {results['test']['total_samples']}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
