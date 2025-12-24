#!/usr/bin/env python3
"""
Evaluation script for trained PyTorch Lightning models.

Usage:
    # Evaluate with checkpoint path
    python evaluate.py checkpoint_path=outputs/2024-01-01/12-00-00/checkpoints/best.ckpt

    # Evaluate with specific dataset
    python evaluate.py checkpoint_path=path/to/best.ckpt dataset.name=cifar100

    # Evaluate with custom data directory
    python evaluate.py checkpoint_path=path/to/best.ckpt dataset.data_dir=/path/to/data
"""

import json
import logging
from pathlib import Path

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from padma.models import get_model_info
from padma.trainers import ImageClassificationModule, ImageClassificationDataModule
from padma.utils import get_accelerator, compute_per_class_metrics

logger = logging.getLogger(__name__)


class EvaluationModule(L.LightningModule):
    """Extended module for detailed evaluation with per-class metrics."""

    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.criterion = torch.nn.CrossEntropyLoss()
        self.all_preds = []
        self.all_targets = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        preds = outputs.argmax(dim=1)
        self.all_preds.extend(preds.cpu().numpy().tolist())
        self.all_targets.extend(targets.cpu().numpy().tolist())

        return {"loss": loss, "preds": preds, "targets": targets}

    def on_test_epoch_end(self):
        """Compute final metrics."""
        per_class = compute_per_class_metrics(
            self.all_preds, self.all_targets, self.cfg.model.num_classes
        )

        # Compute overall accuracy
        correct = sum(1 for p, t in zip(self.all_preds, self.all_targets) if p == t)
        accuracy = correct / len(self.all_preds)

        self.log("test_accuracy", accuracy)

        # Store results for later access
        self.eval_results = {
            "accuracy": accuracy,
            "per_class": per_class,
            "total_samples": len(self.all_preds),
        }


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Check if checkpoint path is provided
    if not hasattr(cfg, "checkpoint_path") or cfg.checkpoint_path is None:
        logger.error("checkpoint_path is required")
        logger.info("Usage: python evaluate.py checkpoint_path=path/to/checkpoint.ckpt")
        return

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    logger.info("=" * 60)
    logger.info("Evaluation Configuration:")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Dataset: {cfg.dataset.name}")
    logger.info(f"Model: {cfg.model.model_name}")
    logger.info("=" * 60)

    # Load checkpoint to get training config
    logger.info("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Use checkpoint config if available (Lightning saves hyperparameters)
    if "hyper_parameters" in checkpoint and "cfg" in checkpoint["hyper_parameters"]:
        train_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])
        logger.info("Using configuration from checkpoint")
        # Override dataset if specified differently
        if hasattr(cfg, "dataset"):
            train_cfg.dataset = cfg.dataset
    else:
        train_cfg = cfg
        logger.info("Using current configuration")

    # Get accelerator
    accelerator = get_accelerator(cfg)
    logger.info(f"Accelerator: {accelerator}")

    # Create model
    logger.info("Creating model...")
    model_factory = instantiate(train_cfg.model)
    model = model_factory.create()
    model_info = get_model_info(model)
    logger.info(f"Model: {train_cfg.model.model_name}")
    logger.info(f"Total parameters: {model_info['total_params']:,}")

    # Load Lightning checkpoint
    lightning_module = ImageClassificationModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        cfg=train_cfg,
    )
    logger.info("Loaded Lightning checkpoint successfully")

    # Create evaluation module with the loaded model
    eval_module = EvaluationModule(model=lightning_module.model, cfg=train_cfg)

    # Create DataModule
    logger.info("Creating datasets...")
    datamodule = ImageClassificationDataModule(cfg=train_cfg)
    datamodule.setup("test")

    if datamodule.val_dataset:
        logger.info(f"Val samples: {len(datamodule.val_dataset):,}")
    if datamodule.test_dataset:
        logger.info(f"Test samples: {len(datamodule.test_dataset):,}")

    # Create trainer for evaluation
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    # Evaluate on validation set
    logger.info("=" * 60)
    logger.info("Validation Set Results:")
    logger.info("=" * 60)

    # Reset predictions
    eval_module.all_preds = []
    eval_module.all_targets = []

    # Use val_dataloader for validation
    trainer.test(eval_module, dataloaders=datamodule.val_dataloader())

    val_results = eval_module.eval_results
    logger.info(f"  Accuracy: {val_results['accuracy']:.4f}")
    logger.info(f"  Total samples: {val_results['total_samples']}")

    # Evaluate on test set if available
    test_results = None
    if datamodule.test_dataset is not None:
        logger.info("=" * 60)
        logger.info("Test Set Results:")
        logger.info("=" * 60)

        # Reset predictions
        eval_module.all_preds = []
        eval_module.all_targets = []

        trainer.test(eval_module, dataloaders=datamodule.test_dataloader())

        test_results = eval_module.eval_results
        logger.info(f"  Accuracy: {test_results['accuracy']:.4f}")
        logger.info(f"  Total samples: {test_results['total_samples']}")

        logger.info("Per-class Accuracy:")
        for cls, acc in test_results["per_class"].items():
            logger.info(f"  {cls}: {acc:.4f}")

    # Save results
    results_path = checkpoint_path.parent / "evaluation_results.json"
    results = {
        "checkpoint": str(checkpoint_path),
        "validation": {
            "accuracy": val_results["accuracy"],
            "total_samples": val_results["total_samples"],
        },
    }
    if test_results is not None:
        results["test"] = {
            "accuracy": test_results["accuracy"],
            "total_samples": test_results["total_samples"],
            "per_class": test_results["per_class"],
        }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Results saved to: {results_path}")


if __name__ == "__main__":
    main()
