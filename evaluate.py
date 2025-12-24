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
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

from padma.models import create_model, get_model_info
from padma.trainers import ImageClassificationModule, ImageClassificationDataModule


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


def compute_per_class_metrics(
    preds: list, targets: list, num_classes: int
) -> Dict[str, Any]:
    """Compute per-class accuracy."""
    correct = defaultdict(int)
    total = defaultdict(int)

    for pred, target in zip(preds, targets):
        total[target] += 1
        if pred == target:
            correct[target] += 1

    per_class_acc = {}
    for cls in range(num_classes):
        if total[cls] > 0:
            per_class_acc[f"class_{cls}"] = correct[cls] / total[cls]
        else:
            per_class_acc[f"class_{cls}"] = 0.0

    return per_class_acc


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
    # Check if checkpoint path is provided
    if not hasattr(cfg, "checkpoint_path") or cfg.checkpoint_path is None:
        print("Error: checkpoint_path is required")
        print("Usage: python evaluate.py checkpoint_path=path/to/checkpoint.ckpt")
        return

    checkpoint_path = Path(cfg.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return

    print("=" * 60)
    print("Evaluation Configuration:")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Dataset: {cfg.dataset.name}")
    print(f"Model: {cfg.model.name}")
    print("=" * 60)

    # Load checkpoint to get training config
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Use checkpoint config if available (Lightning saves hyperparameters)
    if "hyper_parameters" in checkpoint and "cfg" in checkpoint["hyper_parameters"]:
        train_cfg = OmegaConf.create(checkpoint["hyper_parameters"]["cfg"])
        print("Using configuration from checkpoint")
        # Override dataset if specified differently
        if hasattr(cfg, "dataset"):
            train_cfg.dataset = cfg.dataset
    else:
        train_cfg = cfg
        print("Using current configuration")

    # Get accelerator
    accelerator = get_accelerator(cfg)
    print(f"Accelerator: {accelerator}")

    # Create model
    print("\nCreating model...")
    model = create_model(train_cfg)
    model_info = get_model_info(model)
    print(f"Model: {train_cfg.model.name}")
    print(f"Total parameters: {model_info['total_params']:,}")

    # Load Lightning checkpoint
    lightning_module = ImageClassificationModule.load_from_checkpoint(
        checkpoint_path,
        model=model,
        cfg=train_cfg,
    )
    print("Loaded Lightning checkpoint successfully")

    # Create evaluation module with the loaded model
    eval_module = EvaluationModule(model=lightning_module.model, cfg=train_cfg)

    # Create DataModule
    print("\nCreating datasets...")
    datamodule = ImageClassificationDataModule(cfg=train_cfg)
    datamodule.setup("test")

    if datamodule.val_dataset:
        print(f"Val samples: {len(datamodule.val_dataset):,}")
    if datamodule.test_dataset:
        print(f"Test samples: {len(datamodule.test_dataset):,}")

    # Create trainer for evaluation
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_progress_bar=True,
    )

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Validation Set Results:")
    print("=" * 60)

    # Reset predictions
    eval_module.all_preds = []
    eval_module.all_targets = []

    # Use val_dataloader for validation
    trainer.test(eval_module, dataloaders=datamodule.val_dataloader())

    val_results = eval_module.eval_results
    print(f"  Accuracy: {val_results['accuracy']:.4f}")
    print(f"  Total samples: {val_results['total_samples']}")

    # Evaluate on test set if available
    test_results = None
    if datamodule.test_dataset is not None:
        print("\n" + "=" * 60)
        print("Test Set Results:")
        print("=" * 60)

        # Reset predictions
        eval_module.all_preds = []
        eval_module.all_targets = []

        trainer.test(eval_module, dataloaders=datamodule.test_dataloader())

        test_results = eval_module.eval_results
        print(f"  Accuracy: {test_results['accuracy']:.4f}")
        print(f"  Total samples: {test_results['total_samples']}")

        print("\nPer-class Accuracy:")
        for cls, acc in test_results["per_class"].items():
            print(f"  {cls}: {acc:.4f}")

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

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
