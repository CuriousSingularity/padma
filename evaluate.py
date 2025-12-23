#!/usr/bin/env python3
"""
Evaluation script for trained PyTorch models.

Usage:
    # Evaluate with checkpoint path
    python evaluate.py checkpoint_path=outputs/2024-01-01/12-00-00/checkpoints/best.pt

    # Evaluate with specific dataset
    python evaluate.py checkpoint_path=path/to/best.pt dataset.name=cifar100

    # Evaluate with custom data directory
    python evaluate.py checkpoint_path=path/to/best.pt dataset.data_dir=/path/to/data
"""

import json
from pathlib import Path
from typing import Dict, Any

import hydra
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from padma.models import create_model, load_checkpoint, get_model_info
from padma.datasets import create_dataset, create_dataloaders
from padma.utils import MetricsTracker, get_device


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
    use_amp: bool = True,
) -> Dict[str, float]:
    """
    Evaluate model on a dataset.

    Args:
        model: PyTorch model
        loader: Data loader
        criterion: Loss function
        device: Device to use
        num_classes: Number of classes
        use_amp: Whether to use mixed precision

    Returns:
        Dictionary of metrics
    """
    model.eval()
    metrics = MetricsTracker(num_classes=num_classes, device=device)

    all_preds = []
    all_targets = []

    for inputs, targets in tqdm(loader, desc="Evaluating"):
        inputs, targets = inputs.to(device), targets.to(device)

        if use_amp and device == "cuda":
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        metrics.update(outputs, targets, loss.item())

        # Store predictions for detailed analysis
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    results = metrics.compute()

    return results, all_preds, all_targets


def compute_per_class_metrics(
    preds: list, targets: list, num_classes: int
) -> Dict[str, Any]:
    """Compute per-class accuracy."""
    from collections import defaultdict

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


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main evaluation function."""
    # Check if checkpoint path is provided
    if not hasattr(cfg, "checkpoint_path") or cfg.checkpoint_path is None:
        print("Error: checkpoint_path is required")
        print("Usage: python evaluate.py checkpoint_path=path/to/checkpoint.pt")
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

    device = get_device(cfg)
    print(f"Device: {device}")

    # Load checkpoint to get training config
    print("\nLoading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Use checkpoint config if available, otherwise use current config
    if "config" in checkpoint:
        train_cfg = OmegaConf.create(checkpoint["config"])
        print("Using configuration from checkpoint")
        # Override dataset if specified
        if hasattr(cfg, "dataset"):
            train_cfg.dataset = cfg.dataset
    else:
        train_cfg = cfg
        print("Using current configuration")

    # Create model
    print("\nCreating model...")
    model = create_model(train_cfg)
    model_info = get_model_info(model)
    print(f"Model: {train_cfg.model.name}")
    print(f"Total parameters: {model_info['total_params']:,}")

    # Load model weights
    model = load_checkpoint(model, str(checkpoint_path), device)
    model = model.to(device)
    print("Model weights loaded successfully")

    # Create datasets
    print("\nCreating datasets...")
    train_dataset, val_dataset, test_dataset = create_dataset(train_cfg)
    print(f"Val samples: {len(val_dataset):,}")
    if test_dataset:
        print(f"Test samples: {len(test_dataset):,}")

    # Create data loaders
    _, val_loader, test_loader = create_dataloaders(
        train_cfg, train_dataset, val_dataset, test_dataset
    )

    # Create criterion
    criterion = nn.CrossEntropyLoss()

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Validation Set Results:")
    print("=" * 60)
    val_results, val_preds, val_targets = evaluate(
        model=model,
        loader=val_loader,
        criterion=criterion,
        device=device,
        num_classes=train_cfg.model.num_classes,
        use_amp=cfg.mixed_precision and device == "cuda",
    )

    for name, value in val_results.items():
        print(f"  {name}: {value:.4f}")

    # Evaluate on test set if available
    if test_loader is not None:
        print("\n" + "=" * 60)
        print("Test Set Results:")
        print("=" * 60)
        test_results, test_preds, test_targets = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            num_classes=train_cfg.model.num_classes,
            use_amp=cfg.mixed_precision and device == "cuda",
        )

        for name, value in test_results.items():
            print(f"  {name}: {value:.4f}")

        # Compute per-class metrics
        per_class = compute_per_class_metrics(
            test_preds, test_targets, train_cfg.model.num_classes
        )

        print("\nPer-class Accuracy:")
        for cls, acc in per_class.items():
            print(f"  {cls}: {acc:.4f}")

    # Save results
    results_path = checkpoint_path.parent / "evaluation_results.json"
    results = {
        "checkpoint": str(checkpoint_path),
        "validation": val_results,
    }
    if test_loader is not None:
        results["test"] = test_results
        results["per_class"] = per_class

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
