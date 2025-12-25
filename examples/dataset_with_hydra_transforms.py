"""Example showing how to integrate Hydra transforms with dataset factories.

This demonstrates two approaches:
1. Using Hydra instantiation for transforms (new approach)
2. Integrating with existing dataset factory pattern

The key benefit is that all transform parameters are now defined in YAML
and can be overridden via Hydra's CLI without changing code.
"""

from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig

from padma.datasets.base import split_dataset
from padma.utils.transforms import create_transforms_from_config


def create_cifar10_with_hydra_transforms(
    cfg: DictConfig,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create CIFAR-10 datasets using Hydra-instantiated transforms.

    This version loads transforms directly from YAML config using
    hydra.utils.instantiate, providing maximum flexibility.

    Args:
        cfg: Full Hydra configuration with transformation.train_transforms
             and transformation.val_transforms

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)

    Example YAML structure:
        transformation:
          train_transforms:
            - _target_: torchvision.transforms.RandomHorizontalFlip
              p: 0.5
            - _target_: torchvision.transforms.ToTensor
          val_transforms:
            - _target_: torchvision.transforms.ToTensor

    Usage:
        # Use default transforms
        python train.py dataset=cifar10 +transformation=train

        # Use heavy augmentation
        python train.py dataset=cifar10 +transformation=train_heavy

        # Override specific transform parameters
        python train.py +transformation=train \\
            transformation.train_transforms[0].p=0.7
    """
    data_dir = cfg.dataset.data_dir
    train_val_split = cfg.dataset.train_val_split
    seed = cfg.seed

    # Create transforms from Hydra config
    if "transformation" in cfg and "transforms" in cfg.transformation:
        # Single transforms config (used for both train and val)
        transform = create_transforms_from_config(cfg.transformation.transforms)
        train_transform = transform
        val_transform = transform
    elif "transformation" in cfg:
        # Separate train and val transforms
        if "train_transforms" in cfg.transformation:
            train_transform = create_transforms_from_config(
                cfg.transformation.train_transforms
            )
        else:
            raise ValueError("No train_transforms found in transformation config")

        if "val_transforms" in cfg.transformation:
            val_transform = create_transforms_from_config(
                cfg.transformation.val_transforms
            )
        else:
            raise ValueError("No val_transforms found in transformation config")
    else:
        raise ValueError("No transformation config provided")

    # Create datasets with transforms
    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # Split train into train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        train_val_split,
        seed,
    )

    # Create validation dataset with correct transforms
    val_dataset.dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset


def create_mnist_with_hydra_transforms(
    cfg: DictConfig,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create MNIST datasets using Hydra-instantiated transforms.

    Usage:
        python train.py dataset=mnist +transformation=mnist_train

    This will load the MNIST-specific transforms that handle:
    - Grayscale images (28x28)
    - Appropriate normalization (mean=0.1307, std=0.3081)
    - Minimal augmentations suitable for MNIST
    """
    data_dir = cfg.dataset.data_dir
    train_val_split = cfg.dataset.train_val_split
    seed = cfg.seed

    # Create transforms from Hydra config
    if "transformation" not in cfg:
        raise ValueError("No transformation config provided")

    if "train_transforms" in cfg.transformation and "val_transforms" in cfg.transformation:
        train_transform = create_transforms_from_config(
            cfg.transformation.train_transforms
        )
        val_transform = create_transforms_from_config(cfg.transformation.val_transforms)
    elif "transforms" in cfg.transformation:
        transform = create_transforms_from_config(cfg.transformation.transforms)
        train_transform = transform
        val_transform = transform
    else:
        raise ValueError("Invalid transformation config structure")

    # Create datasets
    full_train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # Split train into train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        train_val_split,
        seed,
    )

    # Create validation dataset with correct transforms
    val_dataset.dataset = datasets.MNIST(
        root=data_dir, train=True, download=False, transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset


# Example of how to add this to the dataset registry
"""
To use these new dataset factories, add them to padma/datasets/__init__.py:

from .cifar_hydra import (
    create_cifar10_with_hydra_transforms,
    create_mnist_with_hydra_transforms,
)

DATASET_REGISTRY = {
    # ... existing datasets ...
    "cifar10_hydra": create_cifar10_with_hydra_transforms,
    "mnist_hydra": create_mnist_with_hydra_transforms,
}

Then use them:
    python train.py dataset.name=cifar10_hydra +transformation=train_heavy
"""
