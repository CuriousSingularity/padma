from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig

from .base import get_transforms, split_dataset


def create_cifar10_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create CIFAR-10 train, validation, and test datasets.

    Args:
        cfg: Configuration from Hydra

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir

    train_transform = get_transforms(cfg, is_training=True)
    val_transform = get_transforms(cfg, is_training=False)

    full_train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # Split train into train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        cfg.dataset.train_val_split,
        cfg.seed,
    )

    # Create validation dataset with correct transforms
    val_dataset.dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset


def create_cifar100_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create CIFAR-100 train, validation, and test datasets.

    Args:
        cfg: Configuration from Hydra

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir

    train_transform = get_transforms(cfg, is_training=True)
    val_transform = get_transforms(cfg, is_training=False)

    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # Split train into train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        cfg.dataset.train_val_split,
        cfg.seed,
    )

    # Create validation dataset with correct transforms
    val_dataset.dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset
