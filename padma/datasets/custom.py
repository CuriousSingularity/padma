import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig

from .base import get_transforms


def create_custom_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create custom ImageFolder-based train, validation, and test datasets.

    Expects directory structure:
        data_dir/
            train/
                class1/
                class2/
                ...
            val/
                class1/
                class2/
                ...
            test/ (optional)
                class1/
                class2/
                ...

    Args:
        cfg: Configuration from Hydra

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    train_transform = get_transforms(cfg, is_training=True)
    val_transform = get_transforms(cfg, is_training=False)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    else:
        test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset
