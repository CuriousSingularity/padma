import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig

from .base import get_transforms


def create_imagenet_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ImageNet train, validation, and test datasets.

    Args:
        cfg: Configuration from Hydra

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = get_transforms(cfg, is_training=True)
    val_transform = get_transforms(cfg, is_training=False)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = val_dataset  # ImageNet uses val as test

    return train_dataset, val_dataset, test_dataset
