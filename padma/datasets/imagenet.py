import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig

from padma.utils.transforms import create_transforms_from_config


def create_imagenet_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ImageNet train, validation, and test datasets.

    Args:
        cfg: Full Hydra configuration

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Create transformation pipelines
    train_transform = create_transforms_from_config(cfg.transformation.train_transforms)
    val_transform = create_transforms_from_config(cfg.transformation.val_transforms)

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = val_dataset  # ImageNet uses val as test

    return train_dataset, val_dataset, test_dataset
