import os
from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig

from padma.utils.transforms import create_transforms_from_config


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
        cfg: Full Hydra configuration

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Create transformation pipelines
    train_transform = create_transforms_from_config(cfg.transformation.train_transforms)
    val_transform = create_transforms_from_config(cfg.transformation.val_transforms)

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    else:
        test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset
