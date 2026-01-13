import os
from typing import Tuple, Union

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig, ListConfig

from padma.utils.transforms import create_transforms_from_config


def create_custom_dataset(
    data_dir: str,
    train_transforms: Union[list, ListConfig],
    val_transforms: Union[list, ListConfig],
    seed: int,
    name: str = None,  # Not used, but passed by config
    image_size: int = None,  # Not used, but passed by config
    train_val_split: float = None,  # Not used for custom datasets with predefined splits
) -> Tuple[Dataset, Dataset, Dataset]:
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
        data_dir: Root directory containing train/, val/, and optionally test/ subdirectories
        train_transforms: List of training transforms
        val_transforms: List of validation/test transforms
        seed: Random seed for reproducibility (not used for custom datasets)
        name: Dataset name (not used, for config compatibility)
        image_size: Target image size (not used, for config compatibility)
        train_val_split: Train/val split ratio (not used for custom datasets)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Create transformation pipelines
    train_transform = create_transforms_from_config(train_transforms)
    val_transform = create_transforms_from_config(val_transforms)

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    if os.path.exists(test_dir):
        test_dataset = datasets.ImageFolder(test_dir, transform=val_transform)
    else:
        test_dataset = val_dataset

    return train_dataset, val_dataset, test_dataset
