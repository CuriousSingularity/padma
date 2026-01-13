import os
from typing import Tuple, Union

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig, ListConfig

from padma.utils.transforms import create_transforms_from_config


def create_imagenet_dataset(
    data_dir: str,
    train_transforms: Union[list, ListConfig],
    val_transforms: Union[list, ListConfig],
    seed: int,
    name: str = None,  # Not used, but passed by config
    image_size: int = None,  # Not used, but passed by config
    train_val_split: float = None,  # Not used for ImageNet
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ImageNet train, validation, and test datasets.

    Args:
        data_dir: Directory containing ImageNet data with train/ and val/ subdirectories
        train_transforms: List of training transforms
        val_transforms: List of validation/test transforms
        seed: Random seed for reproducibility (not used for ImageNet)
        name: Dataset name (not used, for config compatibility)
        image_size: Target image size (not used, for config compatibility)
        train_val_split: Train/val split ratio (not used for ImageNet)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # Create transformation pipelines
    train_transform = create_transforms_from_config(train_transforms)
    val_transform = create_transforms_from_config(val_transforms)

    # Create datasets
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = val_dataset  # ImageNet uses val as test

    return train_dataset, val_dataset, test_dataset
