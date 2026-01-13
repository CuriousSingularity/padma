from typing import Tuple, Union

from torch.utils.data import Dataset
from torchvision import datasets
from omegaconf import DictConfig, ListConfig

from .base import split_dataset
from padma.utils.transforms import create_transforms_from_config


def create_mnist_dataset(
    data_dir: str,
    train_val_split: float,
    train_transforms: Union[list, ListConfig],
    val_transforms: Union[list, ListConfig],
    seed: int,
    name: str = None,  # Not used, but passed by config
    image_size: int = None,  # Not used, but passed by config
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create MNIST train, validation, and test datasets.

    Args:
        data_dir: Directory to download/store MNIST data
        train_val_split: Fraction of training data to use for training (rest for validation)
        train_transforms: List of training transforms
        val_transforms: List of validation/test transforms
        seed: Random seed for reproducibility
        name: Dataset name (not used, for config compatibility)
        image_size: Target image size (not used, for config compatibility)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Create transformation pipelines
    train_transform = create_transforms_from_config(train_transforms)
    val_transform = create_transforms_from_config(val_transforms)

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
