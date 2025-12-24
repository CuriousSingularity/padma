from typing import Dict, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from .base import split_dataset


def get_cifar_transforms(
    image_size: int,
    train_augmentation: Optional[Dict] = None,
    val_augmentation: Optional[Dict] = None,
    normalize: Optional[Dict] = None,
    is_training: bool = True
) -> transforms.Compose:
    """
    Create CIFAR-specific transforms.

    Args:
        image_size: Target image size
        train_augmentation: Training augmentation config
        val_augmentation: Validation augmentation config
        normalize: Normalization config
        is_training: Whether to create training transforms

    Returns:
        Composed transforms
    """
    mean = normalize.get("mean", IMAGENET_DEFAULT_MEAN) if normalize else IMAGENET_DEFAULT_MEAN
    std = normalize.get("std", IMAGENET_DEFAULT_STD) if normalize else IMAGENET_DEFAULT_STD

    if is_training:
        train_augmentation = train_augmentation or {}
        transform_list = []

        if train_augmentation.get("random_crop", True):
            transform_list.append(transforms.RandomResizedCrop(image_size))
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))

        if train_augmentation.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        if train_augmentation.get("color_jitter", False):
            transform_list.append(
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            )

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        return transforms.Compose(transform_list)
    else:
        val_augmentation = val_augmentation or {}
        transform_list = [transforms.Resize(int(image_size * 1.14))]

        if val_augmentation.get("center_crop", True):
            transform_list.append(transforms.CenterCrop(image_size))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        return transforms.Compose(transform_list)


def create_cifar10_dataset(
    data_dir: str = "./data",
    image_size: int = 224,
    train_val_split: float = 0.9,
    train_augmentation: Optional[Dict] = None,
    val_augmentation: Optional[Dict] = None,
    normalize: Optional[Dict] = None,
    seed: int = 42,
    **kwargs  # Absorb extra config params
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create CIFAR-10 train, validation, and test datasets.

    Args:
        data_dir: Directory for data storage
        image_size: Target image size
        train_val_split: Train/val split ratio
        train_augmentation: Training augmentation config
        val_augmentation: Validation augmentation config
        normalize: Normalization config
        seed: Random seed for splitting
        **kwargs: Additional config parameters (absorbed)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_transform = get_cifar_transforms(
        image_size, train_augmentation, val_augmentation, normalize, is_training=True
    )
    val_transform = get_cifar_transforms(
        image_size, train_augmentation, val_augmentation, normalize, is_training=False
    )

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


def create_cifar100_dataset(
    data_dir: str = "./data",
    image_size: int = 224,
    train_val_split: float = 0.9,
    train_augmentation: Optional[Dict] = None,
    val_augmentation: Optional[Dict] = None,
    normalize: Optional[Dict] = None,
    seed: int = 42,
    **kwargs  # Absorb extra config params
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create CIFAR-100 train, validation, and test datasets.

    Args:
        data_dir: Directory for data storage
        image_size: Target image size
        train_val_split: Train/val split ratio
        train_augmentation: Training augmentation config
        val_augmentation: Validation augmentation config
        normalize: Normalization config
        seed: Random seed for splitting
        **kwargs: Additional config parameters (absorbed)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_transform = get_cifar_transforms(
        image_size, train_augmentation, val_augmentation, normalize, is_training=True
    )
    val_transform = get_cifar_transforms(
        image_size, train_augmentation, val_augmentation, normalize, is_training=False
    )

    full_train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.CIFAR100(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # Split train into train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        train_val_split,
        seed,
    )

    # Create validation dataset with correct transforms
    val_dataset.dataset = datasets.CIFAR100(
        root=data_dir, train=True, download=False, transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset
