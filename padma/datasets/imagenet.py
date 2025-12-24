import os
from typing import Dict, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_imagenet_transforms(
    image_size: int,
    train_augmentation: Optional[Dict] = None,
    val_augmentation: Optional[Dict] = None,
    normalize: Optional[Dict] = None,
    is_training: bool = True
) -> transforms.Compose:
    """
    Create ImageNet-specific transforms.

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


def create_imagenet_dataset(
    data_dir: str = "./data",
    image_size: int = 224,
    train_augmentation: Optional[Dict] = None,
    val_augmentation: Optional[Dict] = None,
    normalize: Optional[Dict] = None,
    **kwargs  # Absorb extra config params (train_val_split, seed, etc.)
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ImageNet train, validation, and test datasets.

    Args:
        data_dir: Directory for data storage
        image_size: Target image size
        train_augmentation: Training augmentation config
        val_augmentation: Validation augmentation config
        normalize: Normalization config
        **kwargs: Additional config parameters (absorbed)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = get_imagenet_transforms(
        image_size, train_augmentation, val_augmentation, normalize, is_training=True
    )
    val_transform = get_imagenet_transforms(
        image_size, train_augmentation, val_augmentation, normalize, is_training=False
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    test_dataset = val_dataset  # ImageNet uses val as test

    return train_dataset, val_dataset, test_dataset
