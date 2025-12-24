from typing import Dict, Optional, Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms

from .base import split_dataset


def get_mnist_transforms(
    image_size: int,
    train_augmentation: Optional[Dict] = None,
    is_training: bool = True
) -> transforms.Compose:
    """
    Create MNIST-specific transforms.

    Args:
        image_size: Target image size
        train_augmentation: Training augmentation config
        is_training: Whether to create training transforms

    Returns:
        Composed transforms
    """
    if is_training:
        train_augmentation = train_augmentation or {}
        transform_list = [
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for timm models
            transforms.Resize((image_size, image_size)),
        ]

        if train_augmentation.get("random_crop", False):
            transform_list.append(transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        return transforms.Compose(transform_list)
    else:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def create_mnist_dataset(
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
    Create MNIST train, validation, and test datasets.

    Args:
        data_dir: Directory for data storage
        image_size: Target image size
        train_val_split: Train/val split ratio
        train_augmentation: Training augmentation config
        val_augmentation: Validation augmentation config (unused for MNIST)
        normalize: Normalization config (unused, uses ImageNet defaults)
        seed: Random seed for splitting
        **kwargs: Additional config parameters (absorbed)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    train_transform = get_mnist_transforms(image_size, train_augmentation, is_training=True)
    val_transform = get_mnist_transforms(image_size, is_training=False)

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
