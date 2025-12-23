from typing import Tuple

from torch.utils.data import Dataset
from torchvision import datasets, transforms
from omegaconf import DictConfig

from .base import split_dataset


def get_mnist_transforms(cfg: DictConfig, is_training: bool = True) -> transforms.Compose:
    """
    Create MNIST-specific transforms.

    Args:
        cfg: Dataset configuration
        is_training: Whether to create training transforms

    Returns:
        Composed transforms
    """
    image_size = cfg.dataset.image_size

    if is_training:
        aug_cfg = cfg.dataset.train_augmentation
        transform_list = [
            transforms.Grayscale(num_output_channels=3),  # Convert to 3 channels for timm models
            transforms.Resize((image_size, image_size)),
        ]

        if aug_cfg.get("random_crop", False):
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


def create_mnist_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create MNIST train, validation, and test datasets.

    Args:
        cfg: Configuration from Hydra

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = cfg.dataset.data_dir

    train_transform = get_mnist_transforms(cfg, is_training=True)
    val_transform = get_mnist_transforms(cfg, is_training=False)

    full_train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=val_transform
    )

    # Split train into train/val
    train_dataset, val_dataset = split_dataset(
        full_train_dataset,
        cfg.dataset.train_val_split,
        cfg.seed,
    )

    # Create validation dataset with correct transforms
    val_dataset.dataset = datasets.MNIST(
        root=data_dir, train=True, download=False, transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset
