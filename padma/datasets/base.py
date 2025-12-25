from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from omegaconf import DictConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_transforms(cfg: DictConfig, is_training: bool = True) -> transforms.Compose:
    """
    Create data transformations based on configuration.

    Builds a complete transformation pipeline including:
    - Geometric transforms (resize, crop, flip, rotation)
    - Color transforms (jitter)
    - Auto-augmentation (RandAugment, AugMix, AutoAugment)
    - Normalization
    - Advanced transforms (random erasing)

    Args:
        cfg: Full configuration (expects cfg.dataset and cfg.transformation)
        is_training: Whether to create training or validation transforms

    Returns:
        Composed transforms pipeline
    """
    image_size = cfg.dataset.image_size
    transform_cfg = cfg.transformation.train if is_training else cfg.transformation.val
    norm_cfg = cfg.transformation.normalize

    transform_list = []

    # Grayscale conversion (for MNIST and similar datasets)
    if transform_cfg.get("grayscale_to_rgb", False):
        transform_list.append(transforms.Grayscale(num_output_channels=3))

    if is_training:
        # Training transforms
        if transform_cfg.get("random_crop", True):
            transform_list.append(
                transforms.RandomResizedCrop(
                    image_size,
                    scale=tuple(transform_cfg.get("random_crop_scale", [0.08, 1.0])),
                    ratio=tuple(transform_cfg.get("random_crop_ratio", [0.75, 1.333])),
                )
            )
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))

        if transform_cfg.get("horizontal_flip", True):
            transform_list.append(
                transforms.RandomHorizontalFlip(
                    p=transform_cfg.get("horizontal_flip_p", 0.5)
                )
            )

        if transform_cfg.get("vertical_flip", False):
            transform_list.append(
                transforms.RandomVerticalFlip(
                    p=transform_cfg.get("vertical_flip_p", 0.5)
                )
            )

        if transform_cfg.get("rotation", False):
            transform_list.append(
                transforms.RandomRotation(
                    degrees=transform_cfg.get("rotation_degrees", 15)
                )
            )

        if transform_cfg.get("color_jitter", False):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=transform_cfg.get("color_jitter_brightness", 0.4),
                    contrast=transform_cfg.get("color_jitter_contrast", 0.4),
                    saturation=transform_cfg.get("color_jitter_saturation", 0.4),
                    hue=transform_cfg.get("color_jitter_hue", 0.1),
                )
            )

        # Auto-augmentation (mutually exclusive)
        auto_augment = transform_cfg.get("auto_augment")
        if auto_augment:
            if auto_augment.lower() in ["rand", "randaugment"]:
                transform_list.append(
                    transforms.RandAugment(
                        num_ops=transform_cfg.get("rand_augment_num_ops", 2),
                        magnitude=transform_cfg.get("rand_augment_magnitude", 9),
                    )
                )
            elif auto_augment.lower() == "augmix":
                transform_list.append(transforms.AugMix())
            elif auto_augment.lower() == "autoaugment":
                transform_list.append(
                    transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET)
                )
    else:
        # Validation/test transforms
        resize_scale = transform_cfg.get("resize_scale", 1.14)
        if resize_scale != 1.0:
            transform_list.append(
                transforms.Resize(int(image_size * resize_scale))
            )
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))

        if transform_cfg.get("center_crop", True):
            transform_list.append(transforms.CenterCrop(image_size))

    # Convert to tensor
    transform_list.append(transforms.ToTensor())

    # Normalization (always included as part of transformation)
    if norm_cfg.get("enabled", True):
        mean = tuple(norm_cfg.get("mean", IMAGENET_DEFAULT_MEAN))
        std = tuple(norm_cfg.get("std", IMAGENET_DEFAULT_STD))
        transform_list.append(transforms.Normalize(mean=mean, std=std))

    # Random erasing (applied after normalization, training only)
    if is_training and transform_cfg.get("random_erasing", False):
        transform_list.append(
            transforms.RandomErasing(
                p=transform_cfg.get("random_erasing_p", 0.25),
                scale=tuple(transform_cfg.get("random_erasing_scale", [0.02, 0.33])),
                ratio=tuple(transform_cfg.get("random_erasing_ratio", [0.3, 3.3])),
            )
        )

    return transforms.Compose(transform_list)


def split_dataset(
    dataset: Dataset,
    split_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into train and validation sets.

    Args:
        dataset: Dataset to split
        split_ratio: Ratio of training data
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    return random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )


def create_dataloaders(
    cfg: DictConfig,
    train_dataset: Dataset,
    val_dataset: Dataset,
    test_dataset: Dataset = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for train, validation, and test datasets.

    Args:
        cfg: Configuration from Hydra
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset (optional)

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )

    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size * 2,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader


