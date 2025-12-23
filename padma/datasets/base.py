from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from omegaconf import DictConfig
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_transforms(cfg: DictConfig, is_training: bool = True) -> transforms.Compose:
    """
    Create data transforms based on configuration.

    Args:
        cfg: Dataset configuration
        is_training: Whether to create training transforms

    Returns:
        Composed transforms
    """
    image_size = cfg.dataset.image_size
    mean = cfg.dataset.normalize.get("mean", IMAGENET_DEFAULT_MEAN)
    std = cfg.dataset.normalize.get("std", IMAGENET_DEFAULT_STD)

    if is_training:
        aug_cfg = cfg.dataset.train_augmentation
        transform_list = []

        if aug_cfg.get("random_crop", True):
            transform_list.append(transforms.RandomResizedCrop(image_size))
        else:
            transform_list.append(transforms.Resize((image_size, image_size)))

        if aug_cfg.get("horizontal_flip", True):
            transform_list.append(transforms.RandomHorizontalFlip())

        if aug_cfg.get("color_jitter", False):
            transform_list.append(
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
                )
            )

        if aug_cfg.get("auto_augment"):
            aa_policy = aug_cfg.auto_augment
            if aa_policy.startswith("rand"):
                transform_list.append(transforms.RandAugment())
            elif aa_policy == "augmix":
                transform_list.append(transforms.AugMix())

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        return transforms.Compose(transform_list)
    else:
        val_aug_cfg = cfg.dataset.val_augmentation
        transform_list = [transforms.Resize(int(image_size * 1.14))]

        if val_aug_cfg.get("center_crop", True):
            transform_list.append(transforms.CenterCrop(image_size))

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

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
