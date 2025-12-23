from typing import Tuple

from torch.utils.data import Dataset
from omegaconf import DictConfig

from .base import create_dataloaders
from .mnist import create_mnist_dataset
from .cifar import create_cifar10_dataset, create_cifar100_dataset
from .imagenet import create_imagenet_dataset
from .custom import create_custom_dataset


DATASET_REGISTRY = {
    "mnist": create_mnist_dataset,
    "cifar10": create_cifar10_dataset,
    "cifar100": create_cifar100_dataset,
    "imagenet": create_imagenet_dataset,
    "custom": create_custom_dataset,
}


def create_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create train, validation, and test datasets based on configuration.

    Args:
        cfg: Configuration from Hydra

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    dataset_name = cfg.dataset.name.lower()

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available datasets: {list(DATASET_REGISTRY.keys())}"
        )

    return DATASET_REGISTRY[dataset_name](cfg)


__all__ = ["create_dataset", "create_dataloaders", "DATASET_REGISTRY"]
