"""Utilities for creating PyTorch transforms from Hydra configuration.

This module provides functions to dynamically create torchvision.transforms.Compose
objects from Hydra configuration files using hydra.utils.instantiate.

Example usage:
    >>> from omegaconf import DictConfig
    >>> from padma.utils.transforms import create_transforms_from_config
    >>>
    >>> # Assuming cfg.transformation.train_transforms is loaded from YAML
    >>> train_transforms = create_transforms_from_config(cfg.transformation.train_transforms)
    >>> val_transforms = create_transforms_from_config(cfg.transformation.val_transforms)
"""

from typing import List, Optional

import hydra
from omegaconf import DictConfig, ListConfig
from torchvision import transforms


def create_transforms_from_config(
    transform_configs: ListConfig,
) -> transforms.Compose:
    """Create a torchvision.transforms.Compose object from Hydra config.

    Args:
        transform_configs: List of transform configurations from Hydra.
            Each config should have a '_target_' field specifying the transform class.

    Returns:
        A composed transform object ready to use with PyTorch datasets.

    Example:
        Given a YAML config like:
        ```yaml
        transforms:
          - _target_: torchvision.transforms.RandomResizedCrop
            size: 224
            scale: [0.08, 1.0]
          - _target_: torchvision.transforms.ToTensor
          - _target_: torchvision.transforms.Normalize
            mean: [0.485, 0.456, 0.406]
            std: [0.229, 0.224, 0.225]
        ```

        Usage:
        ```python
        transform = create_transforms_from_config(cfg.transforms)
        dataset = MyDataset(transform=transform)
        ```
    """
    transform_list = []

    for transform_cfg in transform_configs:
        # Use Hydra's instantiate to create the transform object
        # This automatically resolves the _target_ and passes the args
        transform = hydra.utils.instantiate(transform_cfg)
        transform_list.append(transform)

    return transforms.Compose(transform_list)


def create_transform_from_dict(
    transform_name: str, transform_args: Optional[dict] = None
) -> object:
    """Create a single transform from name and arguments.

    This is a fallback method for creating transforms without using Hydra configs.
    Useful for programmatic transform creation.

    Args:
        transform_name: Name of the transform class (e.g., 'RandomHorizontalFlip')
        transform_args: Dictionary of arguments to pass to the transform constructor

    Returns:
        An instantiated transform object

    Example:
        >>> from padma.utils.transforms import create_transform_from_dict
        >>> flip = create_transform_from_dict('RandomHorizontalFlip', {'p': 0.5})
        >>> crop = create_transform_from_dict('RandomResizedCrop', {'size': 224})
    """
    if transform_args is None:
        transform_args = {}

    # Get the transform class from torchvision.transforms
    transform_class = getattr(transforms, transform_name)
    return transform_class(**transform_args)


def create_transforms_from_list(transform_defs: List[dict]) -> transforms.Compose:
    """Create a Compose transform from a list of dictionaries.

    This is useful for creating transforms from non-Hydra configs (e.g., pure Python).

    Args:
        transform_defs: List of dictionaries with 'name' and optional 'args' keys

    Returns:
        A composed transform object

    Example:
        >>> transform_defs = [
        ...     {'name': 'RandomHorizontalFlip', 'args': {'p': 0.5}},
        ...     {'name': 'ToTensor'},
        ...     {'name': 'Normalize', 'args': {'mean': [0.5], 'std': [0.5]}}
        ... ]
        >>> transform = create_transforms_from_list(transform_defs)
    """
    transform_list = []

    for transform_def in transform_defs:
        transform_name = transform_def["name"]
        transform_args = transform_def.get("args", {})
        transform = create_transform_from_dict(transform_name, transform_args)
        transform_list.append(transform)

    return transforms.Compose(transform_list)


def get_transform_info(transform: transforms.Compose) -> str:
    """Get a human-readable string describing the composed transform.

    Args:
        transform: A torchvision.transforms.Compose object

    Returns:
        A formatted string listing all transforms in the composition

    Example:
        >>> transform = transforms.Compose([
        ...     transforms.RandomHorizontalFlip(p=0.5),
        ...     transforms.ToTensor()
        ... ])
        >>> print(get_transform_info(transform))
        Compose(
          RandomHorizontalFlip(p=0.5)
          ToTensor()
        )
    """
    return str(transform)
