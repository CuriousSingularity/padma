from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def normalize_timeseries(
    data: np.ndarray,
    method: str = 'zscore',
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
    min_val: Optional[np.ndarray] = None,
    max_val: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Normalize time series data.

    Args:
        data: Time series data of shape (n_samples, channels, timesteps) or (n_samples, timesteps)
        method: Normalization method ('zscore' or 'minmax')
        mean: Pre-computed mean for zscore (if None, computed from data)
        std: Pre-computed std for zscore (if None, computed from data)
        min_val: Pre-computed min for minmax (if None, computed from data)
        max_val: Pre-computed max for minmax (if None, computed from data)

    Returns:
        Tuple of (normalized_data, stats_dict)
        stats_dict contains normalization parameters for later use
    """
    if method == 'zscore':
        if mean is None:
            mean = np.mean(data, axis=0, keepdims=True)
        if std is None:
            std = np.std(data, axis=0, keepdims=True)
            std = np.where(std == 0, 1.0, std)  # Avoid division by zero

        normalized = (data - mean) / std
        stats = {'mean': mean, 'std': std}

    elif method == 'minmax':
        if min_val is None:
            min_val = np.min(data, axis=0, keepdims=True)
        if max_val is None:
            max_val = np.max(data, axis=0, keepdims=True)

        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)  # Avoid division by zero
        normalized = (data - min_val) / range_val
        stats = {'min': min_val, 'max': max_val}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, stats


def apply_jitter(data: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    """
    Add Gaussian noise to time series data for augmentation.

    Args:
        data: Time series data
        sigma: Standard deviation of Gaussian noise

    Returns:
        Augmented data
    """
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def apply_scaling(data: np.ndarray, sigma: float = 0.1) -> np.ndarray:
    """
    Apply random scaling to time series data for augmentation.

    Args:
        data: Time series data
        sigma: Standard deviation of scaling factor

    Returns:
        Scaled data
    """
    scaling_factor = np.random.normal(1.0, sigma)
    return data * scaling_factor


def apply_time_warp(data: np.ndarray, sigma: float = 0.2) -> np.ndarray:
    """
    Apply time warping augmentation (simple implementation).

    Args:
        data: Time series data of shape (..., timesteps)
        sigma: Warping strength

    Returns:
        Warped data
    """
    # Simple implementation: random smooth distortion
    # For production, consider using more sophisticated methods
    from scipy.interpolate import CubicSpline

    orig_steps = np.arange(data.shape[-1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=data.shape[-1])
    random_warps = np.cumsum(random_warps)
    random_warps = (random_warps - random_warps[0]) / (random_warps[-1] - random_warps[0])
    random_warps = random_warps * (data.shape[-1] - 1)

    # Handle both 2D and 3D data
    if data.ndim == 2:
        warped = np.zeros_like(data)
        for i in range(data.shape[0]):
            warper = CubicSpline(orig_steps, data[i])
            warped[i] = warper(random_warps)
    elif data.ndim == 3:
        warped = np.zeros_like(data)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                warper = CubicSpline(orig_steps, data[i, j])
                warped[i, j] = warper(random_warps)
    else:
        return data

    return warped


def split_timeseries_dataset(
    dataset: Dataset,
    split_ratio: float,
    seed: int,
) -> Tuple[Dataset, Dataset]:
    """
    Split a time series dataset into train and validation sets.

    Args:
        dataset: Time series dataset to split
        split_ratio: Ratio of training data (e.g., 0.9 for 90% train, 10% val)
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


class TimeSeriesAugmentation:
    """
    Composable augmentation pipeline for time series data.
    """

    def __init__(
        self,
        jitter: bool = False,
        jitter_sigma: float = 0.03,
        scaling: bool = False,
        scaling_sigma: float = 0.1,
        time_warp: bool = False,
        time_warp_sigma: float = 0.2,
    ):
        """
        Initialize augmentation pipeline.

        Args:
            jitter: Whether to apply jitter (Gaussian noise)
            jitter_sigma: Standard deviation for jitter
            scaling: Whether to apply random scaling
            scaling_sigma: Standard deviation for scaling factor
            time_warp: Whether to apply time warping
            time_warp_sigma: Warping strength
        """
        self.jitter = jitter
        self.jitter_sigma = jitter_sigma
        self.scaling = scaling
        self.scaling_sigma = scaling_sigma
        self.time_warp = time_warp
        self.time_warp_sigma = time_warp_sigma

    def __call__(self, data: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to data.

        Args:
            data: Time series data

        Returns:
            Augmented data
        """
        if self.jitter:
            data = apply_jitter(data, self.jitter_sigma)

        if self.scaling:
            data = apply_scaling(data, self.scaling_sigma)

        if self.time_warp:
            try:
                data = apply_time_warp(data, self.time_warp_sigma)
            except ImportError:
                # scipy not available, skip time warping
                pass

        return data
