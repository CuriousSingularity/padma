import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import Dataset

from .timeseries_base import (
    normalize_timeseries,
    split_timeseries_dataset,
    TimeSeriesAugmentation,
)


class TimeSeriesDataset(Dataset):
    """
    Generic time series dataset wrapper.
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        augmentation: Optional[TimeSeriesAugmentation] = None,
    ):
        """
        Initialize time series dataset.

        Args:
            data: Time series data of shape (n_samples, channels, timesteps) or (n_samples, timesteps)
            labels: Labels of shape (n_samples,)
            augmentation: Optional augmentation pipeline
        """
        self.data = data
        self.labels = labels
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (data_tensor, label_tensor)
        """
        data = self.data[idx].copy()
        label = self.labels[idx]

        # Apply augmentation if provided
        if self.augmentation is not None:
            data = self.augmentation(data)

        # Convert to tensors
        data_tensor = torch.FloatTensor(data)
        label_tensor = torch.LongTensor([label]).squeeze()

        return data_tensor, label_tensor


def download_ecg5000(data_dir: str) -> Tuple[str, str]:
    """
    Download ECG5000 dataset from UCR archive.

    Args:
        data_dir: Directory to save data

    Returns:
        Tuple of (train_file_path, test_file_path)
    """
    base_url = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000/"
    train_filename = "ECG5000_TRAIN.txt"
    test_filename = "ECG5000_TEST.txt"

    data_path = Path(data_dir) / "ECG5000"
    data_path.mkdir(parents=True, exist_ok=True)

    train_file = data_path / train_filename
    test_file = data_path / test_filename

    # Download train file if not exists
    if not train_file.exists():
        print(f"Downloading ECG5000 training data...")
        try:
            urlretrieve(base_url + train_filename, train_file)
            print(f"Downloaded to {train_file}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download ECG5000 training data from {base_url + train_filename}. "
                f"Error: {e}"
            )

    # Download test file if not exists
    if not test_file.exists():
        print(f"Downloading ECG5000 test data...")
        try:
            urlretrieve(base_url + test_filename, test_file)
            print(f"Downloaded to {test_file}")
        except Exception as e:
            raise RuntimeError(
                f"Failed to download ECG5000 test data from {base_url + test_filename}. "
                f"Error: {e}"
            )

    return str(train_file), str(test_file)


def load_ecg5000_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ECG5000 data from file.

    Args:
        file_path: Path to data file

    Returns:
        Tuple of (data, labels)
        data shape: (n_samples, 140) - univariate time series
        labels shape: (n_samples,)
    """
    # Load data (first column is label, rest are time series values)
    full_data = np.loadtxt(file_path)

    # Split into labels and data
    labels = full_data[:, 0].astype(int)
    data = full_data[:, 1:]

    # Labels in ECG5000 are 1-5, convert to 0-4
    labels = labels - 1

    # Add channel dimension for univariate series: (n_samples, timesteps) -> (n_samples, 1, timesteps)
    data = np.expand_dims(data, axis=1)

    return data, labels


def create_ecg5000_dataset(
    data_dir: str = "./data",
    train_val_split: float = 0.9,
    normalize: bool = True,
    normalize_method: str = "zscore",
    train_augmentation: Optional[Dict] = None,
    seed: int = 42,
    **kwargs  # Absorb extra config params
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create ECG5000 train, validation, and test datasets.

    ECG5000 is a univariate time series dataset for heartbeat classification.
    - 5 classes (different heartbeat types)
    - 140 timesteps per sample
    - ~500 training samples, ~4500 test samples

    Args:
        data_dir: Directory for data storage
        train_val_split: Train/val split ratio for the training set
        normalize: Whether to normalize the data
        normalize_method: Normalization method ('zscore' or 'minmax')
        train_augmentation: Training augmentation config (dict with keys: jitter, scaling, etc.)
        seed: Random seed for splitting
        **kwargs: Additional config parameters (absorbed)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    # Download data if needed
    train_file, test_file = download_ecg5000(data_dir)

    # Load data
    train_data, train_labels = load_ecg5000_data(train_file)
    test_data, test_labels = load_ecg5000_data(test_file)

    # Normalize if requested
    if normalize:
        # Compute statistics on training data
        train_data, stats = normalize_timeseries(train_data, method=normalize_method)

        # Apply same normalization to test data
        if normalize_method == 'zscore':
            test_data, _ = normalize_timeseries(
                test_data,
                method=normalize_method,
                mean=stats['mean'],
                std=stats['std']
            )
        else:  # minmax
            test_data, _ = normalize_timeseries(
                test_data,
                method=normalize_method,
                min_val=stats['min'],
                max_val=stats['max']
            )

    # Create augmentation pipeline for training
    augmentation = None
    if train_augmentation is not None:
        augmentation = TimeSeriesAugmentation(
            jitter=train_augmentation.get('jitter', False),
            jitter_sigma=train_augmentation.get('jitter_sigma', 0.03),
            scaling=train_augmentation.get('scaling', False),
            scaling_sigma=train_augmentation.get('scaling_sigma', 0.1),
            time_warp=train_augmentation.get('time_warp', False),
            time_warp_sigma=train_augmentation.get('time_warp_sigma', 0.2),
        )

    # Create full training dataset
    full_train_dataset = TimeSeriesDataset(
        data=train_data,
        labels=train_labels,
        augmentation=augmentation,
    )

    # Create test dataset (no augmentation)
    test_dataset = TimeSeriesDataset(
        data=test_data,
        labels=test_labels,
        augmentation=None,
    )

    # Split training data into train and validation
    train_dataset, val_dataset = split_timeseries_dataset(
        full_train_dataset,
        train_val_split,
        seed,
    )

    # Create validation dataset without augmentation
    val_dataset_no_aug = TimeSeriesDataset(
        data=train_data,
        labels=train_labels,
        augmentation=None,
    )

    # Update validation subset to use non-augmented dataset
    val_dataset.dataset = val_dataset_no_aug

    print(f"ECG5000 dataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Data shape: (1, 140) - univariate, 140 timesteps")
    print(f"  Num classes: 5")

    return train_dataset, val_dataset, test_dataset
