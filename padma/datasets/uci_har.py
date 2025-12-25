import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.request import urlretrieve
import zipfile

import numpy as np
import torch
from torch.utils.data import Dataset

from .timeseries_base import (
    normalize_timeseries,
    split_timeseries_dataset,
    TimeSeriesAugmentation,
)
from .ecg5000 import TimeSeriesDataset  # Reuse the same dataset class


def download_uci_har(data_dir: str) -> str:
    """
    Download UCI HAR dataset.

    Args:
        data_dir: Directory to save data

    Returns:
        Path to extracted dataset directory
    """
    base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/"
    filename = "UCI HAR Dataset.zip"

    data_path = Path(data_dir) / "UCI_HAR"
    data_path.mkdir(parents=True, exist_ok=True)

    zip_file = data_path / filename
    extracted_path = data_path / "UCI HAR Dataset"

    # Download zip file if not exists
    if not extracted_path.exists():
        if not zip_file.exists():
            print(f"Downloading UCI HAR dataset...")
            try:
                urlretrieve(base_url + filename, zip_file)
                print(f"Downloaded to {zip_file}")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to download UCI HAR dataset from {base_url + filename}. "
                    f"Error: {e}"
                )

        # Extract zip file
        print(f"Extracting UCI HAR dataset...")
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(data_path)
        print(f"Extracted to {extracted_path}")

    return str(extracted_path)


def load_uci_har_data(dataset_path: str, split: str = 'train') -> Tuple[np.ndarray, np.ndarray]:
    """
    Load UCI HAR data from files.

    UCI HAR dataset has 9 sensor channels (3 axes x 3 sensors):
    - Body accelerometer (x, y, z)
    - Gravity accelerometer (x, y, z)
    - Body gyroscope (x, y, z)

    Args:
        dataset_path: Path to UCI HAR Dataset directory
        split: Either 'train' or 'test'

    Returns:
        Tuple of (data, labels)
        data shape: (n_samples, 9, 128) - multivariate time series
        labels shape: (n_samples,)
    """
    dataset_path = Path(dataset_path)
    split_path = dataset_path / split

    # Read inertial signals (9 channels)
    inertial_signals_path = split_path / "Inertial Signals"

    signal_files = [
        "body_acc_x_{}.txt",
        "body_acc_y_{}.txt",
        "body_acc_z_{}.txt",
        "body_gyro_x_{}.txt",
        "body_gyro_y_{}.txt",
        "body_gyro_z_{}.txt",
        "total_acc_x_{}.txt",
        "total_acc_y_{}.txt",
        "total_acc_z_{}.txt",
    ]

    # Load each signal file
    signals = []
    for signal_file in signal_files:
        file_path = inertial_signals_path / signal_file.format(split)
        signal_data = np.loadtxt(file_path)
        signals.append(signal_data)

    # Stack signals: (9, n_samples, 128) -> (n_samples, 9, 128)
    data = np.stack(signals, axis=0)
    data = np.transpose(data, (1, 0, 2))

    # Load labels
    labels_file = split_path / f"y_{split}.txt"
    labels = np.loadtxt(labels_file, dtype=int)

    # Labels are 1-6, convert to 0-5
    labels = labels - 1

    return data, labels


def create_uci_har_dataset(
    data_dir: str = "./data",
    train_val_split: float = 0.9,
    normalize: bool = True,
    normalize_method: str = "zscore",
    train_augmentation: Optional[Dict] = None,
    seed: int = 42,
    **kwargs  # Absorb extra config params
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Create UCI HAR train, validation, and test datasets.

    UCI HAR is a multivariate time series dataset for human activity recognition.
    - 6 classes (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING)
    - 9 sensor channels (3 axes x 3 sensors: body_acc, body_gyro, total_acc)
    - 128 timesteps per sample
    - 7352 training samples, 2947 test samples

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
    # Download and extract data if needed
    dataset_path = download_uci_har(data_dir)

    # Load data
    train_data, train_labels = load_uci_har_data(dataset_path, split='train')
    test_data, test_labels = load_uci_har_data(dataset_path, split='test')

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

    print(f"UCI HAR dataset loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Data shape: (9, 128) - multivariate, 9 channels, 128 timesteps")
    print(f"  Num classes: 6")

    return train_dataset, val_dataset, test_dataset
