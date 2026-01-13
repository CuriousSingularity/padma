from typing import Optional

import lightning as L
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig


class TimeSeriesDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for time series classification.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the DataModule.

        Args:
            cfg: Hydra configuration
        """
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg.training.batch_size
        self.num_workers = cfg.num_workers

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Setup datasets for each stage.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict')
        """
        if stage == "fit" or stage is None:
            self.train_dataset, self.val_dataset, self.test_dataset = instantiate(
                self.cfg.dataset,
                seed=self.cfg.seed,
                _recursive_=False
            )

        if stage == "validate":
            if self.val_dataset is None:
                _, self.val_dataset, _ = instantiate(
                    self.cfg.dataset,
                    seed=self.cfg.seed,
                    _recursive_=False
                )

        if stage == "test":
            if self.test_dataset is None:
                _, _, self.test_dataset = instantiate(
                    self.cfg.dataset,
                    seed=self.cfg.seed,
                    _recursive_=False
                )

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        """Create test dataloader."""
        if self.test_dataset is None:
            return None
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def get_dataset_info(self) -> dict:
        """Get dataset information for logging."""
        info = {
            "dataset_name": self.cfg.dataset.name,
            "train_samples": len(self.train_dataset) if self.train_dataset else 0,
            "val_samples": len(self.val_dataset) if self.val_dataset else 0,
        }
        if self.test_dataset:
            info["test_samples"] = len(self.test_dataset)
        return info
