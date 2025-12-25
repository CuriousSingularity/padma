from typing import Any, Dict

import lightning as L
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection


class TimeSeriesClassificationModule(L.LightningModule):
    """
    PyTorch Lightning module for time series classification.
    """

    def __init__(self, model: nn.Module, cfg: DictConfig):
        """
        Initialize the Lightning module.

        Args:
            model: PyTorch time series model
            cfg: Hydra configuration
        """
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(ignore=["model"])

        # Loss function
        self.criterion = self._create_criterion()

        # Metrics
        num_classes = cfg.model.num_classes
        task = "multiclass"

        metric_collection = MetricCollection({
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "precision": Precision(task=task, num_classes=num_classes, average="macro"),
            "recall": Recall(task=task, num_classes=num_classes, average="macro"),
            "f1": F1Score(task=task, num_classes=num_classes, average="macro"),
        })

        self.train_metrics = metric_collection.clone(prefix="ts_train_")
        self.val_metrics = metric_collection.clone(prefix="ts_val_")
        self.test_metrics = metric_collection.clone(prefix="ts_test_")

        # Warmup configuration
        self.warmup_epochs = cfg.training.scheduler.get("warmup_epochs", 0)

    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_cfg = self.cfg.training.loss
        label_smoothing = loss_cfg.get("label_smoothing", 0.0)
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        """Shared step for training, validation, and testing."""
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.criterion(outputs, targets)

        # Get predictions
        preds = outputs.argmax(dim=1)

        # Update metrics
        if stage == "train":
            self.train_metrics.update(preds, targets)
        elif stage == "val":
            self.val_metrics.update(preds, targets)
        else:
            self.test_metrics.update(preds, targets)

        # Log loss
        self.log(f"ts_{stage}_loss", loss, on_step=(stage == "train"), on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def on_train_epoch_end(self) -> None:
        """Log training metrics at end of epoch."""
        metrics = self.train_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.train_metrics.reset()

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics at end of epoch."""
        metrics = self.val_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.val_metrics.reset()

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._shared_step(batch, "test")

    def on_test_epoch_end(self) -> None:
        """Log test metrics at end of epoch."""
        metrics = self.test_metrics.compute()
        self.log_dict(metrics, prog_bar=True)
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and learning rate scheduler."""
        opt_cfg = self.cfg.training.optimizer
        sched_cfg = self.cfg.training.scheduler

        # Create optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_cfg.name.lower() == "adam":
            optimizer = torch.optim.Adam(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        elif opt_cfg.name.lower() == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        elif opt_cfg.name.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params,
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

        # Create scheduler
        if sched_cfg.name.lower() == "cosine":
            # Use CosineAnnealingWarmRestarts with warmup if warmup_epochs > 0
            if self.warmup_epochs > 0:
                from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=0.01,
                    end_factor=1.0,
                    total_iters=self.warmup_epochs,
                )
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.training.epochs - self.warmup_epochs,
                    eta_min=sched_cfg.min_lr,
                )
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_epochs],
                )
            else:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.cfg.training.epochs,
                    eta_min=sched_cfg.min_lr,
                )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif sched_cfg.name.lower() == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }
        elif sched_cfg.name.lower() == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.checkpoint.get("monitor_mode", "max"),
                patience=sched_cfg.patience,
                factor=sched_cfg.factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": self.cfg.checkpoint.get("monitor_metric", "ts_val_accuracy"),
                    "interval": "epoch",
                },
            }
        elif sched_cfg.name.lower() == "onecycle":
            # OneCycleLR needs total steps, will be configured in trainer
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=opt_cfg.lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }
        else:
            return {"optimizer": optimizer}
