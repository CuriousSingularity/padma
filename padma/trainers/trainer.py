import logging
import os
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

from padma.utils import MetricsTracker, get_device

logger = logging.getLogger(__name__)


class Trainer:
    """
    Generic trainer class for PyTorch models with TensorBoard logging.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg: DictConfig,
        test_loader: Optional[DataLoader] = None,
    ):
        """
        Initialize the trainer.

        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            cfg: Hydra configuration
            test_loader: Optional test data loader
        """
        self.cfg = cfg
        self.device = self._get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # Setup loss function
        self.criterion = self._create_criterion()

        # Setup mixed precision training
        self.scaler = GradScaler() if cfg.mixed_precision and self.device == "cuda" else None

        # Setup metrics
        self.train_metrics = MetricsTracker(
            num_classes=cfg.model.num_classes, device=self.device
        )
        self.val_metrics = MetricsTracker(
            num_classes=cfg.model.num_classes, device=self.device
        )

        # Setup TensorBoard logging
        self.writer = self._setup_tensorboard()

        # Setup checkpointing
        self.checkpoint_dir = Path(cfg.checkpoint.save_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = float("-inf") if cfg.checkpoint.monitor_mode == "max" else float("inf")

        # Training state
        self.current_epoch = 0
        self.global_step = 0

        # Early stopping
        self.early_stop_counter = 0

    def _get_device(self) -> str:
        """Get the device to use for training."""
        return get_device(self.cfg)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        opt_cfg = self.cfg.training.optimizer
        params = filter(lambda p: p.requires_grad, self.model.parameters())

        if opt_cfg.name.lower() == "adam":
            return torch.optim.Adam(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        elif opt_cfg.name.lower() == "adamw":
            return torch.optim.AdamW(
                params,
                lr=opt_cfg.lr,
                weight_decay=opt_cfg.weight_decay,
                betas=tuple(opt_cfg.betas),
            )
        elif opt_cfg.name.lower() == "sgd":
            return torch.optim.SGD(
                params,
                lr=opt_cfg.lr,
                momentum=opt_cfg.momentum,
                weight_decay=opt_cfg.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {opt_cfg.name}")

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on configuration."""
        sched_cfg = self.cfg.training.scheduler
        total_steps = len(self.train_loader) * self.cfg.training.epochs

        if sched_cfg.name.lower() == "cosine":
            warmup_steps = sched_cfg.warmup_epochs * len(self.train_loader)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=sched_cfg.min_lr,
            )
        elif sched_cfg.name.lower() == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.step_size,
                gamma=sched_cfg.gamma,
            )
        elif sched_cfg.name.lower() == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                patience=sched_cfg.patience,
                factor=sched_cfg.factor,
            )
        elif sched_cfg.name.lower() == "onecycle":
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg.training.optimizer.lr,
                total_steps=total_steps,
            )
        else:
            return None

    def _create_criterion(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_cfg = self.cfg.training.loss

        if loss_cfg.name.lower() == "cross_entropy":
            return nn.CrossEntropyLoss(label_smoothing=loss_cfg.label_smoothing)
        else:
            return nn.CrossEntropyLoss()

    def _setup_tensorboard(self) -> SummaryWriter:
        """Setup TensorBoard writer."""
        log_dir = self.cfg.logging.tensorboard_dir
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

        # Log configuration
        writer.add_text("config", OmegaConf.to_yaml(self.cfg))

        return writer

    def _warmup_lr(self, step: int, warmup_steps: int) -> None:
        """Apply linear warmup to learning rate."""
        if step < warmup_steps:
            lr_scale = (step + 1) / warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.cfg.training.optimizer.lr * lr_scale

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.train_metrics.reset()

        warmup_steps = self.cfg.training.scheduler.warmup_epochs * len(self.train_loader)
        accumulation_steps = self.cfg.training.accumulation_steps
        log_interval = self.cfg.logging.log_interval

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Warmup learning rate
            self._warmup_lr(self.global_step, warmup_steps)

            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets) / accumulation_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets) / accumulation_steps
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % accumulation_steps == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

                # Step scheduler (for step-based schedulers)
                if self.scheduler is not None and self.global_step >= warmup_steps:
                    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.scheduler.step()

            # Update metrics
            self.train_metrics.update(outputs.detach(), targets, loss.item() * accumulation_steps)

            # Log to TensorBoard
            if batch_idx % log_interval == 0:
                self.writer.add_scalar("train/batch_loss", loss.item() * accumulation_steps, self.global_step)
                self.writer.add_scalar("train/lr", self.optimizer.param_groups[0]["lr"], self.global_step)

            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        metrics = self.train_metrics.compute()

        # Log epoch metrics
        for name, value in metrics.items():
            self.writer.add_scalar(f"train/{name}", value, self.current_epoch)

        return metrics

    @torch.no_grad()
    def validate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.val_metrics.reset()

        loader = loader or self.val_loader

        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if self.scaler is not None:
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            self.val_metrics.update(outputs, targets, loss.item())

        metrics = self.val_metrics.compute()

        # Log validation metrics
        for name, value in metrics.items():
            self.writer.add_scalar(f"val/{name}", value, self.current_epoch)

        return metrics

    def save_checkpoint(self, metrics: Dict[str, float], is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "metrics": metrics,
            "config": OmegaConf.to_container(self.cfg),
        }

        if self.cfg.checkpoint.save_last:
            torch.save(checkpoint, self.checkpoint_dir / "last.pt")

        if is_best and self.cfg.checkpoint.save_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            logger.info(f"Saved best model with {self.cfg.checkpoint.monitor_metric}: {metrics.get(self.cfg.checkpoint.monitor_metric.replace('val_', ''), 0):.4f}")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self) -> Dict[str, Any]:
        """Run the full training loop."""
        logger.info(f"Starting training for {self.cfg.training.epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model: {self.cfg.model.name}")
        logger.info(f"Dataset: {self.cfg.dataset.name}")
        logger.info(f"TensorBoard logs: {self.cfg.logging.tensorboard_dir}")

        best_metrics = {}

        for epoch in range(self.cfg.training.epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Epoch {epoch + 1} - Train: {train_metrics}")

            # Validate
            val_metrics = self.validate()
            logger.info(f"Epoch {epoch + 1} - Val: {val_metrics}")

            # Step plateau scheduler
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics.get("loss", 0))

            # Check if best model
            monitor_key = self.cfg.checkpoint.monitor_metric.replace("val_", "")
            current_metric = val_metrics.get(monitor_key, 0)

            is_best = False
            if self.cfg.checkpoint.monitor_mode == "max":
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1
            else:
                if current_metric < self.best_metric:
                    self.best_metric = current_metric
                    is_best = True
                    self.early_stop_counter = 0
                else:
                    self.early_stop_counter += 1

            if is_best:
                best_metrics = val_metrics

            # Save checkpoint
            self.save_checkpoint(val_metrics, is_best)

            # Early stopping
            if self.cfg.training.early_stopping.enabled:
                if self.early_stop_counter >= self.cfg.training.early_stopping.patience:
                    logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                    break

        self.writer.close()

        return {
            "best_metrics": best_metrics,
            "final_epoch": self.current_epoch,
        }
