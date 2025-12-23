from typing import Dict, Optional
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, MetricCollection


class MetricsTracker:
    """
    Track and compute classification metrics during training and evaluation.
    """

    def __init__(self, num_classes: int, device: str = "cpu", task: str = "multiclass"):
        """
        Initialize metrics tracker.

        Args:
            num_classes: Number of classes for classification
            device: Device to use for metric computation
            task: Task type (binary, multiclass, multilabel)
        """
        self.num_classes = num_classes
        self.device = device
        self.task = task

        self.metrics = MetricCollection({
            "accuracy": Accuracy(task=task, num_classes=num_classes),
            "precision": Precision(task=task, num_classes=num_classes, average="macro"),
            "recall": Recall(task=task, num_classes=num_classes, average="macro"),
            "f1": F1Score(task=task, num_classes=num_classes, average="macro"),
        }).to(device)

        self.reset()

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.reset()
        self.total_loss = 0.0
        self.num_batches = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor, loss: Optional[float] = None) -> None:
        """
        Update metrics with new batch of predictions.

        Args:
            preds: Model predictions (logits or probabilities)
            targets: Ground truth labels
            loss: Optional batch loss
        """
        if preds.dim() > 1:
            preds = preds.argmax(dim=1)

        self.metrics.update(preds, targets)

        if loss is not None:
            self.total_loss += loss
            self.num_batches += 1

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.

        Returns:
            Dictionary of metric names and values
        """
        results = {k: v.item() for k, v in self.metrics.compute().items()}

        if self.num_batches > 0:
            results["loss"] = self.total_loss / self.num_batches

        return results

    def to(self, device: str) -> "MetricsTracker":
        """Move metrics to device."""
        self.device = device
        self.metrics = self.metrics.to(device)
        return self
